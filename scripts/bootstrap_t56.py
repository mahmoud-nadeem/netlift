"""Bootstrap the saved T56 Criteo validation predictions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from netlift.data.pipeline import load_splits
from netlift.evaluation.contracts import fingerprint_evaluation_set

RUNS = {
    "S": "4042d3813ec34d08b3f5c14a19eb9c92",
    "T": "976c5ea630134d08abc526816eba16c1",
    "X": "1241bdc26a834f52b90e2a2d9f449708",
}
METRICS = ("qini", "auuc", "uplift_10", "uplift_20", "uplift_30")


def measure(y, score, t):
    return (
        qini_auc_score(y, score, t, negative_effect=True),
        uplift_auc_score(y, score, t),
        *(uplift_at_k(y, score, t, strategy="overall", k=k) for k in (0.1, 0.2, 0.3)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, required=True)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    val = load_splits("criteo", target="conversion").validation
    y, t = val.y, val.t
    fingerprint = fingerprint_evaluation_set(y, t, row_ids=val.row_id, dataset="criteo")
    ids = pd.Index(val.row_id)
    predictions = {}
    for name, run_id in RUNS.items():
        run = mlflow.get_run(run_id)
        if run.data.params.get("evaluation_fingerprint") != fingerprint:
            raise ValueError(f"{name}: validation fingerprint mismatch")
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="validation_scores.csv"
        )
        frame = pd.read_csv(path, float_precision="round_trip").set_index("row_id")
        if len(frame) != len(ids) or not frame.index.is_unique:
            raise ValueError(f"{name}: unexpected validation rows")
        score = frame.loc[ids, "score"].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError(f"{name}: invalid scores")
        predictions[name] = score

    rng = np.random.default_rng(42)
    control, treated = np.flatnonzero(t == 0), np.flatnonzero(t == 1)
    samples = {name: np.empty((args.resamples, len(METRICS))) for name in RUNS}
    start = perf_counter()
    for i in range(args.resamples):
        idx = np.concatenate(
            (
                rng.choice(control, size=control.size, replace=True),
                rng.choice(treated, size=treated.size, replace=True),
            )
        )
        rng.shuffle(idx)
        sample_y, sample_t = y[idx], t[idx]
        for name, score in predictions.items():
            samples[name][i] = measure(sample_y, score[idx], sample_t)
        if i < 3 or (i + 1) % 25 == 0:
            print(f"Bootstrap {i + 1}/{args.resamples}", flush=True)

    elapsed = perf_counter() - start
    print(f"Average seconds per resample: {elapsed / args.resamples:.2f}")
    if args.resamples < 1000:
        print("Timing check only.")
        return

    def intervals(values):
        return dict(zip(METRICS, np.percentile(values, [2.5, 97.5], axis=0).T.tolist()))

    result = {
        "split": "validation",
        "evaluation_fingerprint": fingerprint,
        "run_ids": RUNS,
        "seed": 42,
        "n_resamples": args.resamples,
        "confidence_level": 0.95,
        "intervals": {name: intervals(values) for name, values in samples.items()},
        "paired_difference_intervals": {
            f"{a} minus {b}": intervals(samples[a] - samples[b])
            for a, b in (("S", "T"), ("S", "X"), ("T", "X"))
        },
    }
    output = Path("../t56-bootstrap-results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
