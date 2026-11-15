"""Train and log S-, T-, and X-learners on Criteo conversion."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from netlift.data.pipeline import load_splits
from netlift.evaluation.contracts import fingerprint_evaluation_set
from netlift.models.meta_learners import (
    SLearner,
    TLearner,
    XLearner,
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
)
from netlift.utils.tracking import experiment_name, run_name, start_run

SEED = 42
PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": 4,
    "random_state": SEED,
    "verbosity": -1,
}


def main() -> None:
    """Fit on train, score validation, and save run artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("s", "t", "x", "all"), default="all")
    args = parser.parse_args()
    models = ("s", "t", "x") if args.model == "all" else (args.model,)
    dataset_path = Path("data/interim/criteo.parquet")
    splits = load_splits("criteo", target="conversion")
    train, validation = splits.train, splits.validation
    fingerprint = fingerprint_evaluation_set(
        validation.y, validation.t, row_ids=validation.row_id, dataset="criteo"
    )

    for name in models:
        label = f"{name}-learner"
        print(f"Training {label}...", flush=True)
        with start_run(
            run_name("T56", label),
            task="T56",
            dataset=dataset_path,
            seed=SEED,
            experiment=experiment_name("criteo", "conversion"),
            tags={"result_status": "preliminary", "split_name": "validation"},
        ) as run:
            model: SLearner | TLearner | XLearner
            if name == "s":
                model = fit_s_learner(train, PARAMS)
            elif name == "t":
                model = fit_t_learner(train, PARAMS)
            else:
                model = fit_x_learner(train, PARAMS, PARAMS)

            scores = model.predict_uplift(validation.X)
            if scores.shape != (validation.n_rows,) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid validation scores for {label}")
            if splits.test.is_open:
                raise RuntimeError("The test split must stay locked.")

            y, t = validation.y, validation.t
            metrics = {
                "preliminary_qini_coefficient": float(
                    qini_auc_score(y, scores, t, negative_effect=True)
                ),
                "preliminary_auuc": float(uplift_auc_score(y, scores, t)),
            }
            for percent in (10, 20, 30):
                metrics[f"preliminary_uplift_at_{percent}pct_overall"] = float(
                    uplift_at_k(y, scores, t, strategy="overall", k=percent / 100)
                )

            run.log_params(
                {
                    **PARAMS,
                    "model": label,
                    "train_rows": train.n_rows,
                    "validation_rows": validation.n_rows,
                    "train_treated_share": float(train.t.mean()),
                    "n_resamples": 0,
                    "evaluation_fingerprint": fingerprint,
                }
            )
            run.log_metrics(metrics)
            run.set_tags({"test_split_opened": "false"})
            with TemporaryDirectory() as tmp:
                folder = Path(tmp)
                predictions = folder / "validation_scores.csv"
                pd.DataFrame(
                    {
                        "row_id": validation.row_id,
                        "score": scores,
                    }
                ).to_csv(predictions, index=False)
                run.log_artifact(predictions)

                fitted = folder / "model.joblib"
                joblib.dump(model, fitted)
                run.log_artifact(fitted)

            print(f"{label}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
