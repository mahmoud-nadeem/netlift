"""T57: champion–challenger table on the Criteo validation split, with the T39 harness.

Scores the champion forest, the K1 T- and X-learners and the tuned response model on
the same validation rows with the same 1,000 within-arm resamples. For each model it reports the Qini coefficient, AUUC and uplift@10/20/30%
(scikit-uplift, strategy ``overall``) with 95% percentile intervals. For each
challenger it reports the paired difference against the forest on the same
resamples (``paired_difference``).

Every score file is ``row_id`` + ``score`` and must cover exactly the rows of the
outcomes file. The locked test split is not read.

The response model comes from Mohammed's K1 bundle (MLflow run ``63e214a5…``,
``data/processed/criteo-conversion-log_loss.zip``, git-ignored). The script reads its
``validation_scores.csv`` straight from the zip after checking the zip's SHA-256, the
run ID and the manifest (tuned, log-loss objective, 30 trials, split seed 42, test
split not opened). Without the bundle the response model's row says "pending".

Outputs:

- ``reports/t57-champion-challenger.md`` and ``.json``: the tables, without the
  per-resample values;
- ``data/processed/t57-champion-challenger-results.json`` (git-ignored): every
  ``MetricResult`` with its resamples, so a later paired comparison can reuse them.

Every column records the commit that computed it and whether the tree was dirty.
``--metrics uplift_at_10 uplift_at_20 uplift_at_30 --keep-other-metrics`` recomputes
only those columns and reuses the others from the results file, after checking that
it holds the same inputs (by SHA-256) and the same bootstrap settings.

Example:
    python scripts/champion_challenger_table.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from netlift.evaluation.contracts import MetricResult, paired_difference
from netlift.evaluation.metrics import auuc, qini_coefficient, uplift_at_k
from netlift.utils.errors import ContractError
from netlift.utils.tracking import file_sha256, git_commit, git_dirty

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHAMPION = "uplift_forest"
RESPONSE_MODEL_RUN = "63e214a596574d16a67b467d8b272ebd"
RESPONSE_BUNDLE = PROCESSED / "criteo-conversion-log_loss.zip"
RESPONSE_BUNDLE_SHA256 = (
    "2b41f120a4e7c6dbd899a4e9e87241ff276c0bdf188460fd639e078710ae3363"
)
RESPONSE_MANIFEST = {
    "tuned": True,
    "objective": "log_loss",
    "n_trials": 30,
    "test_split_opened": False,
    "split_seed": 42,
    "dataset": "criteo",
    "target": "conversion",
    "split_name": "validation",
}
K_VALUES = (0.1, 0.2, 0.3)

Metric = Callable[..., MetricResult]


def metric_table() -> dict[str, tuple[Metric, dict[str, Any]]]:
    """Every metric in the table, with its fixed settings."""
    table: dict[str, tuple[Metric, dict[str, Any]]] = {
        "qini": (qini_coefficient, {}),
        "auuc": (auuc, {}),
    }
    for k in K_VALUES:
        table[f"uplift_at_{round(k * 100)}"] = (
            uplift_at_k,
            {"k": k, "strategy": "overall"},
        )
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=PROCESSED / "t57-champion" / "validation_outcomes.parquet",
    )
    parser.add_argument(
        "--forest",
        type=Path,
        default=PROCESSED / "t57-champion" / "validation_scores.parquet",
    )
    parser.add_argument(
        "--t-learner",
        type=Path,
        default=PROCESSED / "t57-k1" / "t_learner" / "validation_scores.parquet",
    )
    parser.add_argument(
        "--x-learner",
        type=Path,
        default=PROCESSED / "t57-k1" / "x_learner" / "validation_scores.parquet",
    )
    parser.add_argument(
        "--response-model",
        type=Path,
        default=None,
        help=(
            f"The response model's K1 bundle (.zip, MLflow run {RESPONSE_MODEL_RUN}) "
            "or a row_id + score .parquet file. Default: the bundle in data/processed "
            "if it is there, otherwise the row says pending."
        ),
    )
    parser.add_argument(
        "--response-bundle-sha256",
        default=RESPONSE_BUNDLE_SHA256,
        help="Expected SHA-256 of the response-model bundle.",
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(metric_table()),
        default=list(metric_table()),
    )
    parser.add_argument(
        "--keep-other-metrics",
        action="store_true",
        help=(
            "Reuse the metrics not named in --metrics from --results instead of "
            "recomputing them; their inputs and bootstrap settings must match."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports" / "t57-champion-challenger"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROCESSED / "t57-champion-challenger-results.json",
    )
    return parser.parse_args(argv)


def bundle_member(archive: zipfile.ZipFile, name: str) -> str:
    """The one entry of the archive whose file name is ``name``."""
    matches = [entry for entry in archive.namelist() if Path(entry).name == name]
    if len(matches) != 1:
        raise ContractError(
            f"The bundle must hold exactly one {name}; found {matches}."
        )
    return matches[0]


def response_bundle_scores(path: Path, expected_sha256: str) -> pd.DataFrame:
    """The response model's validation scores, after checking the bundle's provenance."""
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ContractError(
            f"{path} has SHA-256 {digest}, not the checked bundle's {expected_sha256}."
        )
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read(bundle_member(archive, "run-metadata.json")))
        manifest = json.loads(archive.read(bundle_member(archive, "manifest.json")))
        run_id = metadata.get("info", {}).get("run_id")
        if run_id != RESPONSE_MODEL_RUN:
            raise ContractError(
                f"The bundle is run {run_id!r}, not the response model {RESPONSE_MODEL_RUN}."
            )
        wrong = {
            key: manifest.get(key)
            for key, value in RESPONSE_MANIFEST.items()
            if manifest.get(key) != value
        }
        if wrong:
            raise ContractError(f"The bundle's manifest differs from K1's: {wrong}.")
        with archive.open(bundle_member(archive, "validation_scores.csv")) as handle:
            return pd.read_csv(handle)


def read_scores(path: Path, expected_sha256: str) -> pd.DataFrame:
    """A row_id + score frame from a .parquet file or the response model's .zip bundle."""
    if path.suffix == ".zip":
        return response_bundle_scores(path, expected_sha256)
    return pd.read_parquet(path)


def aligned_scores(
    path: Path,
    row_id: npt.NDArray[np.int64],
    expected_sha256: str = RESPONSE_BUNDLE_SHA256,
) -> npt.NDArray[np.float64]:
    """Scores in the outcomes' row order; the two files must hold the same rows."""
    frame = read_scores(path, expected_sha256)
    if list(frame.columns) != ["row_id", "score"]:
        raise ContractError(f"{path} must have exactly the columns row_id and score.")
    if frame["row_id"].duplicated().any():
        raise ContractError(f"{path} has duplicate row_id values.")
    indexed = frame.set_index("row_id")["score"]
    if len(indexed) != len(row_id) or not indexed.index.isin(row_id).all():
        raise ContractError(f"{path} does not cover exactly the validation rows.")
    score = indexed.reindex(row_id).to_numpy(dtype=np.float64)
    if not np.isfinite(score).all():
        raise ContractError(f"{path} has non-finite scores.")
    return score


def cell(result: MetricResult) -> str:
    """'value [low, high]' with six decimals."""
    return f"{result.value:.6f} [{result.ci_low:.6f}, {result.ci_high:.6f}]"


def metric_result(data: dict[str, Any]) -> MetricResult:
    """Rebuild a MetricResult saved with ``dataclasses.asdict``."""
    return MetricResult(**{**data, "resamples": tuple(data["resamples"])})


def kept_metrics(
    path: Path,
    computed: list[str],
    models: list[str],
    inputs: dict[str, dict[str, str]],
    bootstrap: dict[str, Any],
) -> tuple[dict[str, dict[str, MetricResult]], dict[str, dict[str, Any]]]:
    """The saved metrics not recomputed now, after checking they fit this run."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("inputs") != inputs or saved.get("bootstrap") != bootstrap:
        raise ContractError(
            f"{path} was computed from other inputs or bootstrap settings; "
            "recompute every metric instead of keeping any."
        )
    kept: dict[str, dict[str, MetricResult]] = {model: {} for model in models}
    provenance: dict[str, dict[str, Any]] = {}
    for metric_name, origin in saved["provenance"].items():
        if metric_name in computed:
            continue
        for model in models:
            kept[model][metric_name] = metric_result(
                saved["results"][model][metric_name]
            )
        provenance[metric_name] = origin
    return kept, provenance


def summary(result: MetricResult) -> dict[str, Any]:
    """A MetricResult without its per-resample values."""
    data = asdict(result)
    data.pop("resamples")
    return data


def main(argv: list[str] | None = None) -> None:
    """Compute every metric for every model, the paired differences, and write the tables."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    outcomes = pd.read_parquet(args.outcomes)
    row_id = outcomes["row_id"].to_numpy(dtype=np.int64)
    y = outcomes["y"].to_numpy()
    t = outcomes["t"].to_numpy()

    response_model = args.response_model
    if response_model is None and RESPONSE_BUNDLE.exists():
        response_model = RESPONSE_BUNDLE
    if response_model is None:
        logger.warning(
            "No response-model bundle at %s; its row says pending.", RESPONSE_BUNDLE
        )
    sources: dict[str, Path | None] = {
        CHAMPION: args.forest,
        "t_learner": args.t_learner,
        "x_learner": args.x_learner,
        "response_model": response_model,
    }
    scores = {
        name: aligned_scores(path, row_id, args.response_bundle_sha256)
        for name, path in sources.items()
        if path is not None
    }
    inputs = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in sources.items()
        if path is not None
    }
    table = metric_table()
    bootstrap = {
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "confidence_level": args.confidence_level,
    }
    origin = {"git_commit": git_commit(), "git_dirty": git_dirty()}
    provenance: dict[str, dict[str, Any]] = {m: origin for m in args.metrics}
    results: dict[str, dict[str, MetricResult]] = {name: {} for name in scores}
    if args.keep_other_metrics:
        kept, kept_provenance = kept_metrics(
            args.results, list(args.metrics), list(scores), inputs, bootstrap
        )
        for model_name, by_metric in kept.items():
            results[model_name].update(by_metric)
        provenance.update(kept_provenance)
    columns = [m for m in table if m in provenance]
    timings: dict[str, float] = {}
    for metric_name in args.metrics:
        function, settings = table[metric_name]
        for model_name, score in scores.items():
            started = time.perf_counter()
            results[model_name][metric_name] = function(
                y,
                score,
                t,
                n_resamples=args.n_resamples,
                confidence_level=args.confidence_level,
                seed=args.seed,
                split_name="validation",
                row_ids=row_id,
                dataset="criteo",
                **settings,
            )
            timings[f"{model_name}/{metric_name}"] = time.perf_counter() - started
            logger.info(
                "%s %s: %s in %.0f s",
                model_name,
                metric_name,
                cell(results[model_name][metric_name]),
                timings[f"{model_name}/{metric_name}"],
            )

    differences: dict[str, dict[str, MetricResult]] = {}
    for model_name in results:
        if model_name == CHAMPION:
            continue
        differences[model_name] = {
            metric_name: paired_difference(
                results[model_name][metric_name], results[CHAMPION][metric_name]
            )
            for metric_name in columns
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "| Model | " + " | ".join(columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    ties = {
        str(r.settings.get("ties", "first rows after a stable sort"))
        for by in results.values()
        for r in by.values()
        if r.name == "uplift_at_k"
    }
    lines = [
        "# T57 champion–challenger table (validation, Criteo conversion)",
        "",
        (
            f"Within-arm bootstrap, {args.n_resamples} resamples, seed {args.seed}, "
            f"{args.confidence_level:.0%} percentile intervals; scikit-uplift; uplift@k "
            f"strategy `overall`, ties at the cut: {', '.join(sorted(ties)) or 'n/a'} "
            "(ADR 0007). Produced by `scripts/champion_challenger_table.py`. "
            "Preliminary: validation only; the champion is confirmed at T70."
        ),
        "",
        "Code (commit that computed each column):",
        "",
        *[
            f"- {m}: `{o['git_commit']}`" + (" (dirty tree)" if o["git_dirty"] else "")
            for m, o in provenance.items()
        ],
        "",
        "Inputs (SHA-256):",
        "",
        *[f"- {name}: `{item['sha256']}`" for name, item in inputs.items()],
        "",
        "## Metrics",
        "",
        header,
        rule,
    ]
    for model_name in sources:
        if model_name in results:
            row = [cell(results[model_name][m]) for m in columns]
        else:
            row = [f"pending (run {RESPONSE_MODEL_RUN[:8]}, bundle not received)"] + [
                "" for _ in columns[1:]
            ]
        lines.append(f"| {model_name} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Paired difference against the forest (challenger − forest)",
        "",
        header,
        rule,
    ]
    for model_name, by_metric in differences.items():
        lines.append(
            f"| {model_name} | "
            + " | ".join(cell(by_metric[m]) for m in columns)
            + " |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "metrics": {
                    model: {m: summary(r) for m, r in by.items()}
                    for model, by in results.items()
                },
                "differences_vs_forest": {
                    model: {m: summary(r) for m, r in by.items()}
                    for model, by in differences.items()
                },
                "pending": [name for name in sources if name not in results],
                "inputs": inputs,
                "bootstrap": bootstrap,
                "provenance": provenance,
                "response_model_run": RESPONSE_MODEL_RUN,
                "seconds": {key: round(value, 1) for key, value in timings.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(
            {
                "inputs": inputs,
                "bootstrap": bootstrap,
                "provenance": provenance,
                "results": {
                    model: {m: asdict(r) for m, r in by.items()}
                    for model, by in results.items()
                },
            }
        ),
        encoding="utf-8",
    )
    logger.info("tables written to %s.md and .json", args.out)


if __name__ == "__main__":
    main()
