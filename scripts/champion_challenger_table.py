"""T57: champion–challenger table on the Criteo validation split, with the T39 harness.

Scores the champion forest and the K1 T- and X-learners, and the response model when
its bundle exists, on the same validation rows with the same 1,000 within-arm
resamples. For each model it reports the Qini coefficient, AUUC and uplift@10/20/30%
(scikit-uplift, strategy ``overall``) with 95% percentile intervals. For each
challenger it reports the paired difference against the forest on the same
resamples (``paired_difference``).

Every score file is ``row_id`` + ``score`` and must cover exactly the rows of the
outcomes file. The locked test split is not read.

Outputs:

- ``reports/t57-champion-challenger.md`` and ``.json``: the tables, without the
  per-resample values;
- ``data/processed/t57-champion-challenger-results.json`` (git-ignored): every
  ``MetricResult`` with its resamples, so a later paired comparison can reuse them.

Example:
    python scripts/champion_challenger_table.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHAMPION = "uplift_forest"
RESPONSE_MODEL_RUN = "63e214a596574d16a67b467d8b272ebd"
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
        help=f"Response-model scores (MLflow run {RESPONSE_MODEL_RUN}); pending if absent.",
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
        "--out", type=Path, default=ROOT / "reports" / "t57-champion-challenger"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROCESSED / "t57-champion-challenger-results.json",
    )
    return parser.parse_args(argv)


def aligned_scores(
    path: Path, row_id: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Scores in the outcomes' row order; the two files must hold the same rows."""
    frame = pd.read_parquet(path)
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

    sources: dict[str, Path | None] = {
        CHAMPION: args.forest,
        "t_learner": args.t_learner,
        "x_learner": args.x_learner,
        "response_model": args.response_model,
    }
    scores = {
        name: aligned_scores(path, row_id)
        for name, path in sources.items()
        if path is not None
    }
    table = metric_table()
    results: dict[str, dict[str, MetricResult]] = {name: {} for name in scores}
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
            for metric_name in args.metrics
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "| Model | " + " | ".join(args.metrics) + " |"
    rule = "|---" * (len(args.metrics) + 1) + "|"
    lines = [
        "# T57 champion–challenger table (validation, Criteo conversion)",
        "",
        (
            f"Within-arm bootstrap, {args.n_resamples} resamples, seed {args.seed}, "
            f"{args.confidence_level:.0%} percentile intervals; scikit-uplift; uplift@k "
            "strategy `overall`. Produced by `scripts/champion_challenger_table.py`. "
            "Preliminary: validation only; the champion is confirmed at T70."
        ),
        "",
        "## Metrics",
        "",
        header,
        rule,
    ]
    for model_name in sources:
        if model_name in results:
            row = [cell(results[model_name][m]) for m in args.metrics]
        else:
            row = [f"pending (run {RESPONSE_MODEL_RUN[:8]}, bundle not received)"] + [
                "" for _ in args.metrics[1:]
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
            + " | ".join(cell(by_metric[m]) for m in args.metrics)
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
                model: {m: asdict(r) for m, r in by.items()}
                for model, by in results.items()
            }
        ),
        encoding="utf-8",
    )
    logger.info("tables written to %s.md and .json", args.out)


if __name__ == "__main__":
    main()
