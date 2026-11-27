"""T57: run the out-of-fold Qini search for one dataset and target (ADR 0006).

Every trial is one MLflow run under the T41 conventions: experiment
``netlift-<dataset>-<target>``, run ``T57-<family>-trial-<n>``, tag ``task=T57``.
Each family's best trial gets a ``T57-<family>-study`` run, and the selection gets
``T57-champion-selection``. The summary is also written to
``reports/t57-<dataset>-<target>-search.json``.

Only the training split is read. Validation scores come later, from the T39
harness, after the champion is refitted.

Example:
    python scripts/run_uplift_search.py --dataset hillstrom --target visit
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast, get_args

from netlift.data.contracts import DatasetName, TargetOutcome
from netlift.data.loaders import CRITEO_PARQUET, HILLSTROM_CSV
from netlift.data.pipeline import load_splits
from netlift.models.uplift_search import (
    FAMILIES,
    Family,
    FamilyResult,
    SearchSettings,
    load_search_config,
    search_families,
    select_champion,
)
from netlift.utils.tracking import experiment_name, run_name, start_run

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
TASK = "T57"


def _slug(family: str) -> str:
    """Family name as an MLflow model name: 'uplift_forest' -> 'uplift-forest'."""
    return family.replace("_", "-")


def _data_file(dataset: str) -> Path:
    """The file whose SHA-256 is recorded with every run."""
    return CRITEO_PARQUET if dataset == "criteo" else HILLSTROM_CSV


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=get_args(DatasetName))
    parser.add_argument("--target", required=True, choices=get_args(TargetOutcome))
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Only for smoke runs; the recorded budget is the one in the config (ADR 0006).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the training split, run each family's study, log it and pick the champion."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    config = load_search_config()
    settings = SearchSettings.from_config(config, n_trials=args.n_trials)
    subsample = config["subsample_frac"][args.dataset]
    dataset = cast(DatasetName, args.dataset)
    target = cast(TargetOutcome, args.target)
    experiment = experiment_name(dataset, target)
    data_file = _data_file(dataset)
    common_tags = {
        "objective": "oof_qini",
        "split": "train",
        "subsample_frac": str(subsample),
        "n_folds": str(settings.n_folds),
        "n_trials_budget": str(settings.n_trials),
        "adr": "0006",
    }

    splits = load_splits(dataset, target, subsample_frac=subsample)
    logger.info(
        "training split: %d rows (subsample_frac=%s)", splits.train.n_rows, subsample
    )

    def log_trial(
        family: Family, number: int, params: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        with start_run(
            run_name(TASK, f"{_slug(family)}-trial-{number}"),
            task=TASK,
            dataset=data_file,
            seed=settings.seed,
            experiment=experiment,
            tags={**common_tags, "family": family, "trial": str(number)},
        ) as run:
            run.log_params(params)
            run.log_metrics(metrics)

    results = search_families(
        splits, tuple(cast(list[Family], args.families)), settings, log_trial=log_trial
    )
    for result in results:
        with start_run(
            run_name(TASK, f"{_slug(result.family)}-study"),
            task=TASK,
            dataset=data_file,
            seed=settings.seed,
            experiment=experiment,
            tags={
                **common_tags,
                "family": result.family,
                "best_trial": str(result.best_trial),
            },
        ) as run:
            run.log_params(result.best_params)
            run.log_metrics(
                {
                    "best_oof_qini": result.best_value,
                    "n_trials_completed": float(result.n_trials_completed),
                    "elapsed_seconds": result.elapsed_s,
                }
            )

    champion: FamilyResult = select_champion(results)
    with start_run(
        run_name(TASK, "champion-selection"),
        task=TASK,
        dataset=data_file,
        seed=settings.seed,
        experiment=experiment,
        tags={**common_tags, "champion_family": champion.family},
    ) as run:
        run.log_params({"champion_family": champion.family, **champion.best_params})
        run.log_metrics({"champion_oof_qini": champion.best_value})

    report = {
        "task": TASK,
        "adr": "0006",
        "dataset": dataset,
        "target": target,
        "train_rows": splits.train.n_rows,
        "subsample_frac": subsample,
        "n_folds": settings.n_folds,
        "n_trials_budget": settings.n_trials,
        "timeout_s": settings.timeout_s,
        "seed": settings.seed,
        "qini_negative_effect": settings.negative_effect,
        "families": [asdict(result) for result in results],
        "champion": champion.family,
        "result_status": "preliminary: out-of-fold on train; validation scores come from T39",
    }
    out = ROOT / "reports" / f"t57-{dataset}-{target}-search.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "champion: %s (oof_qini=%.6f); summary in %s",
        champion.family,
        champion.best_value,
        out,
    )


if __name__ == "__main__":
    main()
