"""T57 to T44: refit the K1 T- and X-learners on the full Criteo training split.

Reads ``reports/t57-criteo-conversion-search.json``, takes the T- and X-learner with
their best parameters, refits each on the full training split (no subsample) and
scores the full validation split. The one with the higher out-of-fold Qini is K1's
uplift model and the other is its sensitivity (kpis.yaml K1, ADR 0006).

Each model is one MLflow run under the T41 conventions (experiment
``netlift-criteo-conversion``, run ``T57-<family>-refit``, tag ``task=T57``) with:

- ``validation_scores.parquet``: ``row_id``, ``score`` for every validation row;
- ``validation_outcomes.parquet``: ``row_id``, ``y``, ``t``;
- ``manifest.json``: role, family, parameters, search provenance, ``tuned=true``,
  ``test_split_opened=false``.

The same files are written to ``data/processed/t57-k1/<family>/`` (git-ignored), and
``data/processed/t57-k1/k1-input.json`` names both runs. The script refuses to start
on a dirty working tree, and the test split is never opened.

Example:
    python scripts/refit_k1_models.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from netlift.data.loaders import CRITEO_PARQUET
from netlift.data.pipeline import load_splits
from netlift.models.k1_refit import (
    K1Choice,
    SearchedModel,
    choose_k1_models,
    read_search_report,
    refit_and_score,
)
from netlift.utils.errors import ContractError
from netlift.utils.tracking import (
    TrackingError,
    experiment_name,
    file_sha256,
    git_commit,
    git_dirty,
    run_name,
    start_run,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
TASK = "T57"
DEFAULT_REPORT = ROOT / "reports" / "t57-criteo-conversion-search.json"
DEFAULT_OUT = ROOT / "data" / "processed" / "t57-k1"
ROLES = ("k1_uplift_model", "k1_sensitivity")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def manifest_for(
    role: str,
    model: SearchedModel,
    choice: K1Choice,
    report_path: Path,
    train_rows: int,
    validation_rows: int,
) -> dict[str, Any]:
    """Everything T44 needs to know about where a score file came from."""
    return {
        "task": TASK,
        "role": role,
        "family": model.family,
        "dataset": "criteo",
        "target": "conversion",
        "split_name": "validation",
        "score_kind": "uplift",
        "tuned": True,
        "tuning": "ADR 0006: Optuna TPE on out-of-fold Qini inside the training split",
        "oof_qini": model.oof_qini,
        "best_trial": model.best_trial,
        "n_trials_completed": model.n_trials_completed,
        "n_trials_budget": choice.n_trials_budget,
        "search_report": str(report_path.relative_to(ROOT))
        if report_path.is_relative_to(ROOT)
        else str(report_path),
        "search_report_sha256": file_sha256(report_path),
        "params": model.params,
        "seed": choice.seed,
        "refit_on": "full training split, no subsample",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_split_opened": False,
        "status": "preliminary: validation intervals come from T39 in T44",
        "git_commit": git_commit(),
        "lightgbm_version": version("lightgbm"),
        "scikit_uplift_version": version("scikit-uplift"),
    }


def main(argv: list[str] | None = None) -> None:
    """Refit both K1 candidates, log them and write the local bundle."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    if git_dirty():
        raise TrackingError(
            "The working tree has uncommitted changes. Commit them first, so the K1 "
            "runs are tagged git_dirty=false and can be rebuilt from their commit."
        )
    choice = choose_k1_models(read_search_report(args.report))
    logger.info(
        "K1 uplift model: %s (oof_qini=%.6f); sensitivity: %s (oof_qini=%.6f)",
        choice.primary.family,
        choice.primary.oof_qini,
        choice.sensitivity.family,
        choice.sensitivity.oof_qini,
    )

    splits = load_splits("criteo", "conversion")
    logger.info(
        "full splits: train %d rows, validation %d rows",
        splits.train.n_rows,
        splits.validation.n_rows,
    )
    outcomes = splits.validation
    summary: dict[str, Any] = {
        "task": TASK,
        "search_report": str(args.report),
        "choice_rule": "higher out-of-fold Qini (ADR 0006); a tie goes to the T-learner",
        "test_split_opened": False,
        "runs": {},
    }
    for role, model in zip(ROLES, (choice.primary, choice.sensitivity), strict=True):
        folder = args.out_dir / model.family
        folder.mkdir(parents=True, exist_ok=True)
        slug = model.family.replace("_", "-")
        with start_run(
            run_name(TASK, f"{slug}-refit"),
            task=TASK,
            dataset=CRITEO_PARQUET,
            seed=choice.seed,
            experiment=experiment_name("criteo", "conversion"),
            tags={
                "k1_role": role,
                "family": model.family,
                "split_name": "validation",
                "test_split_opened": "false",
                "result_status": "preliminary",
                "adr": "0006",
            },
        ) as run:
            scores = refit_and_score(splits, model, seed=choice.seed)
            scores.to_parquet(folder / "validation_scores.parquet", index=False)
            outcomes_frame = {
                "row_id": outcomes.row_id,
                "y": outcomes.y,
                "t": outcomes.t,
            }
            pd.DataFrame(outcomes_frame).to_parquet(
                folder / "validation_outcomes.parquet", index=False
            )
            manifest = manifest_for(
                role,
                model,
                choice,
                args.report,
                splits.train.n_rows,
                splits.validation.n_rows,
            )
            (folder / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            run.log_params({"family": model.family, "k1_role": role, **model.params})
            run.log_metrics(
                {
                    "oof_qini": model.oof_qini,
                    "n_trials_completed": float(model.n_trials_completed),
                }
            )
            for name in (
                "validation_scores.parquet",
                "validation_outcomes.parquet",
                "manifest.json",
            ):
                run.log_artifact(folder / name)
            active = mlflow.active_run()
            if active is None:
                raise TrackingError("No active MLflow run while logging the refit.")
            summary["runs"][role] = {
                "run_id": active.info.run_id,
                "family": model.family,
                "oof_qini": model.oof_qini,
                "local_bundle": str(folder),
                "searched": asdict(model),
            }
            logger.info(
                "%s: %s logged as run %s", role, model.family, active.info.run_id
            )

    if splits.test.is_open:
        raise ContractError("The locked test split was opened during the refit.")
    out = args.out_dir / "k1-input.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("K1 input written to %s", out)


if __name__ == "__main__":
    main()
