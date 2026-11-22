"""T57: refit the champion on the full Criteo training split and register it.

Reads ``reports/t57-criteo-conversion-search.json`` and takes the family with the
best out-of-fold Qini (ADR 0006). Refits it with its best parameters on the full
training split (no subsample), scores the full validation split, and logs one
MLflow run under the T41 conventions (experiment ``netlift-criteo-conversion``,
run ``T57-<family>-champion``, tag ``task=T57``) with:

- ``validation_scores.parquet`` (``row_id``, ``score``),
  ``validation_outcomes.parquet`` (``row_id``, ``y``, ``t``) and ``manifest.json``;
- the model itself, as an MLflow ``pyfunc`` model whose ``predict`` returns uplift.

The model is registered as ``netlift-criteo-conversion-champion`` with the alias
``champion``. The version is tagged with the run ID, the search summary's SHA-256
and the commit. The same files go to ``data/processed/t57-champion/`` (git-ignored).
The script refuses a dirty tree and never opens the test split.

Example:
    python scripts/refit_champion.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from netlift.data.loaders import CRITEO_PARQUET
from netlift.data.pipeline import load_splits
from netlift.models.champion import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    UpliftPyfunc,
    choose_champion,
    save_uplift_model,
)
from netlift.models.k1_refit import read_search_report
from netlift.models.uplift_search import fit_model
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
DEFAULT_OUT = ROOT / "data" / "processed" / "t57-champion"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=ROOT / "requirements.txt",
        help="Pinned requirements recorded with the MLflow model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Refit, score validation, log, register and write the local bundle."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    if git_dirty():
        raise TrackingError(
            "The working tree has uncommitted changes. Commit them first, so the "
            "champion run is tagged git_dirty=false and can be rebuilt from its commit."
        )
    report = read_search_report(args.report)
    champion = choose_champion(report)
    report_sha = file_sha256(args.report)
    commit = git_commit()
    seed = int(report["seed"])
    logger.info(
        "champion: %s, trial %d, oof_qini=%.6f",
        champion.family,
        champion.best_trial,
        champion.oof_qini,
    )

    splits = load_splits("criteo", "conversion")
    train, validation = splits.train, splits.validation
    logger.info("full splits: train %d, validation %d", train.n_rows, validation.n_rows)

    folder = args.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    slug = champion.family.replace("_", "-")
    with start_run(
        run_name(TASK, f"{slug}-champion"),
        task=TASK,
        dataset=CRITEO_PARQUET,
        seed=seed,
        experiment=experiment_name("criteo", "conversion"),
        tags={
            "role": "t57_champion",
            "family": champion.family,
            "split_name": "validation",
            "test_split_opened": "false",
            "result_status": "preliminary",
            "search_report_sha256": report_sha,
            "adr": "0006",
        },
    ) as run:
        started = time.perf_counter()
        model = fit_model(
            champion.family, champion.params, train.X, train.t, train.y, seed=seed
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        score = model.predict_uplift(validation.X)
        predict_seconds = time.perf_counter() - started
        scores = pd.DataFrame({"row_id": validation.row_id, "score": score})
        if not scores["score"].notna().all() or len(scores) != validation.n_rows:
            raise ContractError("Expected one finite score per validation row.")
        scores.to_parquet(folder / "validation_scores.parquet", index=False)
        pd.DataFrame(
            {"row_id": validation.row_id, "y": validation.y, "t": validation.t}
        ).to_parquet(folder / "validation_outcomes.parquet", index=False)
        model_path = save_uplift_model(model, folder)

        active = mlflow.active_run()
        if active is None:
            raise TrackingError("No active MLflow run while logging the champion.")
        run_id = active.info.run_id
        manifest: dict[str, Any] = {
            "task": TASK,
            "role": "t57_champion",
            "family": champion.family,
            "dataset": "criteo",
            "target": "conversion",
            "split_name": "validation",
            "score_kind": "uplift",
            "tuned": True,
            "oof_qini": champion.oof_qini,
            "best_trial": champion.best_trial,
            "n_trials_completed": champion.n_trials_completed,
            "n_trials_budget": int(report["n_trials_budget"]),
            "search_report_sha256": report_sha,
            "params": champion.params,
            "seed": seed,
            "refit_on": "full training split, no subsample",
            "train_rows": train.n_rows,
            "validation_rows": validation.n_rows,
            "fit_seconds": round(fit_seconds, 1),
            "predict_seconds": round(predict_seconds, 1),
            "test_split_opened": False,
            "git_commit": commit,
            "mlflow_run_id": run_id,
            "causalml_version": version("causalml"),
            "scikit_uplift_version": version("scikit-uplift"),
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        run.log_params({"family": champion.family, **champion.params})
        run.log_metrics(
            {
                "oof_qini": champion.oof_qini,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        for name in (
            "validation_scores.parquet",
            "validation_outcomes.parquet",
            "manifest.json",
        ):
            run.log_artifact(folder / name)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=UpliftPyfunc(),
            artifacts={"uplift_model": str(model_path)},
            input_example=validation.X.head(5),
            pip_requirements=str(args.requirements),
        )

    registered = mlflow.register_model(info.model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.MlflowClient()
    for key, value in {
        "task": TASK,
        "family": champion.family,
        "run_id": run_id,
        "search_report_sha256": report_sha,
        "git_commit": commit,
        "oof_qini": f"{champion.oof_qini:.6f}",
        "best_trial": str(champion.best_trial),
        "split_scored": "validation",
        "test_split_opened": "false",
    }.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, registered.version, key, value
        )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, registered.version
    )

    if splits.test.is_open:
        raise ContractError("The locked test split was opened during the refit.")
    summary = {
        "task": TASK,
        "champion": asdict(champion),
        "mlflow_run_id": run_id,
        "registered_model": REGISTERED_MODEL_NAME,
        "version": str(registered.version),
        "alias": CHAMPION_ALIAS,
        "model_uri": f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}",
        "git_commit": commit,
        "search_report_sha256": report_sha,
        "test_split_opened": False,
    }
    (folder / "champion.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "registered %s version %s as @%s (run %s)",
        REGISTERED_MODEL_NAME,
        registered.version,
        CHAMPION_ALIAS,
        run_id,
    )


if __name__ == "__main__":
    main()
