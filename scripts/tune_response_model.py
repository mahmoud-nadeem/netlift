"""Tune the response-model baseline inside the train folds (#107, ADR 0004).

Criteo conversion is the primary target; Hillstrom visit and conversion are the
sensitivity leg. Selection uses out-of-fold scores on the train split only. The
final model is refitted on the full train split and scored on the full validation
split; the locked test split is never opened. Results stay preliminary until T39
supplies intervals.
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from netlift.data.contracts import DatasetName, TargetOutcome
from netlift.data.loaders import CRITEO_PARQUET, HILLSTROM_CSV
from netlift.data.pipeline import load_splits
from netlift.models.baselines import UPLIFT_FRACTIONS, ranking_metrics
from netlift.models.response_tuning import (
    OBJECTIVES,
    base_rate_reference,
    fit_final,
    read_config,
    tune,
    tuning_subset,
)
from netlift.utils.errors import ContractError
from netlift.utils.tracking import start_run

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {"hillstrom": HILLSTROM_CSV, "criteo": CRITEO_PARQUET}
TARGETS = {"hillstrom": ("visit", "conversion"), "criteo": ("conversion",)}


def run_one(
    dataset: DatasetName,
    target: TargetOutcome,
    objective: str,
    config: dict[str, Any],
    *,
    n_trials: int,
) -> dict[str, Any]:
    """Tune, refit, score the full validation split and log one MLflow run."""
    if dataset not in DATA_FILES or target not in TARGETS[dataset]:
        raise ContractError(f"Unsupported dataset/target: {dataset}/{target}.")
    seed = config["seed"]
    fixed = config["fixed"]
    frac = config["subsample_frac"][dataset]
    suffix = "" if objective == "log_loss" else f"-{objective.replace('_', '-')}"
    with start_run(
        f"T38-response-model-tuned{suffix}",
        dataset=DATA_FILES[dataset],
        seed=seed,
        experiment=f"netlift-{dataset}-{target}",
        tags={
            "task": "T38",
            "result_status": "preliminary",
            "split_name": "validation",
        },
    ) as run:
        started = perf_counter()
        # Load once; only the train matrix is reduced for the search.
        full = load_splits(dataset, target=target)
        tune_train = tuning_subset(full.train, frac, seed=full.train.meta.seed)
        result = tune(
            tune_train,
            objective=objective,
            n_trials=n_trials,
            seed=seed,
            space=config["search_space"],
            fixed=fixed,
        )
        tuning_rows = tune_train.n_rows
        search_seconds = perf_counter() - started
        del tune_train
        if full.test.is_open:
            raise ContractError("The locked test split must stay closed.")
        train, val = full.train, full.validation
        model = fit_final(train, result.best_params, seed=seed, fixed=fixed)
        score = np.asarray(
            model.predict_proba(val.X[train.X.columns]), dtype=np.float64
        )[:, 1]
        if score.shape != (val.n_rows,) or not np.isfinite(score).all():
            raise ContractError("Expected one finite score per validation row.")
        treated = val.t == 1
        reference = base_rate_reference(val.y[treated], score[treated], seed=seed)
        metrics = {**ranking_metrics(val.y, score, val.t), **reference}
        metrics["treated_roc_auc"] = reference["auc"]
        metrics["treated_log_loss"] = reference["model_log_loss"]
        if full.test.is_open:
            raise ContractError("The locked test split was opened during the run.")

        manifest = {
            "task": "T38",
            "follow_up": "#107",
            "baseline": "response-model-tuned",
            "dataset": dataset,
            "target": target,
            "split_name": "validation",
            "split_seed": train.meta.seed,
            "seed": seed,
            "status": "preliminary",
            "n_resamples": 0,
            "tuned": True,
            "objective": objective,
            "n_trials": result.n_trials,
            "test_split_opened": False,
            "subsample_frac": frac,
            "tuning_sampling_seed": train.meta.seed,
            "tuning_scope": "train only",
            "treatment": "Mens E-Mail" if dataset == "hillstrom" else 1,
            "control": "No E-Mail" if dataset == "hillstrom" else 0,
            "tuning_rows": tuning_rows,
            "train_rows": train.n_rows,
            "validation_rows": val.n_rows,
            "best_params": result.best_params,
            "best_oof": result.best_oof,
            "library": "scikit-uplift",
            "library_version": version("scikit-uplift"),
            "lightgbm_version": version("lightgbm"),
            "optuna_version": version("optuna"),
            "settings": {
                "negative_effect": True,
                "strategy": "overall",
                "k": UPLIFT_FRACTIONS,
            },
            "reference": "oracle constant on treated validation rows",
            "reference_log_loss_ci_method": "paired percentile bootstrap; constant fixed",
            "reference_n_resamples": 2000,
            "reference_seed": seed,
            "response_auc_ci_method": "DeLong normal; clipped to [0, 1]; 95%",
            "constant_auc_ci_method": "degenerate [0.5, 0.5] for a tied constant score",
            "selection_rule": "higher OOF Qini; tie: lower treated OOF log loss",
            "selection_split": "train",
            "config": config,
            "metrics": metrics,
        }
        with TemporaryDirectory(prefix="netlift-t38-tuned-") as tmp:
            folder = Path(tmp)
            pd.DataFrame({"row_id": val.row_id, "score": score}).to_csv(
                folder / "validation_scores.csv", index=False
            )
            pd.DataFrame({"row_id": val.row_id, "y": val.y, "t": val.t}).to_csv(
                folder / "validation_outcomes.csv", index=False
            )
            result.trials.to_csv(folder / "trials.csv", index=False)
            (folder / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False, default=float),
                encoding="utf-8",
            )
            joblib.dump(model, folder / "response_model.joblib")
            run.log_params(
                {
                    "baseline": "response-model-tuned",
                    "dataset": dataset,
                    "target": target,
                    "objective": objective,
                    "n_trials": result.n_trials,
                    "subsample_frac": str(frac),
                    "metric_library": "scikit-uplift",
                    "negative_effect": True,
                    "uplift_at_k_strategy": "overall",
                    "n_resamples": 0,
                    **{f"best_{k}": v for k, v in result.best_params.items()},
                }
            )
            run.log_metrics(
                {f"preliminary_{k}": float(v) for k, v in metrics.items()}
                | {
                    "tuning_best_oof_log_loss": result.best_oof["oof_log_loss"],
                    "tuning_best_oof_qini": result.best_oof["oof_qini"],
                    "tuning_seconds": search_seconds,
                }
            )
            for name in (
                "validation_scores.csv",
                "validation_outcomes.csv",
                "trials.csv",
                "manifest.json",
                "response_model.joblib",
            ):
                run.log_artifact(folder / name)
        run.set_tags(
            {
                "tuned": "true",
                "n_trials": str(result.n_trials),
                "test_split_opened": "false",
                "tuning_objective": objective,
            }
        )
    return {
        "dataset": dataset,
        "target": target,
        "objective": objective,
        "status": "preliminary",
        "n_trials": result.n_trials,
        **{f"best_{k}": v for k, v in result.best_params.items()},
        **{k: float(v) for k, v in result.best_oof.items()},
        **metrics,
    }


def main() -> None:
    """Run the requested dataset, target and objective."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hillstrom", "criteo"], required=True)
    parser.add_argument(
        "--target", choices=["visit", "conversion", "both"], default="both"
    )
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="log_loss")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/response_tuning.yaml"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else config["n_trials"]
    targets = TARGETS[args.dataset] if args.target == "both" else (args.target,)
    rows = [
        run_one(
            args.dataset,
            cast(TargetOutcome, t),
            args.objective,
            config,
            n_trials=n_trials,
        )
        for t in targets
    ]
    output = args.output or (
        ROOT / f"reports/t38-tuned-{args.dataset}-{args.objective}-preliminary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info(
        "Saved preliminary tuned results to %s; T39 intervals still required.", output
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
