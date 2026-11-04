"""Run T38 on Hillstrom with explicit economics and preliminary MLflow results.

Specification §§9, 25, 27; interfaces 1/2; ADR 0001. Use the canonical T37
split, never a temporary split. T39 must add intervals before headline reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import yaml

from netlift.data.contracts import TargetOutcome
from netlift.data.loaders import HILLSTROM_CSV
from netlift.data.pipeline import load_splits
from netlift.models.baselines import BASELINE_NAMES, UPLIFT_FRACTIONS, build_baseline
from netlift.utils.tracking import start_run

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict[str, Any]:
    """Validate the small T38 configuration before starting any experiment."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"seed", "response_model"}:
        raise ValueError("Config must contain exactly seed and response_model.")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(config["response_model"], dict):
        raise TypeError("response_model must be a mapping of LightGBM parameters.")
    return config


def run_target(
    target: TargetOutcome, config: dict[str, Any], contact_cost: float
) -> list[dict[str, Any]]:
    """Log three runs, including fitting time, and return a preliminary table."""
    if target not in {"visit", "conversion"}:
        raise ValueError("target must be visit or conversion")
    if not np.isfinite(contact_cost) or contact_cost <= 0:
        raise ValueError("contact_cost must be finite and greater than zero")
    splits = load_splits("hillstrom", target=target)
    seed = config["seed"]
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="netlift-t38-") as tmp:
        folder = Path(tmp)
        for name in BASELINE_NAMES:
            # tags works with both T36's wrapper and T41's required-task extension.
            with start_run(
                f"T38-{name}",
                dataset=HILLSTROM_CSV,
                seed=seed,
                experiment=f"netlift-hillstrom-{target}",
                tags={
                    "task": "T38",
                    "result_status": "preliminary",
                    "split_name": "validation",
                },
            ) as run:
                result = build_baseline(
                    splits,
                    name,
                    seed=seed,
                    contact_cost=contact_cost,
                    model_params=config["response_model"],
                )
                manifest = {
                    "task": "T38",
                    "baseline": name,
                    "dataset": "hillstrom",
                    "target": target,
                    "treatment": "Mens E-Mail",
                    "control": "No E-Mail",
                    "split_name": "validation",
                    "split_seed": splits.train.meta.seed,
                    "seed": seed,
                    "status": "preliminary",
                    "n_resamples": 0,
                    "library": "scikit-uplift",
                    "library_version": version("scikit-uplift"),
                    "numpy_version": version("numpy"),
                    "lightgbm_version": version("lightgbm"),
                    "settings": {
                        "negative_effect": True,
                        "strategy": "overall",
                        "k": UPLIFT_FRACTIONS,
                    },
                    "score_kind": "all_contact_indicator"
                    if name == "blanket"
                    else "ranking",
                    "blanket_metric_definition": "mean(y[t=1]) - mean(y[t=0]); k=100%",
                    "contact_cost": contact_cost,
                    "config": config,
                    "metrics": result.metrics,
                }
                result.scores.to_csv(folder / "validation_scores.csv", index=False)
                pd.DataFrame(
                    {
                        "row_id": splits.validation.row_id,
                        "y": splits.validation.y,
                        "t": splits.validation.t,
                    }
                ).to_csv(folder / "validation_outcomes.csv", index=False)
                (folder / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
                )
                run.log_params(
                    {
                        "baseline": name,
                        "target": target,
                        "contact_cost": contact_cost,
                        "split_seed": splits.train.meta.seed,
                        "metric_library": "scikit-uplift",
                        "metric_library_version": version("scikit-uplift"),
                        "negative_effect": True,
                        "uplift_at_k_strategy": "overall",
                        "uplift_at_k_fractions": "0.1,0.2,0.3",
                        "n_resamples": 0,
                        "train_rows": splits.train.n_rows,
                        "validation_rows": splits.validation.n_rows,
                    }
                )
                if result.model is not None:
                    run.log_params(
                        {f"model_{k}": v for k, v in result.model.get_params().items()}
                    )
                    joblib.dump(result.model, folder / "response_model.joblib")
                    run.log_artifact(folder / "response_model.joblib")
                run.log_metrics(
                    {f"preliminary_{k}": v for k, v in result.metrics.items()}
                )
                for filename in (
                    "validation_scores.csv",
                    "validation_outcomes.csv",
                    "manifest.json",
                ):
                    run.log_artifact(folder / filename)
            rows.append(
                {
                    "target": target,
                    "baseline": name,
                    "status": "preliminary",
                    **result.metrics,
                }
            )
    return rows


def main() -> None:
    """Run one or both targets, using real Hillstrom and the canonical split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["visit", "conversion", "both"], default="both"
    )
    parser.add_argument("--contact-cost", type=float, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/t38-baselines-preliminary.csv"
    )
    args = parser.parse_args()
    config = read_config(args.config)
    targets = ("visit", "conversion") if args.target == "both" else (args.target,)
    rows = [
        row
        for target in targets
        for row in run_target(cast(TargetOutcome, target), config, args.contact_cost)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    logger.info(
        "Saved preliminary point estimates to %s; T39 intervals still required.",
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
