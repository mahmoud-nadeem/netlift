"""ADR 0007 evidence: which uplift metrics depend on the row order when scores tie.

For each model's validation scores (row_id + score, keyed to the outcomes file) the
script counts the rows tied at the 10%, 20% and 30% cuts, then computes scikit-uplift's
Qini coefficient, AUUC and uplift@k, and the harness's tie-aware uplift@k, in several
row orders:

- ``row_id``: ascending ``row_id``, the order the score files are written in;
- ``control_then_treated``: control rows, then treated rows, each in ``row_id`` order,
  the order that ``within_arm_bootstrap_indices`` produces;
- ``random_0`` … ``random_4``: random permutations (seed 42).

A metric that does not depend on the order gives the same value in every row of the
output. The locked test split is not read.

Example:
    python scripts/uplift_at_k_tie_check.py
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklift.metrics as sk_metrics  # type: ignore[import-untyped]

from netlift.evaluation.metrics import point_uplift_at_k
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
K1 = ROOT / "data" / "processed" / "t57-k1"
K_VALUES = (0.1, 0.2, 0.3)
N_RANDOM = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=K1 / "t_learner" / "validation_outcomes.parquet",
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        default=[
            f"t_learner={K1 / 't_learner' / 'validation_scores.parquet'}",
            f"x_learner={K1 / 'x_learner' / 'validation_scores.parquet'}",
        ],
        help="name=path pairs of row_id + score parquet files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports" / "uplift-at-k-ties.json"
    )
    return parser.parse_args(argv)


def tie_counts(score: npt.NDArray[np.float64], k: float) -> dict[str, int]:
    """Rows above the cut, and rows tied at it, for the top ``int(n * k)`` rows."""
    n_size = int(len(score) * k)
    cut = np.partition(score, len(score) - n_size)[len(score) - n_size]
    return {
        "n_size": n_size,
        "rows_above_cut": int((score > cut).sum()),
        "rows_tied_at_cut": int((score == cut).sum()),
    }


def metrics_in_order(
    y: npt.NDArray[np.int64], score: npt.NDArray[np.float64], t: npt.NDArray[np.int64]
) -> dict[str, float]:
    """Every metric on the rows in the order given."""
    values = {
        "qini_sklift": float(
            sk_metrics.qini_auc_score(y, score, t, negative_effect=True)
        ),
        "auuc_sklift": float(sk_metrics.uplift_auc_score(y, score, t)),
    }
    for k in K_VALUES:
        label = f"{round(k * 100)}"
        values[f"uplift_at_{label}_sklift"] = float(
            sk_metrics.uplift_at_k(y, score, t, strategy="overall", k=k)
        )
        values[f"uplift_at_{label}_tie_aware"] = point_uplift_at_k(
            y, score, t, k=k, strategy="overall"
        )
    return values


def main(argv: list[str] | None = None) -> None:
    """Count ties at each cut and compute every metric in every row order."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    outcomes = pd.read_parquet(args.outcomes).sort_values("row_id", kind="stable")
    row_id = outcomes["row_id"].to_numpy(dtype=np.int64)
    y = outcomes["y"].to_numpy(dtype=np.int64)
    t = outcomes["t"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    orders: dict[str, npt.NDArray[np.int64]] = {
        "row_id": np.arange(len(row_id)),
        "control_then_treated": np.argsort(t, kind="stable"),
    }
    for number in range(N_RANDOM):
        orders[f"random_{number}"] = rng.permutation(len(row_id))

    report: dict[str, Any] = {
        "rows": len(row_id),
        "seed": args.seed,
        "scikit_uplift_version": version("scikit-uplift"),
        "models": {},
    }
    for pair in args.scores:
        name, _, path = pair.partition("=")
        frame = pd.read_parquet(path).set_index("row_id")["score"]
        if len(frame) != len(row_id) or not frame.index.isin(row_id).all():
            raise ContractError(f"{path} does not cover exactly the outcome rows.")
        score = frame.reindex(row_id).to_numpy(dtype=np.float64)
        by_order = {
            label: metrics_in_order(y[index], score[index], t[index])
            for label, index in orders.items()
        }
        spread = {
            metric: max(values[metric] for values in by_order.values())
            - min(values[metric] for values in by_order.values())
            for metric in next(iter(by_order.values()))
        }
        report["models"][name] = {
            "distinct_scores": len(np.unique(score)),
            "ties": {f"{round(k * 100)}": tie_counts(score, k) for k in K_VALUES},
            "by_order": by_order,
            "max_minus_min_across_orders": spread,
        }
        logger.info("%s: spread across orders %s", name, spread)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("written to %s", args.out)


if __name__ == "__main__":
    main()
