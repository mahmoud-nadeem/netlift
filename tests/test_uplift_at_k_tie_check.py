"""Tests for the ADR 0007 evidence script: tie counts and order dependence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """Import scripts/uplift_at_k_tie_check.py as a module."""
    path = ROOT / "scripts" / "uplift_at_k_tie_check.py"
    spec = importlib.util.spec_from_file_location("uplift_at_k_tie_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tie_counts_split_the_cut() -> None:
    module = load_script()
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert module.tie_counts(score, 0.3) == {
        "n_size": 3,
        "rows_above_cut": 1,
        "rows_tied_at_cut": 4,
    }


def test_only_scikit_uplifts_uplift_at_k_depends_on_the_order(tmp_path: Path) -> None:
    module = load_script()
    rng = np.random.default_rng(0)
    n = 4000
    row_id = np.arange(n, dtype=np.int64) * 2
    t = rng.integers(0, 2, n)
    effect = rng.normal(size=n)
    y = (rng.random(n) < 0.1 + 0.1 * t * (effect > 0)).astype(int)
    pd.DataFrame({"row_id": row_id, "y": y, "t": t}).to_parquet(
        tmp_path / "outcomes.parquet", index=False
    )
    pd.DataFrame({"row_id": row_id, "score": np.round(effect, 0)}).to_parquet(
        tmp_path / "scores.parquet", index=False
    )
    out = tmp_path / "ties.json"
    module.main(
        [
            "--outcomes",
            str(tmp_path / "outcomes.parquet"),
            "--scores",
            f"tied={tmp_path / 'scores.parquet'}",
            "--out",
            str(out),
        ]
    )
    spread = json.loads(out.read_text(encoding="utf-8"))["models"]["tied"][
        "max_minus_min_across_orders"
    ]
    assert spread["qini_sklift"] == 0.0
    assert spread["auuc_sklift"] == 0.0
    assert spread["uplift_at_30_sklift"] > 1e-4
    assert spread["uplift_at_30_tie_aware"] < 1e-12
