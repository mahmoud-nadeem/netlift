"""Shared pytest fixtures.

``synthetic_splits`` is SYNTHETIC data conforming to interface 1. It exists so
code can be built and tested before the real pipeline lands. Never use it for a
reported result.
"""

from __future__ import annotations

import pytest

from netlift.data.contracts import DatasetSplits
from netlift.data.synthetic import make_synthetic_splits


@pytest.fixture
def synthetic_splits() -> DatasetSplits:
    """Seeded synthetic train, validation and locked test splits (target: visit)."""
    return make_synthetic_splits("visit", n_rows=600, seed=0, n_folds=3)
