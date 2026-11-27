"""Validation of the YAML configuration files."""

from __future__ import annotations

import math
from typing import Any

from netlift.utils.errors import ConfigError

REQUIRED_DATA_KEYS = (
    "seed",
    "n_folds",
    "shares",
    "tolerance",
    "min_control_events_per_fold",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_data_config(cfg: Any) -> dict[str, Any]:
    """Check config/data.yaml and return it. Raise ConfigError with the reason."""
    if not isinstance(cfg, dict):
        raise ConfigError("config/data.yaml must hold a mapping at the top level.")
    for key in REQUIRED_DATA_KEYS:
        if key not in cfg:
            raise ConfigError(f"config/data.yaml: missing key {key!r}.")
    if not _is_int(cfg["seed"]) or cfg["seed"] < 0:
        raise ConfigError(f"seed must be a non-negative int, got {cfg['seed']!r}.")
    if not _is_int(cfg["n_folds"]) or cfg["n_folds"] < 2:
        raise ConfigError(f"n_folds must be an int >= 2, got {cfg['n_folds']!r}.")
    shares = cfg["shares"]
    if (
        not isinstance(shares, list | tuple)
        or len(shares) != 3
        or not all(_is_number(s) and s > 0 for s in shares)
        or not math.isclose(sum(shares), 1.0, abs_tol=1e-9)
    ):
        raise ConfigError(
            f"shares must be three positive numbers that sum to 1, got {shares!r}."
        )
    if not _is_number(cfg["tolerance"]) or cfg["tolerance"] <= 0:
        raise ConfigError(f"tolerance must be > 0, got {cfg['tolerance']!r}.")
    events = cfg["min_control_events_per_fold"]
    if not _is_int(events) or events < 0:
        raise ConfigError(
            f"min_control_events_per_fold must be an int >= 0, got {events!r}."
        )
    gap = cfg.get("max_cell_gap_rows", 0)
    if not _is_number(gap) or gap < 0:
        raise ConfigError(f"max_cell_gap_rows must be >= 0, got {gap!r}.")
    return cfg
