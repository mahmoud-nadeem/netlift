"""The decision layer's settings, read from ``config/decision.yaml`` (ADR 0009)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from netlift.utils.errors import ContractError

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "decision.yaml"


@dataclass(frozen=True)
class DecisionSettings:
    """Abstention threshold and the decile-interval bootstrap settings."""

    min_control_events_per_decile: int
    n_deciles: int = 10
    n_resamples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject settings that would make an interval or a verdict meaningless."""
        for name in ("min_control_events_per_decile", "n_deciles", "n_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{name} must be a positive int, got {value!r}.")
        if self.n_deciles != 10:
            raise ContractError(
                "Interface 3 defines deciles 1 to 10; n_deciles must be 10."
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ContractError(
                f"confidence_level must be in (0, 1), got {self.confidence_level!r}."
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DecisionSettings:
        """Build the settings from a parsed YAML mapping."""
        if "min_control_events_per_decile" not in data:
            raise ContractError(
                "config/decision.yaml must set min_control_events_per_decile; "
                "the specification requires it to be a defended configuration value."
            )
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ContractError(f"Unknown decision settings: {unknown}.")
        return cls(**data)


def load_decision_settings(path: Path = CONFIG_PATH) -> DecisionSettings:
    """Read ``config/decision.yaml``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DecisionSettings.from_mapping(data)
