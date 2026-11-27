"""Held-out decile-level uplift intervals for the decision layer (T55, ADR 0009).

The table is built once from the champion's scores on the validation split. Decile 1
holds the highest scores. For each decile it stores the ITT uplift (treated rate
minus control rate), a percentile interval from within-arm resamples, and the number
of control-arm conversions, which the abstention rule checks.

Resampling ``n`` rows of one arm with replacement and counting conversions gives a
``Binomial(n, rate)`` count, so each resample draws one binomial per arm. That is the
within-arm bootstrap of a difference in rates, computed exactly and without
materialising the row indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from netlift.decision.config import DecisionSettings
from netlift.utils.errors import ContractError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
N_EDGES = 9


@dataclass(frozen=True)
class DecileRow:
    """One decile's counts, uplift and interval on the held-out split."""

    decile: int
    rows: int
    treated: int
    control: int
    treated_events: int
    control_events: int
    uplift: float
    ci_low: float
    ci_high: float


def decile_of(edges: tuple[float, ...], score: npt.ArrayLike) -> IntArray:
    """Decile of each score: 1 for the highest; a score equal to an edge goes up."""
    values = np.asarray(score, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError("Scores must be finite to be placed in a decile.")
    negated = -np.asarray(edges, dtype=np.float64)
    return (1 + np.searchsorted(negated, -values, side="left")).astype(np.int64)


@dataclass(frozen=True)
class DecileTable:
    """Decile edges and per-decile intervals; maps a score to its decile."""

    edges: tuple[float, ...]
    rows: tuple[DecileRow, ...]
    split_name: str
    n_resamples: int
    confidence_level: float
    seed: int

    def __post_init__(self) -> None:
        """Require nine descending edges and ten rows numbered 1 to 10."""
        if len(self.edges) != N_EDGES or list(self.edges) != sorted(
            self.edges, reverse=True
        ):
            raise ContractError("A decile table needs nine edges in descending order.")
        if [row.decile for row in self.rows] != list(range(1, 11)):
            raise ContractError(
                "A decile table needs rows for deciles 1 to 10, in order."
            )
        for row in self.rows:
            if not row.ci_low <= row.ci_high:
                raise ContractError(f"Decile {row.decile} has an inverted interval.")

    def decile_of(self, score: npt.ArrayLike) -> IntArray:
        """Decile of each score under this table's edges."""
        return decile_of(self.edges, score)

    def row(self, decile: int) -> DecileRow:
        """The row of one decile."""
        return self.rows[decile - 1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "edges": list(self.edges),
            "rows": [asdict(row) for row in self.rows],
            "split_name": self.split_name,
            "n_resamples": self.n_resamples,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecileTable:
        """Rebuild a table written by :meth:`to_dict`."""
        return cls(
            edges=tuple(float(edge) for edge in data["edges"]),
            rows=tuple(DecileRow(**row) for row in data["rows"]),
            split_name=str(data["split_name"]),
            n_resamples=int(data["n_resamples"]),
            confidence_level=float(data["confidence_level"]),
            seed=int(data["seed"]),
        )


def fit_decile_table(
    score: npt.ArrayLike,
    y: npt.ArrayLike,
    t: npt.ArrayLike,
    settings: DecisionSettings,
    *,
    split_name: str = "validation",
) -> DecileTable:
    """Decile edges and within-arm bootstrap intervals from held-out scores."""
    if split_name == "test":
        raise ContractError(
            "The decile table is built on validation; the test split stays locked."
        )
    values = np.asarray(score, dtype=np.float64)
    outcome = np.asarray(y)
    arm = np.asarray(t)
    if not (len(values) == len(outcome) == len(arm)) or len(values) == 0:
        raise ContractError("score, y and t must be non-empty and of equal length.")
    if not np.isfinite(values).all():
        raise ContractError("Scores must be finite.")
    if not set(np.unique(outcome).tolist()) <= {0, 1}:
        raise ContractError("y must be binary 0/1.")
    if not set(np.unique(arm).tolist()) <= {0, 1}:
        raise ContractError("t must be binary 0/1.")
    edges = tuple(float(e) for e in np.quantile(values, np.linspace(0.9, 0.1, N_EDGES)))
    decile = decile_of(edges, values)
    rng = np.random.default_rng(settings.seed)
    tail = (1.0 - settings.confidence_level) / 2.0 * 100.0
    rows: list[DecileRow] = []
    for number in range(1, 11):
        inside = decile == number
        treated = inside & (arm == 1)
        control = inside & (arm == 0)
        n_t, n_c = int(treated.sum()), int(control.sum())
        if n_t == 0 or n_c == 0:
            raise ContractError(
                f"Decile {number} has {n_t} treated and {n_c} control rows, so its "
                "uplift is undefined. Are many scores tied at the edges?"
            )
        e_t, e_c = int(outcome[treated].sum()), int(outcome[control].sum())
        rate_t, rate_c = e_t / n_t, e_c / n_c
        draws = (
            rng.binomial(n_t, rate_t, settings.n_resamples) / n_t
            - rng.binomial(n_c, rate_c, settings.n_resamples) / n_c
        )
        low, high = np.percentile(draws, [tail, 100.0 - tail])
        rows.append(
            DecileRow(
                decile=number,
                rows=int(inside.sum()),
                treated=n_t,
                control=n_c,
                treated_events=e_t,
                control_events=e_c,
                uplift=rate_t - rate_c,
                ci_low=float(low),
                ci_high=float(high),
            )
        )
    return DecileTable(
        edges=edges,
        rows=tuple(rows),
        split_name=split_name,
        n_resamples=settings.n_resamples,
        confidence_level=settings.confidence_level,
        seed=settings.seed,
    )
