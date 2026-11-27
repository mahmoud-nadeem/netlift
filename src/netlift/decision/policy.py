"""Policy modes: which scored customers a campaign contacts (T55, ADR 0009).

Interface 3 says ``recommendation`` is the model's, and each mode decides what to do
with ``no_action``. Fixed budget and profit-maximising contact only ``contact``
recommendations. Suppression-only contacts everyone except ``suppress``, so
``no_action`` customers get business-as-usual contact. A ``suppress`` (the sleeping-dog
rule) is never contacted in any mode. Fixed target is not in M5 (ADR 0009).
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from netlift.utils.errors import ContractError

PolicyMode = Literal["fixed_budget", "profit_maximising", "suppression_only"]
BoolArray = npt.NDArray[np.bool_]
RECOMMENDATIONS = {"contact", "suppress", "no_action"}


def select_contacts(
    uplift: npt.ArrayLike,
    recommendation: npt.ArrayLike,
    mode: str,
    *,
    contact_cost: float,
    conversion_margin: float,
    budget: float | None = None,
) -> BoolArray:
    """True for each customer the mode contacts."""
    if mode not in get_args(PolicyMode):
        raise ContractError(
            f"mode must be one of {get_args(PolicyMode)}, got {mode!r}."
        )
    if not contact_cost > 0 or not conversion_margin > 0:
        raise ContractError("contact_cost and conversion_margin must be positive.")
    score = np.asarray(uplift, dtype=np.float64)
    action = np.asarray(recommendation)
    if score.shape != action.shape:
        raise ContractError("uplift and recommendation must have the same length.")
    if not set(action.tolist()) <= RECOMMENDATIONS:
        raise ContractError("recommendation holds a value outside Interface 3.")
    if mode == "suppression_only":
        return np.asarray(action != "suppress", dtype=bool)
    eligible = (action == "contact") & (score > 0.0)
    if mode == "profit_maximising":
        return np.asarray(eligible & (score > contact_cost / conversion_margin))
    if budget is None or not budget >= 0:
        raise ContractError("fixed_budget needs a budget of zero or more.")
    k = int(budget // contact_cost)
    chosen = np.zeros(score.shape, dtype=bool)
    candidates = np.flatnonzero(eligible)
    # Highest uplift first; equal scores keep their input order (a stable sort).
    ranked = candidates[np.argsort(-score[candidates], kind="stable")]
    chosen[ranked[:k]] = True
    return chosen
