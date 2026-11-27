"""Interface 3's verdict rules: from a decile's interval to a recommendation (T55).

The interval is the held-out decile-level interval of the customer's decile
(``docs/design/interfaces.md``, section 3). The rules, in order:

1. fewer control-arm conversions in the decile than the configured minimum:
   ``insufficient_control_data``, ``no_action``;
2. lower bound above zero: ``confident``, ``contact``, segment ``persuadable``;
3. upper bound below zero: ``confident``, ``suppress``, segment ``sleeping_dog``
   (the sleeping-dog rule);
4. otherwise the interval touches or straddles zero: ``insufficient_evidence``,
   ``no_action``.

``sure_thing`` and ``lost_cause`` need a predicted control response, which an uplift
model does not give, so they are never emitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from netlift.api.schema import Recommendation, SegmentLabel, Verdict
from netlift.utils.errors import ContractError


@dataclass(frozen=True)
class Decision:
    """The verdict, recommendation and segment label for one decile."""

    verdict: Verdict
    recommendation: Recommendation
    segment_label: SegmentLabel


def decide(
    ci_low: float, ci_high: float, control_events: int, min_control_events: int
) -> Decision:
    """Apply Interface 3's rules to one decile's interval and control-event count."""
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)) or ci_low > ci_high:
        raise ContractError(f"Invalid interval [{ci_low}, {ci_high}].")
    if control_events < 0 or min_control_events < 1:
        raise ContractError(
            f"control_events must be >= 0 and the minimum >= 1; got {control_events} "
            f"and {min_control_events}."
        )
    if control_events < min_control_events:
        return Decision("insufficient_control_data", "no_action", "unknown")
    if ci_low > 0.0:
        return Decision("confident", "contact", "persuadable")
    if ci_high < 0.0:
        return Decision("confident", "suppress", "sleeping_dog")
    return Decision("insufficient_evidence", "no_action", "unknown")
