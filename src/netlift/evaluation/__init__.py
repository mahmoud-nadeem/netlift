"""Evaluation harness: Qini, AUUC and uplift@k with within-arm bootstrap intervals."""

from netlift.evaluation.contracts import MetricResult, paired_difference
from netlift.evaluation.curves import (
    CurveBand,
    cumulative_gain_curve,
    qini_curve,
    uplift_curve,
)
from netlift.evaluation.metrics import auuc, qini_coefficient, uplift_at_k

__all__ = [
    "CurveBand",
    "MetricResult",
    "auuc",
    "cumulative_gain_curve",
    "paired_difference",
    "qini_coefficient",
    "qini_curve",
    "uplift_at_k",
    "uplift_curve",
]
