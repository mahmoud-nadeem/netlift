"""Estimate the complier baseline conversion rate in Criteo-UPLIFT.

Why this script exists
----------------------
Section 6 of the specification expressed the treatment-on-the-treated (Wald)
estimate as a relative lift over the *population* control conversion rate.
That comparison is invalid. The Wald estimate is a local average treatment
effect among compliers -- the small share of treated users who were actually
exposed -- while the population control rate averages over compliers and
never-takers alike. The two numbers do not describe the same people, so their
ratio is not a lift.

Criteo exhibits one-sided non-compliance: no control row carries exposure = 1.
Under that structure, inside the treated arm, exposure = 1 identifies compliers
and exposure = 0 identifies never-takers. Adding the exclusion restriction --
already assumed by the Wald ratio itself -- the never-takers' observed outcome
is their untreated outcome, and the compliers' untreated baseline follows by
subtraction:

    E[Y(0) | complier] = (y_control - (1 - pi) * y_treated_unexposed) / pi
    pi                 = P(exposure = 1 | treatment = 1)

What the result means
---------------------
The estimate is a conversion probability, so it must lie in [0, 1]. An interval
sitting clearly below zero is evidence against the assumption set -- the
exclusion restriction or the one-sided premise -- and not merely a noisy number.
An interval that contains zero means the data cannot resolve the baseline at
this compliance rate, which is itself a reportable limitation. A positive
estimate gives a denominator the relative lift can honestly be expressed
against.

The script reports an interval in every case and draws no conclusion the
interval does not support.

Run:
    python scripts/criteo_complier_baseline.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
NEEDED = ("treatment", "exposure", "conversion", "visit")
DEFAULT_DRAWS = 20_000
DEFAULT_SEED = 42


def resolve_columns(path: Path) -> dict[str, str]:
    """Map each needed column to its spelling in the file header."""
    header = pd.read_csv(path, nrows=0)
    lookup = {str(name).strip().lower(): str(name) for name in header.columns}
    missing = [name for name in NEEDED if name not in lookup]
    if missing:
        raise SystemExit(
            f"Dataset is missing expected columns {missing}. "
            f"Columns present: {list(header.columns)}"
        )
    return {name: lookup[name] for name in NEEDED}


def load_counts(path: Path) -> pd.DataFrame:
    """Return the (treatment, exposure) cell counts with outcome totals."""
    columns = resolve_columns(path)
    frame = pd.read_csv(
        path,
        usecols=list(columns.values()),
        dtype=dict.fromkeys(columns.values(), "float32"),
    )
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    for name in ("treatment", "exposure"):
        values = set(np.unique(frame[name].to_numpy()))
        if not values <= {0.0, 1.0}:
            raise SystemExit(f"Column {name!r} is not binary: found {sorted(values)}")
    grouped = frame.groupby(["treatment", "exposure"], observed=True).agg(
        rows=("conversion", "size"),
        conversions=("conversion", "sum"),
        visits=("visit", "sum"),
    )
    return grouped.reset_index().astype(
        {"treatment": "int8", "exposure": "int8", "rows": "int64"}
    )


def cell(counts: pd.DataFrame, treatment: int, exposure: int) -> tuple[int, int]:
    """Return (rows, conversions) for one cell, or (0, 0) if the cell is empty."""
    match = counts[
        (counts["treatment"] == treatment) & (counts["exposure"] == exposure)
    ]
    if match.empty:
        return 0, 0
    row = match.iloc[0]
    return int(row["rows"]), round(float(row["conversions"]))


def point_estimates(
    n_control: int,
    k_control: np.ndarray,
    n_exposed: np.ndarray,
    k_exposed: np.ndarray,
    n_unexposed: np.ndarray,
    k_unexposed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute every quantity from cell counts. Arrays broadcast over draws."""
    n_treated = n_exposed + n_unexposed
    pi = n_exposed / n_treated
    y_control = k_control / n_control
    y_exposed = k_exposed / np.maximum(n_exposed, 1)
    y_unexposed = k_unexposed / np.maximum(n_unexposed, 1)
    y_treated = (k_exposed + k_unexposed) / n_treated

    itt = y_treated - y_control
    late = itt / pi
    baseline = (y_control - (1.0 - pi) * y_unexposed) / pi
    with np.errstate(divide="ignore", invalid="ignore"):
        lift_vs_complier = np.where(baseline > 0, late / baseline, np.nan)
        lift_vs_population = late / y_control
    return {
        "pi": pi,
        "y_control": y_control,
        "y_treated": y_treated,
        "y_exposed": y_exposed,
        "y_unexposed": y_unexposed,
        "itt": itt,
        "late": late,
        "baseline": baseline,
        "lift_vs_complier": lift_vs_complier,
        "lift_vs_population": lift_vs_population,
    }


def bootstrap(
    n_control: int,
    k_control: int,
    n_exposed: int,
    k_exposed: int,
    n_unexposed: int,
    k_unexposed: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample the sufficient statistics rather than the 14 million rows.

    The control arm is one binomial draw. The treated arm is one multinomial
    draw over the four (exposure, conversion) cells, so that pi is resampled
    jointly with the outcome counts instead of being held fixed.
    """
    rng = np.random.default_rng(seed)
    n_treated = n_exposed + n_unexposed
    probabilities = np.array(
        [
            k_exposed,
            n_exposed - k_exposed,
            k_unexposed,
            n_unexposed - k_unexposed,
        ],
        dtype="float64",
    )
    probabilities /= probabilities.sum()
    treated_draws = rng.multinomial(n_treated, probabilities, size=draws)
    control_draws = rng.binomial(n_control, k_control / n_control, size=draws)
    return point_estimates(
        n_control=n_control,
        k_control=control_draws.astype("float64"),
        n_exposed=(treated_draws[:, 0] + treated_draws[:, 1]).astype("float64"),
        k_exposed=treated_draws[:, 0].astype("float64"),
        n_unexposed=(treated_draws[:, 2] + treated_draws[:, 3]).astype("float64"),
        k_unexposed=treated_draws[:, 2].astype("float64"),
    )


def interval(samples: np.ndarray) -> tuple[float, float]:
    """Return the 95% percentile interval, ignoring undefined draws."""
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        print("Run scripts/../src/netlift/data/download.py first.", file=sys.stderr)
        return 1

    counts = load_counts(args.data)

    n_c0, k_c0 = cell(counts, 0, 0)
    n_c1, _ = cell(counts, 0, 1)
    n_t0, k_t0 = cell(counts, 1, 0)
    n_t1, k_t1 = cell(counts, 1, 1)
    total = n_c0 + n_c1 + n_t0 + n_t1

    print("=" * 72)
    print("Criteo-UPLIFT: raw 2x2 treatment x exposure table (counted, not derived)")
    print("=" * 72)
    print(f"{'':>14}{'exposure=0':>16}{'exposure=1':>16}{'total':>16}")
    print(f"{'treatment=0':>14}{n_c0:>16,}{n_c1:>16,}{n_c0 + n_c1:>16,}")
    print(f"{'treatment=1':>14}{n_t0:>16,}{n_t1:>16,}{n_t0 + n_t1:>16,}")
    print(f"{'total':>14}{n_c0 + n_t0:>16,}{n_c1 + n_t1:>16,}{total:>16,}")
    print()
    print(f"seed={args.seed}  bootstrap draws={args.draws:,}  source={args.data.name}")
    print()

    if n_c1 != 0:
        print("WARNING: control rows with exposure=1 exist.")
        print("One-sided non-compliance does NOT hold and the estimator below")
        print("is not identified. Stop and report this.")
        return 2

    if n_t1 == 0:
        print("No exposed treated rows; nothing to estimate.")
        return 2

    n_control = n_c0
    point = point_estimates(
        n_control=n_control,
        k_control=np.array([float(k_c0)]),
        n_exposed=np.array([float(n_t1)]),
        k_exposed=np.array([float(k_t1)]),
        n_unexposed=np.array([float(n_t0)]),
        k_unexposed=np.array([float(k_t0)]),
    )
    draws = bootstrap(
        n_control=n_control,
        k_control=k_c0,
        n_exposed=n_t1,
        k_exposed=k_t1,
        n_unexposed=n_t0,
        k_unexposed=k_t0,
        draws=args.draws,
        seed=args.seed,
    )

    rows = [
        ("Compliance pi = P(exposed | treated)", "pi", percent),
        ("Control conversion rate", "y_control", percent),
        ("Treated conversion rate", "y_treated", percent),
        ("Treated & exposed conversion rate", "y_exposed", percent),
        ("Treated & unexposed conversion rate", "y_unexposed", percent),
        ("ITT (treated - control)", "itt", percent),
        ("LATE / Wald (effect on compliers)", "late", percent),
        ("Complier baseline E[Y(0)|complier]", "baseline", percent),
    ]
    print("=" * 72)
    print("Estimates with 95% bootstrap intervals")
    print("=" * 72)
    for label, key, fmt in rows:
        value = float(point[key][0])
        low, high = interval(draws[key])
        print(f"{label:<38}{fmt(value):>12}  [{fmt(low)}, {fmt(high)}]")

    print()
    lift_c = float(point["lift_vs_complier"][0])
    low_c, high_c = interval(draws["lift_vs_complier"])
    lift_p = float(point["lift_vs_population"][0])
    print(
        f"{'Relative lift vs COMPLIER baseline':<38}{lift_c:>11.2f}x"
        f"  [{low_c:.2f}x, {high_c:.2f}x]"
    )
    print(
        f"{'Relative lift vs POPULATION control':<38}{lift_p:>11.2f}x"
        "   <- invalid comparison, shown only"
    )
    print(f"{'':38}{'':>12}      for contrast with the old text")

    print()
    below = float(np.mean(draws["baseline"] < 0.0))
    print("=" * 72)
    print("Reading")
    print("=" * 72)
    print(f"Share of bootstrap draws with a negative complier baseline: {below:.1%}")
    if math.isnan(high_c):
        print("The complier baseline is not positive across the interval, so a")
        print("relative lift cannot be expressed against it.")
    if below > 0.975:
        print("The baseline is negative beyond sampling noise. A conversion")
        print("probability cannot be negative, so at least one assumption fails:")
        print("the exclusion restriction, or the one-sided premise. Report this")
        print("as a finding with the interval printed, not as an aside.")
    elif below > 0.025:
        print("The interval straddles zero. At this compliance rate the data")
        print("cannot resolve the complier baseline. Report that limitation")
        print("rather than a lift computed against an unresolved denominator.")
    else:
        print("The baseline is positive across the interval and can serve as the")
        print("denominator for a relative lift among compliers.")
    print()
    print("The LATE can be written two ways -- ITT/pi, and the exposed rate minus")
    print("this baseline. They are algebraically identical, so their agreement")
    print("checks the arithmetic, not the assumptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
