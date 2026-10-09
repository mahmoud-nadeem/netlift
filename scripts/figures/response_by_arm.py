"""
T20: visit rate by treatment arm, with bootstrap confidence intervals.

Response metric: `visit` (per Spec Section 22 -- the recommended development
target, far more stable than `conversion` at this sample size).

This script is the single source of truth for the committed figure: it
prints the numbers and saves the figure. The interpretation of the chart
lives in one place only, the notebook markdown under the chart.
Nothing in notebooks/ is imported here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HILLSTROM_PATH = ROOT / "data" / "raw" / "hillstrom.csv"
OUTPUT_PATH = ROOT / "reports" / "figures" / "response_by_arm.png"

RESPONSE_COL = "visit"
ARM_COL = "segment"
ARM_ORDER = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
ARM_COLORS = ["#4C72B0", "#55A868", "#8C8C8C"]  # control in neutral grey

N_BOOTSTRAP = 2000
CI_LOW, CI_HIGH = 2.5, 97.5
RNG_SEED = 42


def bootstrap_rate_ci(values, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap a 95% CI for a binary rate, resampling within the arm."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    point = values.mean()
    lo, hi = np.percentile(boot_means, [CI_LOW, CI_HIGH])
    return point, lo, hi


def main():
    hill = pd.read_csv(HILLSTROM_PATH)

    missing_arms = set(ARM_ORDER) - set(hill[ARM_COL].unique())
    if missing_arms:
        raise ValueError(f"Expected arms not found in data: {missing_arms}")

    results = []
    for arm in ARM_ORDER:
        arm_values = hill.loc[hill[ARM_COL] == arm, RESPONSE_COL]
        point, lo, hi = bootstrap_rate_ci(arm_values)
        results.append(
            {
                "arm": arm,
                "n": len(arm_values),
                "rate": point,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
        print(
            f"{arm:15s} n={len(arm_values):>6}  "
            f"{RESPONSE_COL} rate={point:.4%}  "
            f"95% CI=[{lo:.4%}, {hi:.4%}]"
        )

    rates = [r["rate"] for r in results]
    err_low = [r["rate"] - r["ci_low"] for r in results]
    err_high = [r["ci_high"] - r["rate"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ARM_ORDER,
        rates,
        yerr=[err_low, err_high],
        capsize=6,
        color=ARM_COLORS,
    )
    rate_label = f"{RESPONSE_COL.capitalize()} rate"
    ax.set_ylabel(rate_label)
    ax.set_title(f"{rate_label} by treatment arm (Hillstrom)\nwith 95% bootstrap CI")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1%}")

    # Headroom based on the tallest CI, not the tallest bar -- an arbitrary
    # multiple of the bar height can clip the error bar or its label.
    max_ci_high = max(r["ci_high"] for r in results)
    ax.set_ylim(0, max_ci_high * 1.15)

    for bar, r in zip(bars, results):
        ax.annotate(
            f"{r['rate']:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, r["ci_high"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    # Print the repo-relative path so no local machine path leaks into outputs.
    print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()