"""
T20: response rate by treatment arm, with bootstrap confidence intervals.

Response metric: `visit` (per Spec Section 22 — the recommended development
target, far more stable than `conversion` at this sample size).

This script is the single source of truth for the committed figure.
Nothing in notebooks/ is imported here, and the figure is never
hand-exported from a notebook cell (repository rule, per T20's issue).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HILLSTROM_PATH = Path("data/raw/hillstrom.csv")
OUTPUT_PATH = Path("reports/figures/response_by_arm.png")

RESPONSE_COL = "visit"
ARM_COL = "segment"
ARM_ORDER = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]

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
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_ylabel(f"{RESPONSE_COL.capitalize()} rate")
    ax.set_title("Response rate by treatment arm (Hillstrom)\nwith 95% bootstrap CI")
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
    print(f"\nSaved: {OUTPUT_PATH}")

    # The two sentences, derived from the computed results -- not assumed.
    # Whichever arm actually has the highest rate drives the first sentence,
    # so this stays correct even if the data or arm ordering changes.
    best = max(results, key=lambda r: r["rate"])
    control = next(r for r in results if r["arm"] == "No E-Mail")

    print("\nWhat a response model would conclude:")
    print(
        f"The {best['arm']} arm has the highest {RESPONSE_COL} rate "
        f"({best['rate']:.2%}), so a response model would rank those "
        f"customers highest and target them."
    )
    print("\nWhy that does not tell us who to send the email to:")
    print(
        f"This chart shows who responded, not whose response was CAUSED by "
        f"the email. The {control['arm']} (control) arm still shows a "
        f"{control['rate']:.2%} {RESPONSE_COL} rate with no email at all -- "
        f"so part of every arm's rate would have happened regardless of "
        f"treatment. Only the gap against this control arm, not the raw "
        f"rate, estimates the email's actual effect."
    )


if __name__ == "__main__":
    main()
