"""Compare naive and propensity-weighted class transformation on Criteo."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklift.metrics import qini_curve

from netlift.data.pipeline import load_splits
from netlift.models.meta_learners import fit_class_transformation


def main() -> None:
    """Train both variants and plot results on the same validation rows."""
    splits = load_splits("criteo", target="conversion")
    train = splits.train
    validation = splits.validation
    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 4,
        "random_state": 42,
        "verbosity": -1,
    }

    variants = (("Naive", False), ("Propensity weighted", True))
    mean_scores = []
    fig, (curve_ax, score_ax) = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, weighted in variants:
        print(f"Training {label} model...", flush=True)
        model = fit_class_transformation(train, params, propensity_weighted=weighted)
        scores = model.predict_uplift(validation.X)
        if scores.shape != (validation.n_rows,) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid validation scores for {label}")

        contacts, gain = qini_curve(validation.y, scores, validation.t)
        curve_ax.plot(100 * contacts / validation.n_rows, gain, label=label)
        mean_scores.append(100 * float(scores.mean()))
        print(f"{label}: mean predicted uplift = {mean_scores[-1]:.4f} pp", flush=True)

    treated = validation.t == 1
    treated_outcomes = validation.y[treated]
    control_outcomes = validation.y[~treated]
    treated_rate = float(treated_outcomes.mean())
    control_rate = float(control_outcomes.mean())
    observed_difference = 100 * (treated_rate - control_rate)

    seed = 42
    n_resamples = 1_000
    rng = np.random.default_rng(seed)
    # For binary outcomes, resampling rows within an arm gives a binomial count.
    treated_draws = rng.binomial(treated_outcomes.size, treated_rate, n_resamples)
    control_draws = rng.binomial(control_outcomes.size, control_rate, n_resamples)
    differences = 100 * (
        treated_draws / treated_outcomes.size - control_draws / control_outcomes.size
    )
    low, high = np.percentile(differences, (2.5, 97.5)).tolist()
    curve_ax.set(
        title="Validation Qini curves (point estimates)",
        xlabel="Validation rows targeted (%)",
        ylabel="Qini gain",
    )
    curve_ax.legend()

    score_ax.bar(range(len(variants)), mean_scores)
    score_ax.set_xticks(
        range(len(variants)),
        labels=[
            f"{label}\n{mean:+.4f} pp"
            for (label, _), mean in zip(variants, mean_scores, strict=True)
        ],
    )
    score_ax.axhline(0, color="grey", linewidth=0.8)
    score_ax.axhline(
        observed_difference,
        color="black",
        linestyle="--",
        label=f"Observed {observed_difference:+.4f} pp\n95% CI [{low:+.4f}, {high:+.4f}] pp",
    )
    score_ax.set(title="Mean predicted uplift", ylabel="Percentage points")
    score_ax.legend(loc="lower right", fontsize=9)

    if splits.test.is_open:
        raise RuntimeError("The test split must stay locked.")
    fig.suptitle("Criteo conversion: class transformation on validation")
    fig.tight_layout()
    reports = Path(__file__).resolve().parents[2] / "reports"
    figure_path = reports / "figures" / "t56-class-transformation-trap.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    result = {
        "dataset": "criteo",
        "target": "conversion",
        "split": "validation",
        "validation_rows": validation.n_rows,
        "mean_predicted_uplift_pp": dict(
            zip(("naive", "propensity_weighted"), mean_scores, strict=True)
        ),
        "observed_arm_difference_pp": observed_difference,
        "observed_arm_difference_ci_pp": [low, high],
        "confidence_level": 0.95,
        "bootstrap_seed": seed,
        "bootstrap_resamples": n_resamples,
    }
    summary_path = reports / "t56-class-transformation-trap.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Observed validation difference: {observed_difference:+.4f} pp")
    print(f"95% CI: [{low:+.4f}, {high:+.4f}] pp")
    print(f"Saved: {figure_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
