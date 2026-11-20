"""Compare naive and propensity-weighted class transformation on Criteo."""

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
    observed_difference = 100 * (
        validation.y[treated].mean() - validation.y[~treated].mean()
    )
    curve_ax.set(
        title="Validation Qini curves",
        xlabel="Validation rows targeted (%)",
        ylabel="Qini gain",
    )
    curve_ax.legend()

    score_ax.bar([label for label, _ in variants], mean_scores)
    score_ax.axhline(0, color="grey", linewidth=0.8)
    score_ax.axhline(
        observed_difference,
        color="black",
        linestyle="--",
        label="Observed arm difference",
    )
    score_ax.set(title="Mean predicted uplift", ylabel="Percentage points")
    score_ax.legend()

    if splits.test.is_open:
        raise RuntimeError("The test split must stay locked.")
    fig.suptitle("Criteo conversion: class transformation on validation")
    fig.tight_layout()
    output = Path("reports/t56-class-transformation-trap.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
