"""One real tracked run, so the MLflow UI is not empty and the wrapper is proven.

Fits a two-model uplift estimator on Hillstrom and logs it through
netlift.utils.tracking. Requires the dataset:

    python src/netlift/data/download.py
    python scripts/demo_tracking_run.py

This run deliberately does NOT call log_kpi. K2 is first reported in milestone
4 against the locked test split, and a smoke run logged as a KPI result would
sit in the record looking like a finding. It is tagged purpose=smoke so it can
be filtered out later.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, uplift_auc_score
from sklift.models import TwoModels

from netlift.utils.tracking import start_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "raw" / "hillstrom.csv"
SEED = 42

FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
REQUIRED = [*FEATURES, "segment", "visit"]
TREATED_ARM = "womens e-mail"
CONTROL_ARM = "no e-mail"


def main() -> int:
    """Fit, score and log one run. Returns a process exit code."""
    if not DATASET.exists():
        logger.error(
            "Dataset not found at %s. Run src/netlift/data/download.py first.", DATASET
        )
        return 1

    frame = pd.read_csv(DATASET)

    # Normalise case rather than assume it. Copies of this dataset differ in
    # whether the headers are capitalised, and a KeyError three lines later
    # does not tell you that is what happened.
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        logger.error(
            "Dataset is missing expected columns %s. Columns present: %s",
            missing,
            list(frame.columns),
        )
        return 1

    # Two arms only, so the comparison is a clean treated-versus-control one.
    arm = frame["segment"].astype(str).str.strip().str.lower()
    frame = frame[arm.isin([TREATED_ARM, CONTROL_ARM])].copy()
    if frame.empty:
        logger.error(
            "No rows in the expected arms. Values found in 'segment': %s",
            sorted(arm.unique()),
        )
        return 1

    arm = frame["segment"].astype(str).str.strip().str.lower()
    treatment = (arm == TREATED_ARM).astype(int)
    outcome = frame["visit"].astype(int)

    features = pd.get_dummies(frame[FEATURES], drop_first=True)

    # Stratified jointly on (treatment, outcome): stratifying on the outcome
    # alone can leave the arms unbalanced in the test split, which moves the
    # measured uplift for a reason that has nothing to do with the model.
    strata = treatment.astype(str) + "_" + outcome.astype(str)
    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
        features, outcome, treatment, test_size=0.3, random_state=SEED, stratify=strata
    )

    with start_run(
        "smoke: two-model, logistic base, Hillstrom womens vs control",
        dataset=DATASET,
        seed=SEED,
        tags={"purpose": "smoke"},
    ) as run:
        run.log_params(
            {
                "learner": "two_model_vanilla",
                "base_estimator": "logistic_regression",
                "outcome": "visit",
                "arms": "womens_email_vs_none",
                "test_size": 0.3,
                "stratify": "treatment_x_outcome",
            }
        )

        model = TwoModels(
            estimator_trmnt=LogisticRegression(max_iter=1000),
            estimator_ctrl=LogisticRegression(max_iter=1000),
            method="vanilla",
        )
        model.fit(x_train, y_train, t_train)
        predicted = model.predict(x_test)

        run.log_metrics(
            {
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "treated_share_train": float(t_train.mean()),
                "qini_demo": float(qini_auc_score(y_test, predicted, t_test)),
                "auuc_demo": float(uplift_auc_score(y_test, predicted, t_test)),
            }
        )

    logger.info(
        "Run logged. Open the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
