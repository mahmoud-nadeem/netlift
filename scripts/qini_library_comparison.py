"""
NetLift — T18 supporting experiment.
Compares Qini scores from scikit-uplift vs causalml on the SAME
predictions, to check whether the two libraries agree.

Run with the project's pinned environment:
    pip install -r requirements.txt
    python scripts/qini_library_comparison.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklift.metrics as skm
import causalml.metrics as cmm
# ---------------------------------------------------------------
# 1. Synthetic randomized treatment/control data
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 4000

X = rng.normal(0, 1, size=(n, 1))
treatment = rng.integers(0, 2, size=n)  # 0/1, randomized 50/50

true_tau = 0.15 * (X[:, 0] > 0.3).astype(float) - 0.05 * (X[:, 0] < -0.5).astype(float)
base_rate = 0.08 + 0.02 * X[:, 0]
p1 = np.clip(base_rate + true_tau, 0.001, 0.999)
p0 = np.clip(base_rate, 0.001, 0.999)
p = np.where(treatment == 1, p1, p0)
y = (rng.random(n) < p).astype(int)

print(f"n={n}  treated={treatment.sum()}  control={n - treatment.sum()}")
print(f"overall conversion rate: {y.mean():.4f}\n")

# ---------------------------------------------------------------
# 2. A real T-learner (two logistic models), shared by both libraries
# ---------------------------------------------------------------
Xt, yt = X[treatment == 1], y[treatment == 1]
Xc, yc = X[treatment == 0], y[treatment == 0]
m1 = LogisticRegression().fit(Xt, yt)
m0 = LogisticRegression().fit(Xc, yc)
uplift_pred = m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1]

# ---------------------------------------------------------------
# 3. Score the SAME predictions with both libraries
# ---------------------------------------------------------------
sklift_qini = skm.qini_auc_score(y_true=y, uplift=uplift_pred, treatment=treatment)

df = pd.DataFrame({"y": y, "w": treatment, "uplift_pred": uplift_pred})
causalml_qini_raw  = cmm.qini_score(df, outcome_col="y", treatment_col="w", normalize=False)["uplift_pred"]
causalml_qini_norm = cmm.qini_score(df, outcome_col="y", treatment_col="w", normalize=True)["uplift_pred"]

print("=== scikit-uplift ===")
print(f"qini_auc_score:                 {sklift_qini:.4f}")
print()
print("=== causalml ===")
print(f"qini_score (normalize=False):   {causalml_qini_raw:.4f}")
print(f"qini_score (normalize=True):    {causalml_qini_norm:.4f}")
print()
print("Conclusion: the two numbers do NOT match, because 'normalize'")
print("means a different operation in each library (see docs/decisions/")
print("0001-uplift-library.md for the explanation).")