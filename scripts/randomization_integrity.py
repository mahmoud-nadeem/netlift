"""Randomization integrity check (spec Section 20).

For every covariate: standardized mean difference (SMD) between treated and control.
Then: treatment-predictability AUC (naive and group-aware CV) and a propensity overlap check.
Writes a dated report to reports/randomization-integrity.md.

Thresholds live in config/randomization.yaml (pre-registered), not in this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

log = logging.getLogger("randomization_integrity")

DEFAULTS: dict[str, Any] = {
    "smd_threshold": 0.1,
    "auc_fail": 0.55,
    "overlap_eps": 0.01,
    "overlap_max_frac": 0.01,
    "seed": 42,
    "n_boot": 200,
    "auc_sample": 500_000,
}

HILLSTROM_FEATURES = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
HILLSTROM_CATEGORICAL = ["history_segment", "zip_code", "channel"]
HILLSTROM_ARM = "segment"
HILLSTROM_ARM_VALUES = {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
# Criteo `exposure` is post-treatment (zero for all control rows) and is NOT a covariate.
CRITEO_FEATURES = [f"f{i}" for i in range(12)]


def load_config(path: str) -> dict[str, Any]:
    """Load thresholds from YAML, falling back to defaults for missing keys."""
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        cfg.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    else:
        log.warning("Config %s not found, using built-in defaults", path)
    return cfg


def read_table(path: str, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from a CSV or Parquet file."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns)


def smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """Standardized mean difference with pooled SD."""
    pooled_var = (np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return float((x_t.mean() - x_c.mean()) / np.sqrt(pooled_var))


def cross_fit(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, seed: int
) -> np.ndarray:
    """Out-of-fold treatment probabilities; group-aware if groups is given."""
    model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    if groups is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[
        :, 1
    ]


def auc_with_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """AUC with a 95% percentile bootstrap CI."""
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def run_comparison(
    name: str, X: pd.DataFrame, treated: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    """SMDs, treatment AUC (naive and group-aware) and overlap for one comparison."""
    if X.isna().any().any():
        raise ValueError(f"{name}: missing values in features")
    treated = np.asarray(treated, dtype=bool)
    n_t, n_c = int(treated.sum()), int((~treated).sum())
    if n_t == 0 or n_c == 0:
        raise ValueError(f"{name}: one arm is empty ({n_t} treated, {n_c} control)")

    se = float(
        np.sqrt(1 / n_t + 1 / n_c)
    )  # approx. sampling noise of an SMD under no imbalance
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": col,
                "mean_treated": float(x[treated].mean()),
                "mean_control": float(x[~treated].mean()),
                "smd": smd(x[treated], x[~treated]),
            }
        )
    smd_df = pd.DataFrame(rows)
    smd_df["abs_smd"] = smd_df["smd"].abs()
    smd_df["z"] = smd_df["smd"] / se
    smd_df = smd_df.sort_values("abs_smd", ascending=False).reset_index(drop=True)
    max_smd = float(smd_df["abs_smd"].iloc[0])
    top_feature = str(smd_df["feature"].iloc[0])
    flagged = smd_df.loc[smd_df["abs_smd"] > cfg["smd_threshold"], "feature"].tolist()

    # Treatment-predictability AUC on a random sample for speed on large data.
    y = treated.astype(int)
    seed = int(cfg["seed"])
    Xs, ys = X, y
    if cfg["auc_sample"] and len(y) > cfg["auc_sample"]:
        idx = np.random.default_rng(seed).choice(
            len(y), size=int(cfg["auc_sample"]), replace=False
        )
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    # Rows with identical feature profiles are kept together in the same CV fold, so the model
    # cannot gain AUC just by memorising duplicated profiles.
    groups = pd.util.hash_pandas_object(Xs, index=False).to_numpy()
    log.info("%s: computing AUC (n=%d)", name, len(ys))
    p_naive = cross_fit(Xs, ys, None, seed)
    p_group = cross_fit(Xs, ys, groups, seed)
    auc_n, lo_n, hi_n = auc_with_ci(ys, p_naive, int(cfg["n_boot"]), seed)
    auc_g, lo_g, hi_g = auc_with_ci(ys, p_group, int(cfg["n_boot"]), seed)

    eps = float(cfg["overlap_eps"])
    outside = float(np.mean((p_group < eps) | (p_group > 1 - eps)))
    overlap_ok = outside <= float(cfg["overlap_max_frac"])

    reasons = []
    if max_smd > cfg["smd_threshold"]:
        reasons.append(
            f"{len(flagged)} covariate(s) with |SMD| > {cfg['smd_threshold']}: {', '.join(flagged[:5])}"
        )
    if lo_g > cfg["auc_fail"]:
        reasons.append(
            f"treatment predictable from covariates (AUC CI lower bound {lo_g:.4f} > {cfg['auc_fail']})"
        )
    if not overlap_ok:
        reasons.append(
            f"overlap violated ({outside:.2%} of rows have propensity outside [{eps}, {1 - eps}])"
        )
    if reasons:
        status = "FAIL"
    elif lo_g > 0.5 or max(abs(z) for z in smd_df["z"]) > 5:
        status = "NOTE"
    else:
        status = "PASS"
    log.info(
        "%s: max|SMD|=%.4f (%s) AUC naive=%.4f grouped=%.4f [%.4f, %.4f] outside=%.4f -> %s",
        name,
        max_smd,
        top_feature,
        auc_n,
        auc_g,
        lo_g,
        hi_g,
        outside,
        status,
    )
    return {
        "name": name,
        "n_treated": n_t,
        "n_control": n_c,
        "smd": smd_df,
        "se": se,
        "max_smd": max_smd,
        "top_feature": top_feature,
        "flagged": flagged,
        "auc_naive": auc_n,
        "auc_naive_lo": lo_n,
        "auc_naive_hi": hi_n,
        "auc": auc_g,
        "auc_lo": lo_g,
        "auc_hi": hi_g,
        "n_auc": len(ys),
        "n_unique_profiles": len(np.unique(groups)),
        "p_min": float(p_group.min()),
        "p_max": float(p_group.max()),
        "outside": outside,
        "overlap_ok": overlap_ok,
        "reasons": reasons,
        "status": status,
    }


def hillstrom_comparisons(path: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """All three pairwise arm comparisons on Hillstrom."""
    df = read_table(path, HILLSTROM_FEATURES + [HILLSTROM_ARM])
    arms = df[HILLSTROM_ARM]
    unknown = set(arms.unique()) - HILLSTROM_ARM_VALUES
    if unknown:
        raise ValueError(f"Unexpected Hillstrom arm values: {unknown}")
    X_all = pd.get_dummies(
        df[HILLSTROM_FEATURES], columns=HILLSTROM_CATEGORICAL, dtype=float
    ).astype(float)
    pairs = [
        ("Hillstrom: Mens E-Mail vs No E-Mail", "Mens E-Mail", "No E-Mail"),
        ("Hillstrom: Womens E-Mail vs No E-Mail", "Womens E-Mail", "No E-Mail"),
        ("Hillstrom: Mens E-Mail vs Womens E-Mail", "Mens E-Mail", "Womens E-Mail"),
    ]
    results = []
    for name, t, c in pairs:
        keep = arms.isin([t, c]).to_numpy()
        results.append(
            run_comparison(
                name,
                X_all[keep].reset_index(drop=True),
                (arms[keep] == t).to_numpy(),
                cfg,
            )
        )
    return results


def criteo_comparison(
    path: str, nrows: int | None, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Treatment vs control on Criteo (f0-f11 only; `exposure` excluded by design)."""
    df = read_table(path, CRITEO_FEATURES + ["treatment"])
    if nrows and nrows < len(df):
        # Random sample, not head(): the file is ordered, so the first rows are all one arm.
        df = df.sample(n=nrows, random_state=int(cfg["seed"])).reset_index(drop=True)
    if not set(df["treatment"].unique()) <= {0, 1}:
        raise ValueError("Criteo treatment column is not 0/1")
    return run_comparison(
        "Criteo: treatment vs control",
        df[CRITEO_FEATURES],
        df["treatment"].to_numpy() == 1,
        cfg,
    )


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    results: list[dict[str, Any]], cfg: dict[str, Any], argv: list[str]
) -> str:
    """Build the Markdown report."""
    failed = [r for r in results if r["status"] == "FAIL"]
    noted = [r for r in results if r["status"] == "NOTE"]
    out = [
        "# Randomization Integrity Check",
        "",
        f"Report date: {datetime.now(tz=UTC).date().isoformat()}. Generated by `scripts/randomization_integrity.py`.",
        "",
    ]
    if failed:
        out += [
            "## Verdict: RANDOMIZATION FAILED. STOP, DO NOT MODEL (spec Section 20)",
            "",
        ]
        for r in failed:
            out.append(f"- **{r['name']}**: " + "; ".join(r["reasons"]) + ".")
        out += [
            "",
            (
                "The team must be told now: effect estimates on these comparisons cannot be treated as causal "
                "until this is explained."
            ),
            "",
        ]
    else:
        out += [
            "## Verdict: PASS",
            "",
            (
                "Every comparison satisfies the pre-registered criteria (SMD threshold, treatment-predictability "
                "AUC, overlap). Modelling may proceed."
            ),
            "",
        ]
        if noted:
            out += ["Notes (detectable but below the failure thresholds):", ""]
            for r in noted:
                out.append(
                    f"- **{r['name']}**: largest |SMD| is {r['max_smd']:.4f} on `{r['top_feature']}`, "
                    f"about {r['max_smd'] / r['se']:.0f}x the sampling noise expected under perfect randomization; "
                    f"group-aware AUC {r['auc']:.4f} (95% CI {r['auc_lo']:.4f} to {r['auc_hi']:.4f}). "
                    f"Small, but not zero. All SMDs stay under {cfg['smd_threshold']}."
                )
            out.append("")
    summary = pd.DataFrame(
        [
            {
                "comparison": r["name"],
                "n_treated": r["n_treated"],
                "n_control": r["n_control"],
                "max_abs_smd": r["max_smd"],
                "auc_group_cv": r["auc"],
                "auc_ci_95": f"[{r['auc_lo']:.4f}, {r['auc_hi']:.4f}]",
                "rows_outside_overlap": r["outside"],
                "status": r["status"],
            }
            for r in results
        ]
    )
    out += ["## Summary", "", md_table(summary), ""]
    for r in results:
        out += [
            f"## {r['name']}",
            "",
            f"- Status: **{r['status']}**",
            f"- Max |SMD|: {r['max_smd']:.4f} (`{r['top_feature']}`); expected sampling noise of one SMD: {r['se']:.4f}",
            f"- Covariates above the SMD threshold ({cfg['smd_threshold']}): {', '.join(r['flagged']) or 'none'}",
            (
                f"- Treatment AUC, group-aware CV (used for the verdict): {r['auc']:.4f} "
                f"(95% bootstrap CI {r['auc_lo']:.4f} to {r['auc_hi']:.4f}, n={r['n_auc']:,}, "
                f"{r['n_unique_profiles']:,} distinct feature profiles)"
            ),
            (
                f"- Treatment AUC, plain CV (for comparison): {r['auc_naive']:.4f} "
                f"(95% CI {r['auc_naive_lo']:.4f} to {r['auc_naive_hi']:.4f})"
            ),
            (
                f"- Overlap: cross-fitted propensity ranges from {r['p_min']:.4f} to {r['p_max']:.4f}; "
                f"{r['outside']:.4%} of rows outside [{cfg['overlap_eps']}, {1 - cfg['overlap_eps']}] "
                f"-> {'OK' if r['overlap_ok'] else 'VIOLATED'}"
            ),
            "",
            "SMD for every covariate, sorted by |SMD| (`z` = SMD / expected sampling noise):",
            "",
            md_table(r["smd"][["feature", "mean_treated", "mean_control", "smd", "z"]]),
            "",
        ]
    out += [
        "## Method and pre-registered thresholds",
        "",
        (
            f"- SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2). Categorical covariates are "
            f"one-hot encoded, one SMD per level. Flag if |SMD| > {cfg['smd_threshold']} (spec: conventional 0.1, set in config)."
        ),
        (
            f"- Treatment AUC: 5-fold out-of-fold predictions from a gradient-boosted classifier using covariates only. "
            f"Rows with identical covariate profiles are kept in the same fold (StratifiedGroupKFold) so duplicated profiles "
            f"cannot inflate the AUC. FAIL if the lower bound of the 95% bootstrap CI exceeds {cfg['auc_fail']} "
            f"(team choice: the spec says only 'materially above 0.5')."
        ),
        (
            f"- Overlap: out-of-fold propensity scores must lie inside [{cfg['overlap_eps']}, {1 - cfg['overlap_eps']}] for at "
            f"least {1 - cfg['overlap_max_frac']:.0%} of rows."
        ),
        (
            "- Excluded: outcomes (visit, conversion, spend) and Criteo `exposure` (post-treatment, zero for every control row "
            "by construction, per the spec)."
        ),
        f"- Seed {cfg['seed']}; AUC computed on at most {cfg['auc_sample']:,} random rows per comparison.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python " + " ".join(argv),
        "```",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hillstrom", help="Path to Hillstrom CSV/Parquet")
    ap.add_argument("--criteo", help="Path to Criteo CSV/Parquet")
    ap.add_argument(
        "--criteo-nrows",
        type=int,
        default=None,
        help="Use a random sample of N Criteo rows (testing)",
    )
    ap.add_argument("--config", default="config/randomization.yaml")
    ap.add_argument("--out", default="reports/randomization-integrity.md")
    args = ap.parse_args()
    if not (args.hillstrom or args.criteo):
        ap.error("give at least one of --hillstrom / --criteo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)

    results: list[dict[str, Any]] = []
    if args.hillstrom:
        results += hillstrom_comparisons(args.hillstrom, cfg)
    if args.criteo:
        results.append(criteo_comparison(args.criteo, args.criteo_nrows, cfg))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(results, cfg, sys.argv[:]), encoding="utf-8")
    log.info("Wrote %s", out)
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
