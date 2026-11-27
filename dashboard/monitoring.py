"""Show the latest feature, score, and replay-effect drift report."""

import streamlit as st

from dashboard.api_client import get_json
from dashboard.model_evidence import STATUS_LABELS


def display_drift(report: dict) -> None:
    """Render metrics and distinguish missing reports from observed values."""
    if report["status"] == "not_available":
        st.info("No drift report is available yet.")
        return

    if report["is_mock"]:
        st.info("Mock data · illustrative drift and data quality")
    model_version = report["model_version"] or "Unavailable"
    generated_at = report["generated_at"] or "Unavailable"
    st.caption(f"Model version: {model_version} · Report time: {generated_at}")

    st.subheader("Feature drift")
    if report["feature_drift"]:
        st.dataframe(
            [
                {
                    "Feature": feature["feature"],
                    "PSI": feature["psi"],
                    "KS statistic": feature["ks_statistic"],
                    "Status": STATUS_LABELS[feature["status"]],
                }
                for feature in report["feature_drift"]
            ],
            hide_index=True,
            column_config={
                "PSI": st.column_config.NumberColumn(format="%.3f"),
                "KS statistic": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    else:
        st.caption("Feature drift is unavailable.")

    scores, quality = st.columns(2)
    with scores:
        st.subheader("Score distribution")
        distribution = report["score_drift"]
        if distribution is None or distribution["status"] == "unavailable":
            st.caption("Score distribution is unavailable.")
        else:
            st.dataframe(
                [
                    {"Statistic": label, "Uplift (pp)": 100 * distribution[key]}
                    for key, label in (
                        ("mean", "Mean"),
                        ("standard_deviation", "Standard deviation"),
                        ("p05", "5th percentile"),
                        ("median", "Median"),
                        ("p95", "95th percentile"),
                    )
                ],
                hide_index=True,
                column_config={
                    "Uplift (pp)": st.column_config.NumberColumn(format="%.3f")
                },
            )

    with quality:
        st.subheader("Data quality")
        data = report["data_quality"]
        if data is None or data["status"] == "unavailable":
            st.caption("Data quality metrics are unavailable.")
        else:
            null_rate = data["max_null_rate"]
            st.metric(
                "Maximum null rate",
                "Unavailable" if null_rate is None else f"{null_rate:.2%}",
            )
            st.metric("Range violations", data["range_violations"])
            st.metric("Schema changes", data["schema_changes"])

    st.subheader("Effect drift · Simulated replay")
    effect = report["effect_drift"]
    if effect is None:
        st.caption("Replay effect drift is unavailable.")
        return
    if effect["source"] == "mock_replay_output":
        st.caption("Mock replay output")
    metric = effect["metric"]
    scale = 100 if metric == "incremental_conversions_rate" else 1
    unit = " pp" if scale == 100 else ""
    low, high = effect["ci_95"]
    st.write(
        f"**{metric.replace('_', ' ').capitalize()}** · "
        f"Current {effect['current_estimate'] * scale:.3f}{unit} · "
        f"Reference {effect['reference_estimate'] * scale:.3f}{unit}"
    )
    st.write(
        f"Current minus reference: {effect['difference'] * scale:+.3f}{unit} "
        f"(95% interval [{low * scale:+.3f}, {high * scale:+.3f}]{unit})."
    )


def monitoring() -> None:
    """Fetch the latest monitoring report on demand."""
    if st.button("Load drift report", type="primary"):
        st.session_state.pop("drift_report", None)
        try:
            with st.spinner("Loading drift report…"):
                st.session_state["drift_report"] = get_json("/monitor/drift")
        except (RuntimeError, TypeError) as exc:
            st.error(str(exc))
            return

    report = st.session_state.get("drift_report")
    if report is not None:
        display_drift(report)
