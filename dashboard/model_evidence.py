"""Present model metadata and metrics supplied by the NetLift API."""

import streamlit as st

from dashboard.api_client import get_json

METRIC_LABELS = {
    "qini_coefficient": ("Qini coefficient", 1),
    "auuc": ("AUUC", 1),
    "uplift_at_5pct": ("Uplift@5% (pp)", 100),
    "uplift_at_10pct": ("Uplift@10% (pp)", 100),
    "uplift_at_20pct": ("Uplift@20% (pp)", 100),
    "uplift_at_30pct": ("Uplift@30% (pp)", 100),
}
STATUS_LABELS = {
    "available": "Available",
    "illustrative_mock": "Mock data",
    "unavailable": "Unavailable",
}


def metric_row(metric: dict) -> dict[str, str]:
    """Keep each metric's reported interval and format uplift in points."""
    name = metric["name"]
    label, scale = METRIC_LABELS.get(name, (name.replace("_", " "), 1))
    value = metric["value"]
    interval = metric["ci_95"]
    status = metric["status"]
    return {
        "Metric": label,
        "Estimate": "—" if value is None else f"{value * scale:.4f}",
        "95% interval": (
            "—"
            if interval is None
            else f"[{interval[0] * scale:.4f}, {interval[1] * scale:.4f}]"
        ),
        "Status": STATUS_LABELS.get(status, status.replace("_", " ").capitalize()),
    }


def model_evidence() -> None:
    """Load active model evidence on demand and show gaps explicitly."""
    if st.button("Load model evidence", type="primary"):
        st.session_state.pop("model_evidence", None)
        try:
            with st.spinner("Loading model evidence…"):
                st.session_state["model_evidence"] = get_json("/model/info")
        except (RuntimeError, TypeError) as exc:
            st.error(str(exc))
            return

    evidence = st.session_state.get("model_evidence")
    if evidence is None:
        return

    if evidence["is_mock"]:
        st.info("Mock data · illustrative model metrics and intervals")
    st.subheader(evidence["model_version"])
    st.caption(
        f"Source: {evidence['source']} · "
        f"Training date: {evidence['training_date'] or 'Unavailable'}"
    )
    rows = [metric_row(metric) for metric in evidence["metrics"]]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No evaluation metrics are available for this model version.")
    st.caption(
        "Detailed evidence unavailable from this endpoint: Qini curves, calibration, "
        "decile results, SHAP, negative-control results and model comparisons."
    )
