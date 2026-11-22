"""NetLift Streamlit dashboard entry point."""

import csv
import hashlib
import io
import json
import zipfile

import pandas as pd
import streamlit as st

from dashboard.api_client import post
from dashboard.model_evidence import model_evidence
from dashboard.monitoring import monitoring
from dashboard.policy_comparison import policy_comparison

ECONOMICS_FIELDS = ("contact_cost", "conversion_margin")
DEMO_CSV = b"""customer_id,f0,f3
customer-001,0.2,1.5
mock-insufficient-evidence:002,0.4,0.5
mock-insufficient-control-data:003,0.5,0.7
mock-sleeping-dog:004,0.6,0.8
"""
VERDICTS = {
    "confident": "Confident",
    "insufficient_evidence": "Insufficient evidence",
    "insufficient_control_data": "Insufficient control data",
}
ACTIONS = {"contact": "Contact", "suppress": "Suppress", "no_action": "No action"}


def make_batch_csv(upload: bytes, cost: float, margin: float) -> bytes:
    """Validate CSV structure and apply one set of economics to every row."""
    reader = csv.reader(
        io.StringIO(upload.decode("utf-8-sig"), newline=""), strict=True
    )
    headers = [field.strip() for field in next(reader, [])]
    if not all(headers) or len(headers) != len({field.casefold() for field in headers}):
        raise ValueError("Use unique, non-empty CSV column names.")

    features = [
        name for name in headers if name not in ("customer_id", *ECONOMICS_FIELDS)
    ]
    if "customer_id" not in headers or not features:
        raise ValueError("Include customer_id and at least one numeric feature column.")

    non_economic_columns = [
        index for index, name in enumerate(headers) if name not in ECONOMICS_FIELDS
    ]
    customer_id_index = headers.index("customer_id")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [headers[index] for index in non_economic_columns] + list(ECONOMICS_FIELDS)
    )
    row_count = 0
    for row in reader:
        if not row:
            continue
        if len(row) != len(headers):
            raise ValueError(
                f"CSV line {reader.line_num} has the wrong number of fields."
            )
        if not row[customer_id_index].strip():
            raise ValueError(f"CSV line {reader.line_num} is missing a customer ID.")
        writer.writerow([row[index] for index in non_economic_columns] + [cost, margin])
        row_count += 1
    if row_count == 0:
        raise ValueError("Add at least one customer row.")
    return output.getvalue().encode("utf-8")


def score_batch(payload: bytes) -> tuple[pd.DataFrame, dict]:
    """Submit the batch and load the two in-memory API response files."""
    response = post(
        "/score/batch", payload, content_type="text/csv", accept="application/zip"
    )
    try:
        with zipfile.ZipFile(io.BytesIO(response)) as archive:
            scores = pd.read_csv(
                io.BytesIO(archive.read("scores.csv")),
                dtype={"customer_id": str},
                keep_default_na=False,
            )
            summary = json.loads(archive.read("summary.json"))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RuntimeError(
            "The scoring API returned an invalid batch archive."
        ) from exc
    return scores, summary


def show_results(scores: pd.DataFrame, summary: dict) -> None:
    """Show scores, abstention verdicts and per-customer evidence."""
    if summary["is_mock"]:
        st.info("Mock data · illustrative scores")
    st.caption(f"Model version: {summary['model_version']}")
    row_count, evidence, controls = st.columns(3)
    counts = summary["verdict_counts"]
    row_count.metric("Customers scored", summary["row_count"])
    evidence.metric("Insufficient evidence", counts.get("insufficient_evidence", 0))
    controls.metric(
        "Insufficient control data", counts.get("insufficient_control_data", 0)
    )

    table = pd.DataFrame(
        {
            "Customer ID": scores["customer_id"],
            "Uplift (pp)": scores["uplift_score"] * 100,
            "Recommendation": scores["recommendation"].map(ACTIONS),
            "Verdict": scores["verdict"].map(VERDICTS),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Uplift (pp)": st.column_config.NumberColumn(format="%.2f")},
    )

    selected = st.selectbox(
        "Customer detail",
        options=range(len(scores)),
        format_func=lambda index: scores.iloc[index]["customer_id"],
    )
    row = scores.iloc[selected]
    if row["verdict"] == "insufficient_evidence":
        st.info("Insufficient evidence · the uplift interval includes zero.")
    elif row["verdict"] == "insufficient_control_data":
        st.warning(
            "Insufficient control data · the segment has too few control events."
        )
    low, high = json.loads(row["uplift_ci_95"])
    st.write(
        f"**{ACTIONS[row['recommendation']]}** · "
        f"{row['uplift_score'] * 100:+.2f} pp uplift · "
        f"95% decile-level interval [{low * 100:+.2f}, {high * 100:+.2f}] pp"
    )
    st.caption(
        f"Decile {row['decile']} · Segment: {row['segment_label'].replace('_', ' ')} · "
        f"Scored at {row['scored_at']}"
    )
    st.metric(
        "Expected incremental profit per contact",
        f"{row['expected_incremental_profit']:,.2f}",
    )
    drivers = json.loads(row["top_drivers"])
    if drivers:
        st.dataframe(drivers, hide_index=True, width="stretch")


def campaign_planner() -> None:

    source = st.radio(
        "Customer source", ["Upload CSV", "Demo customers"], horizontal=True
    )
    if source == "Upload CSV":
        upload = st.file_uploader(
            "Customer CSV",
            type=["csv"],
            help="Include customer_id and numeric feature columns. Campaign inputs set cost and margin for all rows.",
        )
        data = upload.getvalue() if upload is not None else None
    else:
        data = DEMO_CSV

    cost_col, margin_col = st.columns(2)
    cost = cost_col.number_input(
        "Cost per contact", min_value=0.01, value=5.0, format="%.2f"
    )
    margin = margin_col.number_input(
        "Margin per conversion", min_value=0.01, value=200.0, format="%.2f"
    )

    if data is None:
        st.info("Upload a customer CSV or select Demo customers to run batch scoring.")
    run_key = (
        (hashlib.sha256(data).hexdigest(), cost, margin) if data is not None else None
    )
    if st.button("Score customers", type="primary", disabled=data is None):
        st.session_state.pop("scored_batch", None)
        try:
            payload = make_batch_csv(data or b"", cost, margin)
            with st.spinner("Scoring customers…"):
                scores, summary = score_batch(payload)
        except (csv.Error, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.session_state["scored_batch"] = (run_key, scores, summary)

    result = st.session_state.get("scored_batch")
    if result is not None and result[0] == run_key:
        show_results(result[1], result[2])
    elif result is not None and run_key is not None:
        st.info(
            "Inputs changed. Score customers again to see results for these inputs."
        )


def main() -> None:
    st.set_page_config(page_title="NetLift Dashboard", layout="wide")
    st.title("NetLift")
    planner, comparison, evidence, monitor = st.tabs(
        ["Campaign Planner", "Policy Comparison", "Model Evidence", "Monitoring"]
    )
    with planner:
        campaign_planner()
    with comparison:
        policy_comparison()
    with evidence:
        model_evidence()
    with monitor:
        monitoring()


if __name__ == "__main__":
    main()
