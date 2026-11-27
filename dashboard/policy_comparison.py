"""Equal-contact-budget policy comparison on the validation split."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.api_client import post

POLICY_LABELS = {
    "uplift": "Uplift",
    "response_model": "Response model",
    "random": "Random",
    "blanket": "Blanket",
}
CONCLUSION_LABELS = {
    "uplift_better": "Uplift shows more incremental conversions",
    "response_model_better": "Response model shows more incremental conversions",
    "not_distinguishable": "No clear difference",
}


def simulate(inputs: dict) -> dict:
    """Run a what-if through the API using the selected validation scenario."""
    payload = post(
        "/policy/simulate",
        json.dumps(inputs).encode("utf-8"),
        content_type="application/json",
        accept="application/json",
    )
    result = json.loads(payload)
    if not isinstance(result, dict) or not result.get("policies"):
        raise ValueError("The API returned an invalid policy comparison.")
    return result


def comparison_chart(policies: list[dict], budget_contacts: float) -> alt.LayerChart:
    """Plot cumulative conversions and intervals on a shared contact-count axis."""
    points = pd.DataFrame(
        [
            {
                "policy": POLICY_LABELS[policy["policy"]],
                "contacts": point["fraction_targeted"] * policy["contacted"],
                "conversions": point["cumulative_incremental_conversions"],
                "low": point["ci_95"][0],
                "high": point["ci_95"][1],
            }
            for policy in policies
            for point in policy["curve"]
        ]
    )
    base = alt.Chart(points).encode(x=alt.X("contacts:Q", title="Contacts"))
    color = alt.Color("policy:N", title="Policy")
    bands = base.mark_area(opacity=0.12).encode(
        y=alt.Y("low:Q", title="Cumulative incremental conversions"),
        y2="high:Q",
        color=alt.Color("policy:N", legend=None),
    )
    lines = base.mark_line(point=True).encode(
        y=alt.Y("conversions:Q", title="Cumulative incremental conversions"),
        color=color,
        strokeDash=alt.StrokeDash("policy:N", title="Policy"),
        tooltip=[
            "policy:N",
            alt.Tooltip("contacts:Q", format=",.0f"),
            alt.Tooltip("conversions:Q", format=",.2f"),
            alt.Tooltip("low:Q", title="95% lower", format=",.2f"),
            alt.Tooltip("high:Q", title="95% upper", format=",.2f"),
        ],
    )
    limit = (
        alt.Chart(pd.DataFrame({"contacts": [budget_contacts]}))
        .mark_rule(color="#64748b", strokeDash=[5, 4])
        .encode(
            x="contacts:Q",
            tooltip=alt.Tooltip("contacts:Q", title="Budget limit", format=",.2f"),
        )
    )
    return (bands + lines + limit).properties(height=320)


def show_comparison(result: dict) -> None:
    """Display the returned policy results and the uplift–response contrast."""
    if result["is_mock"]:
        st.info("Mock data · illustrative conversions and intervals")
    st.caption(
        f"Scenario: {result['scenario_id']} · Split: {result['data_slice']} · "
        f"Model: {result['model_version']} · Simulation: {result['simulation_id']}"
    )
    comparison = result["comparison"]
    low, high = comparison["ci_95"]
    difference = comparison["incremental_conversions_difference"]
    st.write(
        f"**{CONCLUSION_LABELS[comparison['conclusion']]}** · "
        f"Uplift minus response model: {difference:+,.2f} conversions "
        f"(95% interval [{low:,.2f}, {high:,.2f}])."
    )
    budget, cost, margin = (
        result["budget"],
        result["contact_cost"],
        result["conversion_margin"],
    )
    policies = result["policies"]
    rows = []
    for policy in policies:
        low, high = policy["incremental_conversions_ci_95"]
        rows.append(
            {
                "Policy": POLICY_LABELS[policy["policy"]],
                "Contacts": policy["contacted"],
                "Total cost": policy["total_cost"],
                "Incremental conversions": policy["incremental_conversions"],
                "95% interval": f"[{low:,.2f}, {high:,.2f}]",
                "Incremental profit": policy["incremental_profit"],
                "Profit per unit spent": policy["profit_per_unit_spent"],
                "Budget status": (
                    "Over budget" if policy["total_cost"] > budget else "Within budget"
                ),
            }
        )
    st.caption(
        f"Budget: {budget:,.2f} · Cost per contact: {cost:,.2f} · "
        f"Margin per conversion: {margin:,.2f}. "
        "Blanket is the full-coverage reference."
    )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.2f")
            for name in (
                "Total cost",
                "Incremental conversions",
                "Incremental profit",
                "Profit per unit spent",
            )
        },
    )
    if any(policy["curve"] for policy in policies):
        st.altair_chart(comparison_chart(policies, budget / cost), width="stretch")
        st.caption("The dashed vertical line marks the contact budget.")


def policy_comparison() -> None:
    """Collect simulation inputs and keep results tied to the submitted inputs."""
    scenario, model, seed_col = st.columns([2, 3, 1])
    scenario_id = scenario.text_input("Scenario ID", "campaign-demo").strip()
    model_version = model.text_input("Model version", "netlift-mock-2026.11.04").strip()
    seed = seed_col.number_input("Seed", min_value=0, value=42, step=1)
    budget_col, cost_col, margin_col = st.columns(3)
    budget = budget_col.number_input(
        "Contact budget", min_value=0.01, value=20000.0, step=500.0, format="%.2f"
    )
    cost = cost_col.number_input(
        "Cost per contact",
        min_value=0.01,
        value=5.0,
        format="%.2f",
        key="comparison_cost",
    )
    margin = margin_col.number_input(
        "Margin per conversion",
        min_value=0.01,
        value=200.0,
        format="%.2f",
        key="comparison_margin",
    )
    st.caption("Validation split · equal-contact-budget comparison")
    inputs = {
        "scenario_id": scenario_id,
        "data_slice": "validation",
        "model_version": model_version,
        "seed": seed,
        "budget": budget,
        "contact_cost": cost,
        "conversion_margin": margin,
    }
    if st.button("Compare policies", type="primary"):
        st.session_state.pop("policy_comparison", None)
        if not scenario_id or not model_version:
            st.error("Enter a scenario ID and model version.")
        elif budget < cost:
            st.error("The budget must cover at least one contact.")
        else:
            try:
                with st.spinner("Comparing policies…"):
                    result = simulate(inputs)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["policy_comparison"] = (inputs, result)

    saved = st.session_state.get("policy_comparison")
    if saved is not None:
        if saved[0] == inputs:
            show_comparison(saved[1])
        else:
            st.info("Inputs changed. Compare policies again to update the results.")
