"""Tests for the T55 decision layer: verdict rules, decile table, policy modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from netlift.api.schema import ScoringResponse
from netlift.decision.config import (
    CONFIG_PATH,
    DecisionSettings,
    load_decision_settings,
)
from netlift.decision.deciles import DecileTable, fit_decile_table
from netlift.decision.policy import select_contacts
from netlift.decision.verdict import decide
from netlift.utils.errors import ContractError

MIN = 10

# --------------------------------------------------------------------------- #
# Verdict rules (Interface 3), one test per rule and per boundary
# --------------------------------------------------------------------------- #


def test_a_positive_interval_is_a_confident_contact() -> None:
    decision = decide(0.001, 0.004, control_events=MIN, min_control_events=MIN)
    assert (decision.verdict, decision.recommendation, decision.segment_label) == (
        "confident",
        "contact",
        "persuadable",
    )


def test_a_negative_interval_is_a_confident_suppress() -> None:
    decision = decide(-0.004, -0.001, control_events=50, min_control_events=MIN)
    assert (decision.verdict, decision.recommendation, decision.segment_label) == (
        "confident",
        "suppress",
        "sleeping_dog",
    )


def test_a_straddling_interval_is_insufficient_evidence() -> None:
    decision = decide(-0.001, 0.002, control_events=50, min_control_events=MIN)
    assert (decision.verdict, decision.recommendation, decision.segment_label) == (
        "insufficient_evidence",
        "no_action",
        "unknown",
    )


@pytest.mark.parametrize(("low", "high"), [(0.0, 0.003), (-0.003, 0.0), (0.0, 0.0)])
def test_an_interval_touching_zero_is_not_confident(low: float, high: float) -> None:
    decision = decide(low, high, control_events=50, min_control_events=MIN)
    assert decision.verdict == "insufficient_evidence"
    assert decision.recommendation == "no_action"


@pytest.mark.parametrize(("low", "high"), [(0.001, 0.004), (-0.004, -0.001)])
def test_too_few_control_events_abstain_even_with_a_clear_interval(
    low: float, high: float
) -> None:
    decision = decide(low, high, control_events=MIN - 1, min_control_events=MIN)
    assert (decision.verdict, decision.recommendation, decision.segment_label) == (
        "insufficient_control_data",
        "no_action",
        "unknown",
    )


def test_exactly_the_minimum_of_control_events_is_enough() -> None:
    assert decide(0.001, 0.004, MIN, MIN).verdict == "confident"


def test_every_decision_is_a_valid_scoring_response() -> None:
    """The rules never produce a response the Interface 3 schema would reject."""
    from datetime import UTC, datetime

    grid = np.linspace(-0.01, 0.01, 21)
    for low in grid:
        for high in grid[grid >= low]:
            for events in (0, MIN - 1, MIN, 500):
                decision = decide(float(low), float(high), events, MIN)
                assert not (
                    decision.segment_label == "sleeping_dog"
                    and decision.recommendation == "contact"
                )
                ScoringResponse(
                    customer_id="c",
                    model_version="v",
                    uplift_score=0.0,
                    uplift_ci_95=(float(low), float(high)),
                    decile=1,
                    segment_label=decision.segment_label,
                    expected_incremental_profit=0.0,
                    recommendation=decision.recommendation,
                    verdict=decision.verdict,
                    top_drivers=(),
                    scored_at=datetime.now(UTC),
                )


def test_an_invalid_interval_or_threshold_is_refused() -> None:
    with pytest.raises(ContractError, match="Invalid interval"):
        decide(0.002, 0.001, 50, MIN)
    with pytest.raises(ContractError, match="Invalid interval"):
        decide(float("nan"), 0.001, 50, MIN)
    with pytest.raises(ContractError, match="minimum"):
        decide(0.001, 0.002, 50, 0)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_the_repository_config_loads() -> None:
    settings = load_decision_settings(CONFIG_PATH)
    assert settings.min_control_events_per_decile >= 1
    assert settings.n_deciles == 10


def test_the_threshold_has_no_default(tmp_path: Path) -> None:
    path = tmp_path / "decision.yaml"
    path.write_text("n_resamples: 1000\n", encoding="utf-8")
    with pytest.raises(ContractError, match="min_control_events_per_decile"):
        load_decision_settings(path)


def test_unknown_settings_are_refused() -> None:
    with pytest.raises(ContractError, match="Unknown"):
        DecisionSettings.from_mapping(
            {"min_control_events_per_decile": 10, "typo_setting": 1}
        )


# --------------------------------------------------------------------------- #
# Decile table
# --------------------------------------------------------------------------- #


def synthetic(
    n: int = 40_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores whose top decile has a real positive effect and the bottom a negative one."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    base = 0.10
    effect = np.where(score > 1.3, 0.15, np.where(score < -1.3, -0.08, 0.0))
    y = (rng.random(n) < base + t * effect).astype(int)
    return score, y, t


SETTINGS = DecisionSettings(min_control_events_per_decile=MIN, n_resamples=500)


def test_deciles_order_scores_from_highest_to_lowest() -> None:
    score, y, t = synthetic()
    table = fit_decile_table(score, y, t, SETTINGS)
    decile = table.decile_of(score)
    assert decile.min() == 1 and decile.max() == 10
    assert score[decile == 1].min() >= score[decile == 2].max()
    assert sum(row.rows for row in table.rows) == len(score)
    assert all(abs(row.rows - len(score) / 10) <= 1 for row in table.rows)


def test_a_score_equal_to_an_edge_goes_to_the_higher_decile() -> None:
    score, y, t = synthetic()
    table = fit_decile_table(score, y, t, SETTINGS)
    assert table.decile_of([table.edges[0]])[0] == 1
    assert table.decile_of([table.edges[0] - 1e-12])[0] == 2
    assert table.decile_of([1e9, -1e9]).tolist() == [1, 10]


def test_the_intervals_find_the_planted_effects() -> None:
    score, y, t = synthetic()
    table = fit_decile_table(score, y, t, SETTINGS)
    assert table.row(1).ci_low > 0.0
    assert table.row(10).ci_high < 0.0
    assert table.row(5).ci_low < 0.0 < table.row(5).ci_high


def test_the_counts_and_uplift_are_exact() -> None:
    score, y, t = synthetic()
    table = fit_decile_table(score, y, t, SETTINGS)
    inside = table.decile_of(score) == 3
    row = table.row(3)
    assert row.control_events == int(y[inside & (t == 0)].sum())
    assert row.treated == int((inside & (t == 1)).sum())
    expected = y[inside & (t == 1)].mean() - y[inside & (t == 0)].mean()
    assert row.uplift == pytest.approx(expected)


def test_the_binomial_draws_match_a_row_level_within_arm_bootstrap() -> None:
    """The binomial shortcut has the spread of resampling rows within each arm."""
    score, y, t = synthetic()
    table = fit_decile_table(score, y, t, SETTINGS)
    inside = table.decile_of(score) == 1
    yt, yc = y[inside & (t == 1)], y[inside & (t == 0)]
    rng = np.random.default_rng(1)
    draws = [
        rng.choice(yt, len(yt)).mean() - rng.choice(yc, len(yc)).mean()
        for _ in range(2000)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    row = table.row(1)
    assert row.ci_low == pytest.approx(low, abs=0.01)
    assert row.ci_high == pytest.approx(high, abs=0.01)


def test_the_table_is_reproducible_and_round_trips_through_json() -> None:
    score, y, t = synthetic()
    first = fit_decile_table(score, y, t, SETTINGS)
    second = fit_decile_table(score, y, t, SETTINGS)
    assert first == second
    assert DecileTable.from_dict(json.loads(json.dumps(first.to_dict()))) == first


def test_the_table_is_never_built_on_test() -> None:
    score, y, t = synthetic()
    with pytest.raises(ContractError, match="locked"):
        fit_decile_table(score, y, t, SETTINGS, split_name="test")


def test_a_decile_without_control_rows_is_refused() -> None:
    score, y, t = synthetic()
    t = t.copy()
    t[score > np.quantile(score, 0.9)] = 1
    with pytest.raises(ContractError, match="Decile 1"):
        fit_decile_table(score, y, t, SETTINGS)


# --------------------------------------------------------------------------- #
# Policy modes
# --------------------------------------------------------------------------- #

UPLIFT = np.array([0.05, 0.04, 0.03, 0.02, 0.01, -0.02, 0.06])
ACTION = np.array(
    ["contact", "contact", "no_action", "contact", "contact", "suppress", "no_action"]
)


def test_fixed_budget_contacts_the_top_k_contact_recommendations() -> None:
    chosen = select_contacts(
        UPLIFT,
        ACTION,
        "fixed_budget",
        budget=3.0,
        contact_cost=1.0,
        conversion_margin=10.0,
    )
    assert chosen.tolist() == [True, True, False, True, False, False, False]


def test_fixed_budget_leaves_budget_unspent_rather_than_contact_others() -> None:
    chosen = select_contacts(
        UPLIFT,
        ACTION,
        "fixed_budget",
        budget=100.0,
        contact_cost=1.0,
        conversion_margin=10.0,
    )
    assert chosen.sum() == 4
    assert not chosen[ACTION != "contact"].any()


def test_profit_maximising_contacts_above_cost_over_margin() -> None:
    chosen = select_contacts(
        UPLIFT, ACTION, "profit_maximising", contact_cost=1.0, conversion_margin=40.0
    )
    assert chosen.tolist() == [True, True, False, False, False, False, False]


def test_suppression_only_contacts_everyone_but_suppress() -> None:
    chosen = select_contacts(
        UPLIFT, ACTION, "suppression_only", contact_cost=1.0, conversion_margin=10.0
    )
    assert chosen.tolist() == [True, True, True, True, True, False, True]


@pytest.mark.parametrize(
    "mode", ["fixed_budget", "profit_maximising", "suppression_only"]
)
def test_no_mode_ever_contacts_a_suppress(mode: str) -> None:
    chosen = select_contacts(
        UPLIFT, ACTION, mode, budget=1e9, contact_cost=0.01, conversion_margin=1e6
    )
    assert not chosen[ACTION == "suppress"].any()


def test_bad_mode_and_budget_are_refused() -> None:
    with pytest.raises(ContractError, match="mode"):
        select_contacts(
            UPLIFT, ACTION, "fixed_target", contact_cost=1.0, conversion_margin=1.0
        )
    with pytest.raises(ContractError, match="budget"):
        select_contacts(
            UPLIFT, ACTION, "fixed_budget", contact_cost=1.0, conversion_margin=1.0
        )
