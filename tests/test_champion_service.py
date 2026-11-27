"""Tests for the T59 scoring service against a champion (fake model and real pyfunc)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from netlift.api.app import create_app
from netlift.api.champion_service import (
    MODEL_DIR_ENV,
    ChampionNetLiftService,
    service_from_environment,
)
from netlift.api.service import MockNetLiftService
from netlift.decision.deciles import DecileRow, DecileTable
from netlift.utils.errors import ContractError

COLUMNS = ("f0", "f1")
MIN = 10


def row(decile: int, low: float, high: float, control_events: int) -> DecileRow:
    """A decile row with only the fields the verdict reads made meaningful."""
    return DecileRow(
        decile=decile,
        rows=100,
        treated=80,
        control=20,
        treated_events=5,
        control_events=control_events,
        uplift=(low + high) / 2,
        ci_low=low,
        ci_high=high,
    )


# Deciles: 1 confident contact, 2 straddles, 3 too few control events, 4-9 straddle,
# 10 confident suppress.
TABLE = DecileTable(
    edges=(0.05, 0.04, 0.03, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03),
    rows=(
        row(1, 0.01, 0.03, 50),
        row(2, -0.01, 0.02, 50),
        row(3, 0.01, 0.03, MIN - 1),
        *(row(d, -0.01, 0.01, 50) for d in range(4, 10)),
        row(10, -0.03, -0.01, 50),
    ),
    split_name="validation",
    n_resamples=1000,
    confidence_level=0.95,
    seed=42,
)


class FeatureSum:
    """A fake champion: uplift = f0 + f1."""

    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        """Sum of the two features, per row."""
        assert list(model_input.columns) == list(COLUMNS)
        return model_input.sum(axis=1).to_numpy()


def client(model: Any = None) -> TestClient:
    """A test client on the champion service."""
    service = ChampionNetLiftService(
        model=model or FeatureSum(),
        feature_columns=COLUMNS,
        decile_table=TABLE,
        min_control_events=MIN,
        model_version="netlift-criteo-conversion-champion/1",
    )
    return TestClient(create_app(service))


def score(uplift: float, **extra: Any) -> dict[str, Any]:
    """POST /score for a customer whose fake uplift is ``uplift``."""
    body = {
        "customer_id": "c_1",
        "features": {"f0": uplift, "f1": 0.0, **extra},
        "contact_cost": 0.1,
        "conversion_margin": 50.0,
    }
    response = client().post("/score", json=body)
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data


@pytest.mark.parametrize(
    ("uplift", "decile", "verdict", "recommendation", "segment"),
    [
        (0.06, 1, "confident", "contact", "persuadable"),
        (0.045, 2, "insufficient_evidence", "no_action", "unknown"),
        (0.035, 3, "insufficient_control_data", "no_action", "unknown"),
        (0.005, 6, "insufficient_evidence", "no_action", "unknown"),
        (-0.05, 10, "confident", "suppress", "sleeping_dog"),
    ],
)
def test_each_verdict_rule_reaches_the_api(
    uplift: float, decile: int, verdict: str, recommendation: str, segment: str
) -> None:
    data = score(uplift)
    assert data["decile"] == decile
    assert data["verdict"] == verdict
    assert data["recommendation"] == recommendation
    assert data["segment_label"] == segment
    expected = TABLE.row(decile)
    assert data["uplift_ci_95"] == [expected.ci_low, expected.ci_high]
    assert data["uplift_score"] == pytest.approx(uplift)
    assert data["expected_incremental_profit"] == pytest.approx(50.0 * uplift - 0.1)
    assert data["model_version"] == "netlift-criteo-conversion-champion/1"
    assert data["top_drivers"] == []


def test_a_missing_or_extra_feature_is_refused_with_422() -> None:
    body = {
        "customer_id": "c_1",
        "features": {"f0": 0.1},
        "contact_cost": 0.1,
        "conversion_margin": 50.0,
    }
    assert client().post("/score", json=body).status_code == 422
    body["features"] = {"f0": 0.1, "f1": 0.0, "f2": 1.0}
    assert client().post("/score", json=body).status_code == 422


def test_health_and_model_say_live() -> None:
    api = client()
    health = api.get("/health").json()
    assert health["mode"] == "live" and health["model_loaded"] is True
    info = api.get("/model/info").json()
    assert info["is_mock"] is False and info["source"] == "registry"
    assert f"{MIN} control-arm conversions" in info["message"]


def test_a_non_finite_prediction_is_refused() -> None:
    class Broken:
        def predict(self, model_input: pd.DataFrame) -> np.ndarray:
            return np.array([np.nan])

    body = {
        "customer_id": "c_1",
        "features": {"f0": 0.1, "f1": 0.0},
        "contact_cost": 0.1,
        "conversion_margin": 50.0,
    }
    assert client(Broken()).post("/score", json=body).status_code == 422


def test_forbidden_feature_columns_are_refused() -> None:
    with pytest.raises(ContractError, match="forbidden"):
        ChampionNetLiftService(
            model=FeatureSum(),
            feature_columns=("f0", "exposure"),
            decile_table=TABLE,
            min_control_events=MIN,
            model_version="v",
        )


def test_without_the_environment_variable_the_app_uses_the_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    assert isinstance(service_from_environment(), MockNetLiftService)


def test_a_model_folder_without_a_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    with pytest.raises(ContractError, match="manifest.json"):
        service_from_environment()


def test_an_exported_folder_with_a_real_pyfunc_model_loads_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow.pyfunc

    class SumModel(mlflow.pyfunc.PythonModel):
        def predict(
            self, context: Any, model_input: pd.DataFrame, params: Any = None
        ) -> np.ndarray:
            return model_input.sum(axis=1).to_numpy()

    folder = tmp_path / "champion"
    mlflow.pyfunc.save_model(
        str(folder / "model"),
        python_model=SumModel(),
        input_example=pd.DataFrame({"f0": [0.1], "f1": [0.2]}),
    )
    (folder / "decile_table.json").write_text(
        json.dumps(TABLE.to_dict()), encoding="utf-8"
    )
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "feature_columns": list(COLUMNS),
                "min_control_events_per_decile": MIN,
                "model_version": "netlift-criteo-conversion-champion/1",
                "training_date": "2026-11-22",
                "metrics": [
                    {
                        "name": "qini",
                        "value": 0.158,
                        "ci_95": [0.128, 0.19],
                        "status": "available",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(MODEL_DIR_ENV, str(folder))
    api = TestClient(create_app())
    body = {
        "customer_id": "c_1",
        "features": {"f0": 0.05, "f1": 0.01},
        "contact_cost": 0.1,
        "conversion_margin": 50.0,
    }
    data = api.post("/score", json=body).json()
    assert data["verdict"] == "confident" and data["recommendation"] == "contact"
    info = api.get("/model/info").json()
    assert info["training_date"] == "2026-11-22"
    assert info["metrics"][0]["value"] == pytest.approx(0.158)
