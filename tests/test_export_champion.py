"""End-to-end test of scripts/export_champion.py on a temporary MLflow registry."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from netlift.api.champion_service import ChampionNetLiftService
from netlift.api.models import ScoreRequest
from netlift.utils.tracking import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """Import scripts/export_champion.py as a module."""
    path = ROOT / "scripts" / "export_champion.py"
    spec = importlib.util.spec_from_file_location("export_champion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_registered_champion_exports_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow
    import mlflow.pyfunc

    original = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        _export_and_score(tmp_path, mlflow)
    finally:
        mlflow.set_tracking_uri(original)


def _export_and_score(tmp_path: Path, mlflow: Any) -> None:
    """Register a small pyfunc model, export it and score through the service."""
    import mlflow.pyfunc

    mlflow.set_experiment("export-test")

    class Half(mlflow.pyfunc.PythonModel):
        def predict(
            self, context: Any, model_input: pd.DataFrame, params: Any = None
        ) -> np.ndarray:
            return (model_input["f0"] * 0.5).to_numpy()

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=Half(),
            input_example=pd.DataFrame({"f0": [0.1], "f1": [0.2]}),
        )
    version = mlflow.register_model(info.model_uri, "test-champion")
    client = mlflow.MlflowClient()
    client.set_model_version_tag("test-champion", version.version, "family", "forest")
    client.set_registered_model_alias("test-champion", "champion", version.version)

    rng = np.random.default_rng(0)
    n = 20_000
    score = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + 0.1 * t * (score > 1.0)).astype(int)
    row_id = np.arange(n, dtype=np.int64)
    pd.DataFrame({"row_id": row_id, "score": score}).to_parquet(
        tmp_path / "scores.parquet", index=False
    )
    pd.DataFrame({"row_id": row_id[::-1], "y": y[::-1], "t": t[::-1]}).to_parquet(
        tmp_path / "outcomes.parquet", index=False
    )
    out = tmp_path / "champion"
    load_script().main(
        [
            "--name",
            "test-champion",
            "--scores",
            str(tmp_path / "scores.parquet"),
            "--outcomes",
            str(tmp_path / "outcomes.parquet"),
            "--out",
            str(out),
        ]
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_columns"] == ["f0", "f1"]
    assert manifest["model_version"] == f"test-champion/{version.version}"
    assert manifest["family"] == "forest"
    assert manifest["test_split_opened"] is False
    for name, digest in manifest["files_sha256"].items():
        assert file_sha256(out / name) == digest
    table = json.loads((out / "decile_table.json").read_text(encoding="utf-8"))
    assert sum(row["rows"] for row in table["rows"]) == n

    service = ChampionNetLiftService.from_bundle(out)
    request = ScoreRequest(
        customer_id="c_1",
        features={"f0": 10.0, "f1": 0.0},
        contact_cost=0.1,
        conversion_margin=50.0,
    )
    response = service.score(request)
    assert response.uplift_score == pytest.approx(5.0)
    assert response.decile == 1
