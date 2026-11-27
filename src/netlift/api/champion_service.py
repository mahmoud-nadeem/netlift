"""The scoring service against the registered champion (T59, with T55's decision layer).

``ChampionNetLiftService`` implements the ``NetLiftService`` protocol of
``service.py``. ``POST /score`` predicts the customer's uplift with the champion,
places the score in its validation decile, and applies Interface 3's verdict rules to
that decile's held-out interval (``netlift.decision``).

The model arrives as a folder exported by ``scripts/export_champion.py``:

- ``model/``: the registered model's MLflow ``pyfunc`` files;
- ``decile_table.json``: the decile edges and intervals on the validation split;
- ``manifest.json``: registry name and version, run ID, feature columns, the
  abstention threshold and the SHA-256 of every file.

The container needs no MLflow server: ``mlflow.pyfunc.load_model`` reads the folder.

Still mock in this adapter: ``/policy/simulate`` (T61's replay simulator),
``/monitoring/drift`` (T64) and ``top_drivers`` (empty until T61's SHAP).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from netlift.api.models import (
    DriftReportResponse,
    HealthResponse,
    MetricEstimate,
    ModelInfoResponse,
    PolicySimulationRequest,
    PolicySimulationResponse,
    ScoreRequest,
)
from netlift.api.schema import ScoringResponse
from netlift.api.service import MockNetLiftService
from netlift.decision.deciles import DecileTable
from netlift.decision.verdict import decide
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)
MODEL_DIR_ENV = "NETLIFT_MODEL_DIR"
FORBIDDEN_COLUMNS = frozenset({"treatment", "exposure", "visit", "conversion"})


class UpliftPredictor(Protocol):
    """Anything with ``predict(DataFrame) -> uplift per row``: an MLflow pyfunc model."""

    def predict(self, model_input: pd.DataFrame) -> Any: ...


class ChampionNetLiftService:
    """Scores with the champion and decides with the held-out decile intervals."""

    def __init__(
        self,
        *,
        model: UpliftPredictor,
        feature_columns: Sequence[str],
        decile_table: DecileTable,
        min_control_events: int,
        model_version: str,
        training_date: date | None = None,
        metrics: Sequence[MetricEstimate] = (),
    ) -> None:
        if not feature_columns:
            raise ContractError("The champion needs at least one feature column.")
        forbidden = sorted(FORBIDDEN_COLUMNS & set(feature_columns))
        if forbidden:
            raise ContractError(f"Feature columns include forbidden ones: {forbidden}.")
        if min_control_events < 1:
            raise ContractError("min_control_events must be at least 1.")
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.decile_table = decile_table
        self.min_control_events = min_control_events
        self.model_version = model_version
        self.training_date = training_date
        self.metrics = list(metrics)
        self._mock = MockNetLiftService()
        self.ready = True

    @classmethod
    def from_bundle(cls, folder: Path) -> ChampionNetLiftService:
        """Load an exported champion folder (``scripts/export_champion.py``)."""
        import mlflow.pyfunc

        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        table = DecileTable.from_dict(
            json.loads((folder / "decile_table.json").read_text(encoding="utf-8"))
        )
        trained = manifest.get("training_date")
        return cls(
            model=mlflow.pyfunc.load_model(str(folder / "model")),
            feature_columns=manifest["feature_columns"],
            decile_table=table,
            min_control_events=int(manifest["min_control_events_per_decile"]),
            model_version=str(manifest["model_version"]),
            training_date=date.fromisoformat(trained) if trained else None,
            metrics=[MetricEstimate(**item) for item in manifest.get("metrics", [])],
        )

    def health(self) -> HealthResponse:
        """Ready once the champion and its decile table are loaded."""
        return HealthResponse(
            status="ready",
            ready=True,
            mode="live",
            model_loaded=True,
            message=f"Champion {self.model_version} is loaded.",
        )

    def _features(self, request: ScoreRequest) -> pd.DataFrame:
        """One-row frame in the champion's column order; refuse missing or extra columns."""
        given = set(request.features)
        missing = sorted(set(self.feature_columns) - given)
        extra = sorted(given - set(self.feature_columns))
        if missing or extra:
            raise ContractError(
                f"The champion needs exactly its feature columns. Missing: {missing}; "
                f"not expected: {extra}."
            )
        row = {name: [float(request.features[name])] for name in self.feature_columns}
        return pd.DataFrame(row, columns=list(self.feature_columns))

    def score(self, request: ScoreRequest) -> ScoringResponse:
        """Uplift, decile, held-out interval and Interface 3's verdict for one customer."""
        prediction = np.asarray(self.model.predict(self._features(request)))
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ContractError("The champion must return one finite uplift per row.")
        uplift = float(prediction[0])
        decile = int(self.decile_table.decile_of([uplift])[0])
        row = self.decile_table.row(decile)
        decision = decide(
            row.ci_low, row.ci_high, row.control_events, self.min_control_events
        )
        response = ScoringResponse(
            customer_id=request.customer_id,
            model_version=self.model_version,
            uplift_score=uplift,
            uplift_ci_95=(row.ci_low, row.ci_high),
            decile=decile,
            segment_label=decision.segment_label,
            expected_incremental_profit=request.conversion_margin * uplift
            - request.contact_cost,
            recommendation=decision.recommendation,
            verdict=decision.verdict,
            top_drivers=(),
            scored_at=datetime.now(UTC),
        )
        logger.info(
            "Scored with the champion",
            extra={
                "customer_id": request.customer_id,
                "model_version": self.model_version,
                "decile": decile,
                "verdict": response.verdict,
                "control_events_in_decile": row.control_events,
                "control_event_threshold": self.min_control_events,
            },
        )
        return response

    def model_info(self) -> ModelInfoResponse:
        """The champion's version, training date and validation metrics."""
        return ModelInfoResponse(
            model_version=self.model_version,
            training_date=self.training_date,
            metrics=self.metrics,
            source="registry",
            is_mock=False,
            message=(
                "Validation metrics of the registered champion (preliminary until T70). "
                f"Abstention threshold: {self.min_control_events} control-arm "
                "conversions per decile."
            ),
        )

    def drift_report(self) -> DriftReportResponse:
        """Still the mock: drift monitoring is T64's."""
        return self._mock.drift_report()

    def simulate(self, request: PolicySimulationRequest) -> PolicySimulationResponse:
        """Still the mock: the replay simulator is T61's."""
        return self._mock.simulate(request)


def service_from_environment() -> ChampionNetLiftService | MockNetLiftService:
    """The champion when ``NETLIFT_MODEL_DIR`` names a folder, otherwise the mock."""
    folder = os.environ.get(MODEL_DIR_ENV)
    if not folder:
        return MockNetLiftService()
    path = Path(folder)
    if not (path / "manifest.json").is_file():
        raise ContractError(
            f"{MODEL_DIR_ENV}={folder} has no manifest.json; export the champion with "
            "scripts/export_champion.py."
        )
    return ChampionNetLiftService.from_bundle(path)
