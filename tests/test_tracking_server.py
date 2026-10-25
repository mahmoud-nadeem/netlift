"""Integration test: the tracking wrapper against a real MLflow tracking server (T41).

Starts ``mlflow server`` as a subprocess on a free port, with a SQLite backend and
an artifact folder in a temporary directory, logs a run through the wrapper, then
reads every piece back with ``MlflowClient``. This is the setup the team uses
day to day (``docs/experiment-tracking.md``), so it is tested as it runs.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from netlift.evaluation.contracts import (
    MetricResult,
    log_metric_result,
    resamples_artifact_file,
)
from netlift.utils.tracking import experiment_name, run_name, start_run

STARTUP_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 15.0
FINGERPRINT = "cd" * 32


def _free_port() -> int:
    """A port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until_healthy(url: str, process: subprocess.Popen[bytes], log: Path) -> None:
    """Poll the server's health endpoint until it answers OK, or fail with its log."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"mlflow server exited early:\n{log.read_text(errors='replace')}"
            )
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    pytest.fail(
        f"mlflow server not healthy after {STARTUP_TIMEOUT_S} s:\n"
        f"{log.read_text(errors='replace')}"
    )


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Stop the server and its worker processes, gracefully first."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        # The server runs in its own process group, so the break reaches its
        # workers too and does not reach pytest.
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=SHUTDOWN_TIMEOUT_S)


@pytest.fixture
def tracking_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A running MLflow server; yields its URL and its artifact folder."""
    port = _free_port()
    artifacts = tmp_path / "mlartifacts"
    log = tmp_path / "server.log"
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "--artifacts-destination",
        artifacts.as_uri(),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    platform_options: dict[str, object] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    url = f"http://127.0.0.1:{port}"
    with log.open("wb") as log_handle:
        process = subprocess.Popen(  # type: ignore[call-overload]
            command, stdout=log_handle, stderr=subprocess.STDOUT, **platform_options
        )
        try:
            _wait_until_healthy(url, process, log)
            yield url, artifacts
        finally:
            _stop(process)


def _metric_result() -> MetricResult:
    return MetricResult(
        name="uplift_at_k",
        value=0.0123,
        ci_low=0.0081,
        ci_high=0.0170,
        confidence_level=0.95,
        n_resamples=1000,
        seed=42,
        library="scikit-uplift",
        library_version="0.5.1",
        resamples=tuple(0.0123 + (i % 7 - 3) / 1e4 for i in range(1000)),
        resample_fingerprint=FINGERPRINT,
        settings={"k": 0.1, "strategy": "overall"},
    )


def test_wrapper_logs_a_run_to_a_real_server(
    tracking_server: tuple[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, artifacts = tracking_server
    monkeypatch.setenv("MLFLOW_TRACKING_URI", url)
    result = _metric_result()

    with start_run(
        run_name("T41", "server-check"),
        task="T41",
        experiment=experiment_name("synthetic", "conversion"),
        seed=42,
    ) as run:
        key = log_metric_result(run, "K2", result)

    client = MlflowClient(tracking_uri=url)
    experiment = client.get_experiment_by_name("netlift-synthetic-conversion")
    assert experiment is not None
    (found,) = client.search_runs([experiment.experiment_id])
    data = found.data

    assert key == "K2_uplift_at_10pct_overall"
    assert data.metrics[key] == pytest.approx(0.0123)
    assert data.metrics[f"{key}_ci_low"] == pytest.approx(0.0081)
    assert data.metrics[f"{key}_ci_high"] == pytest.approx(0.0170)
    assert data.params[f"{key}_library"] == "scikit-uplift==0.5.1"
    assert data.params[f"{key}_seed"] == "42"
    assert data.params[f"{key}_resample_fingerprint"] == FINGERPRINT
    assert data.tags["task"] == "T41"
    assert data.tags["mlflow.runName"] == "T41-server-check"

    downloaded = client.download_artifacts(
        found.info.run_id, resamples_artifact_file(key), str(tmp_path / "download")
    )
    document = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    assert document["name"] == "uplift_at_k"
    assert document["seed"] == 42
    assert document["n_resamples"] == 1000
    assert document["resample_fingerprint"] == FINGERPRINT
    assert MetricResult(**document) == result
    # The server, not the client, wrote the file into its artifact folder.
    assert list(artifacts.rglob(f"{key}.json"))
