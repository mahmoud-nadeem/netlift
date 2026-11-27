"""T59: export the registered champion into a folder the Docker image can load.

Run on the machine that holds the MLflow registry. Writes ``artifacts/champion/``
(git-ignored):

- ``model/``: the registered model's MLflow ``pyfunc`` files
  (``models:/netlift-criteo-conversion-champion@champion`` by default);
- ``decile_table.json``: decile edges and within-arm bootstrap intervals from the
  champion's validation scores (T55, ADR 0009);
- ``manifest.json``: registry name, version and run ID, the model's feature columns,
  the abstention threshold, the validation metrics, and the SHA-256 of every file.

The validation scores and outcomes come from the champion refit
(``data/processed/t57-champion/``). The locked test split is not read. The image
copies the folder and sets ``NETLIFT_MODEL_DIR`` to it; no MLflow server is needed.

Example:
    python scripts/export_champion.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from netlift.decision.config import CONFIG_PATH, load_decision_settings
from netlift.decision.deciles import fit_decile_table
from netlift.utils.errors import ContractError
from netlift.utils.tracking import file_sha256, git_commit, git_dirty

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL_NAME = "netlift-criteo-conversion-champion"
ALIAS = "champion"
CHAMPION_DATA = ROOT / "data" / "processed" / "t57-champion"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument(
        "--scores", type=Path, default=CHAMPION_DATA / "validation_scores.parquet"
    )
    parser.add_argument(
        "--outcomes", type=Path, default=CHAMPION_DATA / "validation_outcomes.parquet"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional champion-challenger table JSON; the champion's rows become "
        "the /model metrics.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "champion")
    return parser.parse_args(argv)


def champion_metrics(path: Path | None, model: str) -> list[dict[str, Any]]:
    """The champion's validation metrics from a champion-challenger table JSON."""
    if path is None:
        return []
    table = json.loads(path.read_text(encoding="utf-8"))
    rows = table["metrics"].get(model, {})
    return [
        {
            "name": name,
            "value": row["value"],
            "ci_95": [row["ci_low"], row["ci_high"]],
            "status": "available",
        }
        for name, row in rows.items()
    ]


def main(argv: list[str] | None = None) -> None:
    """Download the registered model, fit the decile table and write the manifest."""
    import mlflow
    from mlflow.models import get_model_info

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    settings = load_decision_settings(args.config)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(args.name, args.alias)
    uri = f"models:/{args.name}@{args.alias}"
    info = get_model_info(uri)
    if info.signature is None:
        raise ContractError(
            f"{uri} has no signature, so its feature columns are unknown."
        )
    columns = list(info.signature.inputs.input_names())

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=uri, dst_path=str(args.out / "model")
    )

    scores = pd.read_parquet(args.scores)
    outcomes = pd.read_parquet(args.outcomes)
    joined = scores.merge(outcomes, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(scores) or len(joined) != len(outcomes):
        raise ContractError("Scores and outcomes must cover the same validation rows.")
    table = fit_decile_table(joined["score"], joined["y"], joined["t"], settings)
    (args.out / "decile_table.json").write_text(
        json.dumps(table.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    files = {
        str(path.relative_to(args.out)).replace("\\", "/"): file_sha256(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
    }
    family = version.tags.get("family", "uplift_forest")
    manifest = {
        "registered_model": args.name,
        "alias": args.alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_version": f"{args.name}/{version.version}",
        "family": family,
        "training_date": pd.Timestamp(version.creation_timestamp, unit="ms")
        .date()
        .isoformat(),
        "search_report_sha256": version.tags.get("search_report_sha256"),
        "model_git_commit": version.tags.get("git_commit"),
        "feature_columns": columns,
        "min_control_events_per_decile": settings.min_control_events_per_decile,
        "decile_split": table.split_name,
        "metrics": champion_metrics(args.metrics, family),
        "exported_by_commit": git_commit(),
        "exported_from_dirty_tree": git_dirty(),
        "test_split_opened": False,
        "files_sha256": files,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for row in table.rows:
        logger.info(
            "decile %d: uplift %.6f [%.6f, %.6f], control events %d",
            row.decile,
            row.uplift,
            row.ci_low,
            row.ci_high,
            row.control_events,
        )
    logger.info("exported %s version %s to %s", args.name, version.version, args.out)


if __name__ == "__main__":
    main()
