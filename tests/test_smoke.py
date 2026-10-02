import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NETLIFT_DIR = ROOT / "src" / "netlift"
CONFIG_DIR = ROOT / "config"


def test_netlift_imports():
    """Verify netlift and every subpackage can be imported."""
    importlib.import_module("netlift")

    for package_dir in NETLIFT_DIR.iterdir():
        if package_dir.is_dir() and (package_dir / "__init__.py").exists():
            package_name = f"netlift.{package_dir.name}"
            importlib.import_module(package_name)


def test_yaml_configs_parse():
    """Verify every YAML configuration file parses successfully."""
    yaml_files = list(CONFIG_DIR.glob("*.yaml"))

    assert yaml_files, "No YAML configuration files found."

    for yaml_file in yaml_files:
        with yaml_file.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        assert data is not None, f"{yaml_file} is empty or invalid."


def test_kpis_contain_all_required_ids():
    """Verify config/kpis.yaml contains K1 through K11."""
    kpi_file = CONFIG_DIR / "kpis.yaml"

    with kpi_file.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    assert "kpis" in data
    assert isinstance(data["kpis"], list)

    actual_ids = {kpi["id"] for kpi in data["kpis"]}
    expected_ids = {f"K{i}" for i in range(1, 12)}

    assert actual_ids == expected_ids
