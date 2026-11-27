"""Leakage audit: outcomes, treatment and exposure never enter the feature matrix.

The forbidden columns are listed in config/data.yaml (section ``leakage``). A CI
test checks that the list equals ``contracts.FORBIDDEN_COLUMNS`` and runs the audit
on the real X built by ``prepare_criteo`` and ``prepare_hillstrom``.

Run on the real datasets:  python -m netlift.data.leakage [--dataset NAME]
Exit code: 0 clean, 1 leakage found, 2 configuration or data error.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Hashable, Iterable
from typing import Any

from netlift.data.contracts import HILLSTROM_MENS_VS_CONTROL
from netlift.data.loaders import Prepared, load_criteo, load_hillstrom
from netlift.data.pipeline import load_config
from netlift.utils.errors import ConfigError, ContractError, LeakageError
from netlift.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def forbidden_columns(cfg: dict[str, Any]) -> frozenset[str]:
    """Read the forbidden column names (lower-case) from the config."""
    section = cfg.get("leakage")
    if not isinstance(section, dict):
        raise ConfigError("config/data.yaml: missing the 'leakage' section.")
    cols = section.get("forbidden_columns")
    if (
        not isinstance(cols, list)
        or not cols
        or not all(isinstance(c, str) for c in cols)
    ):
        raise ConfigError("leakage.forbidden_columns must be a non-empty name list.")
    return frozenset(c.strip().lower() for c in cols)


def audit_columns(
    columns: Iterable[Hashable], forbidden: frozenset[str], *, source: str
) -> None:
    """Raise LeakageError if any column name is forbidden (case-insensitive)."""
    names = [str(c) for c in columns]
    found = sorted(n for n in names if n.strip().lower() in forbidden)
    if found:
        logger.error("%s: leakage, forbidden column(s) in X: %s", source, found)
        raise LeakageError(f"{source}: forbidden column(s) in X: {found}.")
    logger.info("%s: leakage audit passed (%d feature columns)", source, len(names))


def audit_prepared(
    prepared: Prepared, cfg: dict[str, Any], *, source: str = "X"
) -> None:
    """Audit the feature matrix of a Prepared dataset against the config."""
    audit_columns(prepared.X.columns, forbidden_columns(cfg), source=source)


def _loaders() -> dict[str, Callable[[], Prepared]]:
    return {
        "criteo": load_criteo,
        "hillstrom": lambda: load_hillstrom(HILLSTROM_MENS_VS_CONTROL),
    }


def main(argv: list[str] | None = None) -> int:
    """Audit the real datasets. Return 0 clean, 1 leakage, 2 config or data error."""
    parser = argparse.ArgumentParser(description="Leakage audit on the real datasets.")
    parser.add_argument(
        "--dataset", choices=["all", "criteo", "hillstrom"], default="all"
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        setup_logging(cfg)
        loaders = _loaders()
        names = sorted(loaders) if args.dataset == "all" else [args.dataset]
        for name in names:
            audit_prepared(loaders[name](), cfg, source=name)
    except LeakageError:
        return 1  # already logged by audit_columns
    except (ConfigError, ContractError, OSError) as exc:
        logger.error("Leakage audit could not run: %s", exc)
        return 2
    logger.info("Leakage audit passed for: %s", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
