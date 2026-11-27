"""One place that configures logging, driven by the config's logging section."""

from __future__ import annotations

import logging
from typing import Any

from netlift.utils.errors import ConfigError

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(cfg: dict[str, Any] | None = None) -> None:
    """Configure the root logger from cfg["logging"] (level, format)."""
    section = (cfg or {}).get("logging") or {}
    level = logging.getLevelName(str(section.get("level", "INFO")).upper())
    if not isinstance(level, int):
        raise ConfigError(f"Unknown logging level: {section.get('level')!r}.")
    logging.basicConfig(
        level=level, format=section.get("format", DEFAULT_FORMAT), force=True
    )
