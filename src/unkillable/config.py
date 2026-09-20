"""Config file loading for the Unkillable Library.

Settings live in a YAML file (``config.yml`` at the repo root) with sections
for each subsystem.  A minimal loader keeps the surface small — values are
read lazily and never re-parsed except on explicit reload.
"""

from __future__ import annotations

import os
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

_DEFAULT_PATH = Path("config.yml")


def load_config(path: str | os.PathLike[str] | None = None) -> dict:
    """Load the YAML config as a dict; returns {} on any missing/invalid file.

    ``None``/empty means "run with built-in defaults".
    """
    config_path = Path(path) if path else Path(os.environ.get("UNKILLABLE_CONFIG", _DEFAULT_PATH))
    try:
        if not config_path.exists():
            return {}
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            log.warning("%s: expected a mapping at the top level", config_path)
            return {}
        return data
    except Exception as exc:  # noqa: BLE001 - config must never crash the server
        log.warning("Failed to read config %s: %s", config_path, exc)
        return {}


def section(config: dict, name: str) -> dict:
    """Return one config section, or {} if absent/not a mapping."""
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}