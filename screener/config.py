"""Configuration loading.

Config is a plain nested mapping loaded from ``config.yaml``, wrapped in a
small accessor that supports dotted paths with defaults. Secrets never live in
the config file - they come from the environment (``.env`` locally, GitHub
Secrets in CI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"

_MISSING = object()


class Config:
    """Dotted-path accessor over the parsed YAML config."""

    def __init__(self, data: Mapping[str, Any], path: Path | None = None) -> None:
        self._data = dict(data)
        self.path = path

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"Missing config key: {dotted!r}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(path={self.path!r}, keys={sorted(self._data)})"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from ``path``, ``$SCREENER_CONFIG``, or ``config.yaml``."""
    resolved = Path(path or os.environ.get("SCREENER_CONFIG", DEFAULT_CONFIG_PATH))
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. Copy the repo's config.yaml and edit it."
        )
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    db_override = os.environ.get("SCREENER_DB_PATH")
    if db_override:
        data.setdefault("storage", {})["db_path"] = db_override
    return Config(data, resolved)


def load_dotenv_if_present(path: str = ".env") -> None:
    """Best-effort ``.env`` load. A missing file is not an error."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    if Path(path).exists():
        load_dotenv(path)


def require_env(names: Iterable[str]) -> dict[str, str]:
    """Fetch required env vars, raising a single message listing all missing."""
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if not value:
            missing.append(name)
        else:
            values[name] = value
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )
    return values
