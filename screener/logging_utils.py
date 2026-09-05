"""Logging setup shared by the CLI and the module entry points."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, to stderr."""
    global _CONFIGURED
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    if _CONFIGURED:
        logging.getLogger().setLevel(numeric)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)
    # Requests/urllib3 are chatty at DEBUG and drown out our own logs.
    logging.getLogger("urllib3").setLevel(max(numeric, logging.WARNING))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
