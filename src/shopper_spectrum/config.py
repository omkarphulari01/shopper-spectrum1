"""Configuration loading and project-wide logging setup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root = three levels up from this file (src/shopper_spectrum/config.py)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


@dataclass
class Config:
    """Lightweight wrapper around the YAML config with attribute access."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a path under `paths` relative to the project root."""
        return ROOT / self.data["paths"][key]


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config file into a Config object."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("shopper_spectrum")
