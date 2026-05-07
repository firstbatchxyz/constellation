"""Configuration helpers for run artifacts and JSON configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

RUNS_ENV_VAR = "CONSTELLATION_RUNS_DIR"
DEFAULT_RUNS_DIR = "~/constellation-runs"


def runs_dir() -> Path:
    return Path(os.environ.get(RUNS_ENV_VAR, DEFAULT_RUNS_DIR)).expanduser()


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return expand_config(value)


def expand_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, str):
        return value.replace("{runs_dir}", str(runs_dir()))
    return value


def artifact_path(value: str | Path) -> Path:
    path = Path(str(value).replace("{runs_dir}", str(runs_dir()))).expanduser()
    if path.is_absolute():
        return path
    return runs_dir() / path
