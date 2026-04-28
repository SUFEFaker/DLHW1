from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _sanitize_key(key: str) -> str:
    return key.replace(".", "__dot__")


def _restore_key(key: str) -> str:
    return key.replace("__dot__", ".")


def save_state_dict(path: str | Path, state_dict: dict[str, np.ndarray]) -> None:
    sanitized = {_sanitize_key(name): array for name, array in state_dict.items()}
    np.savez(Path(path), **sanitized)


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as checkpoint:
        return {_restore_key(name): checkpoint[name] for name in checkpoint.files}


def resolve_metadata_path(checkpoint_path: str | Path, metadata_path: str | Path | None) -> Path:
    if metadata_path is not None:
        return Path(metadata_path)
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(".json")
