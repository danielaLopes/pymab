import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any


def setup_logging(
    file_path: str | Path = "logging.json",
    level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> None:
    """Set up logging from a JSON config path, falling back to basic config."""

    path = Path(file_path)
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)
    if path.exists():
        with path.open() as f:
            config: dict[str, Any] = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)
