import logging
import logging.config
import os
import json


def setup_logging(
    file_path="logging.json", level=logging.INFO, env_key="LOG_CFG"
):
    """Setup logging configuration"""
    path = file_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)
