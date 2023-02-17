import logging
import os
from typing import Optional, Dict

import yaml


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_config(path: str) -> Dict:
        with open(path, 'r') as f:
            cfg: Dict = yaml.safe_load(f)
        return cfg


def setup_config(path: Optional[str] = None) -> Dict:
    try:
        if path is not None:
            config = ConfigReader.read_config(path)
        else:
            config = ConfigReader.read_config(os.environ['CONFIG_FILE_PATH'])
    except (FileNotFoundError, KeyError, TypeError) as e:
        default: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_config.yml')
        config = ConfigReader.read_config(default)

    return config


base_config: Dict = setup_config()


def set_up_logger() -> None:
    logger = logging.getLogger()

    # Get handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('logfile.log')

    # Log formatting
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Set logging level
    logger.setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.INFO)
