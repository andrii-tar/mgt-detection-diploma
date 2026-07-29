from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def resolve_data_root() -> Path:
    load_dotenv()
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value is None:
        raise RuntimeError("TEXT_DETECTION_DATA is not set")
    return Path(value)
