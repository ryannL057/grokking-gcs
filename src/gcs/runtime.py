"""Explicit roots for the preserved experiment runners."""
import os
from pathlib import Path


def data_root() -> Path:
    value=os.environ.get('GCS_DATA_ROOT')
    if not value:
        raise RuntimeError('Use scripts/experiment.py --data-root PATH; historical data is not searched automatically.')
    return Path(value).resolve()
