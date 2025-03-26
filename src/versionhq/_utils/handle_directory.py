import os
import datetime
from pathlib import Path


def handle_directory(directory_name: str = None, filename: str = None, ext: str = 'png') -> Path:
    """Creates and returns the absolute file path"""

    os.makedirs(directory_name, exist_ok=True)

    date = str(datetime.datetime.now().strftime('%j'))
    cwd = Path.cwd()
    DIRECTORY = cwd / f'{directory_name}/{filename}_{date}.{ext}'

    return DIRECTORY
