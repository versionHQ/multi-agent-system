import appdirs
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)


def fetch_db_storage_path() -> str:
    directory_name = get_project_directory_name()
    data_dir = Path(appdirs.user_data_dir(appname=directory_name, appauthor="Version IO Sdn Bhd", version=None, roaming=False))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def get_project_directory_name() -> str:
    """
    Returns the current project directory name
    """
    project_directory_name = os.environ.get("STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    else:
        cwd = Path.cwd()
        project_directory_name = cwd.name
        return project_directory_name
