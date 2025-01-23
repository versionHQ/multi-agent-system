from enum import Enum


class TaskOutputFormat(str, Enum):
    """Enum that represents the output format of a task."""

    JSON = "json"
    PYDANTIC = "pydantic model"
    RAW = "raw"
