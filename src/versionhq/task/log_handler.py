from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from versionhq.storage.task_output_storage import TaskOutputSQLiteStorage


class ExecutionLog(BaseModel):
    task_id: str
    output: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    task_index: int
    inputs: Dict[str, Any] = Field(default_factory=dict)
    was_replayed: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)



class TaskOutputStorageHandler:

    def __init__(self):
        self.storage = TaskOutputSQLiteStorage()


    def update(self, task, task_index: int, was_replayed: bool = False, inputs: Dict[str, Any] = {}) -> None:
        """
        task: task instance
        """
        saved_outputs = self.load()
        if saved_outputs is None:
            raise ValueError("Logs cannot be None")

        self.add(task, task_index, was_replayed, inputs)


    def add(self, task, task_index: int, was_replayed: bool = False, inputs: Dict[str, Any] = {}) -> None:
        from versionhq.task.model import Task

        output_to_store = dict()

        if isinstance(task, Task):
            output_to_store = dict(
                description=str(task.description),
                raw=str(task.output.raw),
                responsible_agent=str(task.processed_by_agents),
            )

        self.storage.add(task=task, output=output_to_store, task_index=task_index, was_replayed=was_replayed, inputs=inputs)


    def reset(self) -> None:
        self.storage.delete_all()


    def load(self) -> Optional[List[Dict[str, Any]]]:
        return self.storage.load()
