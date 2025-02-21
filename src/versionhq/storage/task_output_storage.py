import json
import sqlite3
import datetime
from typing import Any, Dict, List, Optional

from versionhq._utils.logger import Logger
from versionhq.storage.utils import fetch_db_storage_path

storage_path = fetch_db_storage_path()
default_db_name = "task_output"


class TaskOutputSQLiteStorage:
    """
    An SQLite storage class to handle storing task outputs.
    """

    def __init__(self, db_path: str = f"{storage_path}/{default_db_name}.db") -> None:
        self.db_path = db_path
        self._logger = Logger(verbose=True)
        self._initialize_db()


    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table.
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_output (
                        task_id TEXT PRIMARY KEY,
                        output JSON,
                        inputs JSON,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"SQL database initialization failed: {str(e)}", color="red")


    def add(self, task, output: Dict[str, Any], inputs: Dict[str, Any] = {}):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                        """INSERT OR REPLACE INTO task_output
                            (task_id, output, inputs, timestamp)
                            VALUES (?, ?, ?, ?)
                        """,
                     (str(task.id), json.dumps(output), json.dumps(inputs), datetime.datetime.now())
                )
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"SAVING TASK OUTPUT ERROR: {e}", color="red")


    def update(self, task_id: str, **kwargs):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                fields, values = [], []
                for k, v in kwargs.items():
                    fields.append(f"{k} = ?")
                    values.append(json.dumps(v) if isinstance(v, dict) else v)

                query = f"UPDATE task_output SET {', '.join(fields)} WHERE task = ?"
                values.append(task_id)
                cursor.execute(query, tuple(values))
                conn.commit()

                if cursor.rowcount == 0:
                    self._logger.log(
                        level="warning", message=f"No row found with task_id {task_id}. No update performed.", color="yellow",
                    )

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"UPDATE TASK OUTPUTS ERROR: {e}", color="red")


    def load(self) -> Optional[List[Dict[str, Any]]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT *
                FROM task_output
                ORDER BY task_id
                """)

                rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = {
                        "task_id": row[0],
                        "output": json.loads(row[1]),
                        "inputs": json.loads(row[2]),
                        "timestamp": row[3],
                    }
                    results.append(result)
                return results

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"LOADING TASK OUTPUTS ERROR: {e}", color="red")
            return None


    def delete_all(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM task_output")
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"ERROR: Failed to delete all: {e}", color="red")



class TaskOutputStorageHandler:
    """A class to task output storage."""

    from versionhq.task.model import Task

    def __init__(self):
        self.storage = TaskOutputSQLiteStorage()


    def update(self, task: Task, inputs: Dict[str, Any] = {}) -> None:
        saved_outputs = self.load()
        if saved_outputs is None:
            raise ValueError("Logs cannot be None")

        self.add(task, inputs)


    def add(self, task: Task, inputs: Dict[str, Any] = {}) -> None:
        output_to_store = dict(
            id=str(task.id),
            description=str(task.description),
            raw=str(task.output.raw),
            responsible_agents=str(task.processed_agents),
            tokens=task.output._tokens,
            latency=task.output.latency,
            score=task.output.aggregate_score if task.output.aggregate_score else "None",
        )
        self.storage.add(task=task, output=output_to_store, inputs=inputs)


    def reset(self) -> None:
        self.storage.delete_all()


    def load(self) -> Optional[List[Dict[str, Any]]]:
        return self.storage.load()
