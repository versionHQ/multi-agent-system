import json
import sqlite3
import datetime
from typing import Any, Dict, List, Optional

from versionhq._utils.logger import Logger
from versionhq.storage.utils import fetch_db_storage_path

storage_path = fetch_db_storage_path()
default_db_name = "task_outputs"


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
                    CREATE TABLE IF NOT EXISTS task_outputs (
                        task_id TEXT PRIMARY KEY,
                        output JSON,
                        task_index INTEGER,
                        inputs JSON,
                        was_replayed BOOLEAN,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"SQL database initialization failed: {str(e)}", color="red")


    def add(self, task, output: Dict[str, Any], task_index: int, was_replayed: bool = False, inputs: Dict[str, Any] = {}):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                        """INSERT OR REPLACE INTO task_outputs
                            (task_id, output, task_index, inputs, was_replayed, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """,
                     (str(task.id), json.dumps(output), task_index, json.dumps(inputs), was_replayed, datetime.datetime.now())
                )
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"SAVING TASK OUTPUTS ERROR: {e}", color="red")


    def update(self, task_index: int, **kwargs):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                fields, values = [], []
                for k, v in kwargs.items():
                    fields.append(f"{k} = ?")
                    values.append(json.dumps(v) if isinstance(v, dict) else v)

                query = f"UPDATE latest_kickoff_task_outputs SET {', '.join(fields)} WHERE task_index = ?"
                values.append(task_index)
                cursor.execute(query, tuple(values))
                conn.commit()

                if cursor.rowcount == 0:
                    self._logger.log(
                        level="warning", message=f"No row found with task_index {task_index}. No update performed.", color="yellow",
                    )

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"UPDATE TASK OUTPUTS ERROR: {e}", color="red")


    def load(self) -> Optional[List[Dict[str, Any]]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT *
                FROM task_outputs
                ORDER BY task_index
                """)

                rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = {
                        "task_id": row[0],
                        "output": json.loads(row[1]),
                        "task_index": row[2],
                        "inputs": json.loads(row[3]),
                        "was_replayed": row[4],
                        "timestamp": row[5],
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
                cursor.execute("DELETE FROM task_outputs")
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"ERROR: Failed to delete all: {e}", color="red")
