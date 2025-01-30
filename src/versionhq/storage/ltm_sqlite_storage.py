import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from versionhq._utils.logger import Logger
from versionhq.storage.utils import fetch_db_storage_path


class LTMSQLiteStorage:
    """
    An updated SQLite storage class for LTM data storage.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path(fetch_db_storage_path()) / "ltm_storage.db")

        self.db_path = db_path
        self._logger: Logger = Logger(verbose=True)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()


    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score REAL
                    )
                """
                )

                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(
                level="error",
                message=f"MEMORY ERROR: An error occurred during database initialization: {str(e)}",
                color="red",
            )

    def save(self, task_description: str, metadata: Dict[str, Any], datetime: str, score: int | float) -> None:
        """
        Saves data to the LTM table with error handling.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                VALUES (?, ?, ?, ?)
            """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._logger.log(
                level="error",
                message=f"MEMORY ERROR: An error occurred while saving to LTM: {str(e)}",
                color="red",
            )


    def load(self, task_description: str, latest_n: int) -> Optional[List[Dict[str, Any]]]:
        """
        Queries the LTM table by task description with error handling.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                """,
                    (task_description,),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]

        except sqlite3.Error as e:
            self._logger.log(
                level="error",
                message=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None


    def reset(self) -> None:
        """
        Resets the LTM table with error handling.
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(
                level="error",
                message=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {str(e)}",
                color="red",
            )
        return None
