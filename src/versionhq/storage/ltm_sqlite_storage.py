import json
import datetime
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
                        datetime REAL,
                        data TEXT,
                        metadata TEXT
                    )
                """
                )
                conn.commit()

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"MEMORY ERROR: An error occurred during database initialization: {str(e)}", color="red")


    def save(self, data: Dict[str, Any] | str, metadata: Optional[Dict[str, Any]] = {}) -> None:
        """
        Saves data to the LTM table with error handling.
        """
        data = data if isinstance(data, dict) else dict(data=data)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
            """
                INSERT INTO long_term_memories (datetime, data, metadata)
                VALUES (?, ?, ?)
            """,
                    (datetime.datetime.now().timestamp(), json.dumps(data), json.dumps(metadata)),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"MEMORY ERROR: An error occurred while saving to LTM: {str(e)}", color="red")


    def load(self, query: str, latest_n: int) -> Optional[List[Dict[str, Any]]]:
        """
        Queries the data row in the storage with error handling.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                f"""
                    SELECT datetime, data, metadata
                    FROM long_term_memories
                    WHERE data LIKE '%{query}%'
                    ORDER BY datetime
                    LIMIT {latest_n}
                """
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "datetime": row[0],
                            "data": json.loads(row[1]),
                            "metadata": json.loads(row[2]),
                        }
                        for row in rows
                    ]

        except sqlite3.Error as e:
            self._logger.log(level="error", message=f"MEMORY ERROR: An error occurred while querying LTM: {str(e)}", color="red")
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
            self._logger.log(level="error", message=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {str(e)}", color="red")
        return None
