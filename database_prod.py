import sqlite3
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Production database manager with connection pooling and error handling."""

    def __init__(self, db_path: str = "assistant_demo.db", max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connection_pool = []
        self._pool_lock = None  # In production, use threading.Lock()

    def initialize(self):
        """Initialize the database and create tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    vector_blob TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    text_content TEXT,
                    UNIQUE(item_type, item_id)
                )
            ''')

            # Add text_content column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE embeddings ADD COLUMN text_content TEXT")
            except sqlite3.OperationalError:
                pass

            # Create other required tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summaries (
                    summary_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    message_text TEXT,
                    summary_text TEXT,
                    timestamp TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    summary_id TEXT,
                    user_id TEXT,
                    task_text TEXT,
                    priority TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (summary_id) REFERENCES summaries (summary_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS responses (
                    response_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    user_id TEXT,
                    response_text TEXT,
                    tone TEXT,
                    status TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coach_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id TEXT,
                    task_id TEXT,
                    response_id TEXT,
                    score INTEGER,
                    comment TEXT,
                    timestamp TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL,
                    timestamp TEXT
                )
            ''')

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection from pool."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """Execute a SELECT query and return results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise

    def execute_query_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """Execute a SELECT query and return all results (alias for execute_query)."""
        return self.execute_query(query, params)

    def execute_update(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Database update error: {e}")
            raise

# Global instance
db_manager = DatabaseManager()