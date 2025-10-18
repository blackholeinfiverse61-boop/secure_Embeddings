import sqlite3
import json
from datetime import datetime
from typing import Optional

def init_database(db_path: str = "assistant_demo.db"):
    """Initialize the SQLite database with required tables for Chandresh's work."""
    conn = sqlite3.connect(db_path)
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
    
    # Add text_content column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE embeddings ADD COLUMN text_content TEXT")
    except sqlite3.OperationalError:
        # Column already exists, which is fine
        pass
    
    # Create summaries table (needed for integration)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            user_id TEXT,
            message_text TEXT,
            summary_text TEXT,
            timestamp TEXT
        )
    ''')
    
    # Create tasks table (needed for integration)
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
    
    # Create responses table (for integration with Noopur's work)
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
    
    # Create coach_feedback table (for integration with Parth's work)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coach_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary_id TEXT,
            task_id TEXT,
            response_id TEXT,
            score INTEGER,
            comment TEXT,
            timestamp TEXT,
            FOREIGN KEY (summary_id) REFERENCES summaries (summary_id),
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (response_id) REFERENCES responses (response_id)
        )
    ''')
    
    # Create metrics table (for integration with Nilesh's work)
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
    conn.close()
    print(f"Database initialized successfully at {db_path}")

if __name__ == "__main__":
    init_database()