#!/usr/bin/env python3
"""
Script to inspect the embeddings in the database.
"""

import sqlite3
import json

def inspect_database():
    """Inspect the contents of the embeddings table."""
    print("Inspecting embeddings database...")
    
    conn = sqlite3.connect("assistant_demo.db")
    cursor = conn.cursor()
    
    # Count total embeddings
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_count = cursor.fetchone()[0]
    print(f"Total embeddings: {total_count}")
    
    # Show embeddings by type
    cursor.execute("SELECT item_type, COUNT(*) FROM embeddings GROUP BY item_type")
    type_counts = cursor.fetchall()
    print("Embeddings by type:")
    for item_type, count in type_counts:
        print(f"  {item_type}: {count}")
    
    # Show recent embeddings
    cursor.execute("SELECT item_type, item_id, timestamp FROM embeddings ORDER BY timestamp DESC LIMIT 10")
    recent_embeddings = cursor.fetchall()
    print("\nRecent embeddings:")
    for item_type, item_id, timestamp in recent_embeddings:
        print(f"  {item_type} {item_id} ({timestamp})")
    
    # Show our test embeddings
    cursor.execute("SELECT item_type, item_id FROM embeddings WHERE item_id LIKE 'comptest_%'")
    test_embeddings = cursor.fetchall()
    print("\nTest embeddings:")
    for item_type, item_id in test_embeddings:
        print(f"  {item_type} {item_id}")
    
    conn.close()

if __name__ == "__main__":
    inspect_database()