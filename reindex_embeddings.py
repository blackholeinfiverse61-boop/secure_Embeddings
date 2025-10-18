import sqlite3
import json
from embedding_service import embedding_service

def reindex_embeddings():
    """Reindex all embeddings in the database using the proper sentence-transformers method."""
    print("Reindexing all embeddings...")
    
    # Connect to the database
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    # Get all embeddings with their text content
    cursor.execute("SELECT item_type, item_id, text_content FROM embeddings WHERE text_content IS NOT NULL")
    results = cursor.fetchall()
    
    print(f"Found {len(results)} embeddings to reindex:")
    
    reindexed_count = 0
    for item_type, item_id, text_content in results:
        if text_content:  # Only reindex if we have text content
            try:
                # Store embedding again (this will use the proper sentence-transformers method now)
                success = embedding_service.store_embedding(item_type, item_id, text_content)
                if success:
                    reindexed_count += 1
                    print(f"- Reindexed {item_type} {item_id}")
                else:
                    print(f"- Failed to reindex {item_type} {item_id}")
            except Exception as e:
                print(f"- Error reindexing {item_type} {item_id}: {e}")
    
    conn.close()
    
    print(f"\nReindexing complete. Successfully reindexed {reindexed_count}/{len(results)} embeddings.")

if __name__ == "__main__":
    reindex_embeddings()