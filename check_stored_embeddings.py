import json
import sqlite3

def check_stored_embeddings():
    """Check what embeddings are stored in the database."""
    print("Checking stored embeddings in database...")
    
    # Connect to the database
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    # Get all embeddings
    cursor.execute("SELECT item_type, item_id, vector_blob FROM embeddings")
    results = cursor.fetchall()
    
    print(f"Found {len(results)} embeddings in database:")
    
    for item_type, item_id, vector_blob in results:
        # Parse the embedding
        embedding = json.loads(vector_blob)
        
        # Count non-zero values
        non_zero_count = sum(1 for val in embedding if val != 0.0)
        zero_percentage = (1 - non_zero_count / len(embedding)) * 100
        
        print(f"- {item_type} {item_id}: {non_zero_count}/{len(embedding)} non-zero values ({zero_percentage:.1f}% zero)")
        
        # Show first 5 non-zero values instead of first 5 values
        non_zero_values = [val for val in embedding if val != 0.0]
        if non_zero_values:
            display_values = non_zero_values[:5]  # First 5 non-zero values
            print(f"  First 5 non-zero values: {[round(val, 4) for val in display_values]}")
        else:
            print("  All values are zero")
    
    conn.close()

if __name__ == "__main__":
    check_stored_embeddings()