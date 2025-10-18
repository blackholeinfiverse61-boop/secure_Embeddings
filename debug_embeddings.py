import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding_service import embedding_service
import json

def debug_embedding_generation():
    """Debug the embedding generation process."""
    print("Debugging embedding generation...")
    
    # Test texts
    test_texts = [
        "This is a test summary for embedding storage",
        "Another example text for testing embeddings",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand human language"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1}: '{text}' ---")
        
        # Generate embedding
        embedding = embedding_service.generate_embedding(text)
        
        # Show statistics
        non_zero_count = sum(1 for val in embedding if val != 0.0)
        zero_count = len(embedding) - non_zero_count
        zero_percentage = (zero_count / len(embedding)) * 100
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Non-zero values: {non_zero_count}")
        print(f"Zero values: {zero_count} ({zero_percentage:.1f}%)")
        
        # Show first 10 values
        print(f"First 10 values: {[round(val, 4) for val in embedding[:10]]}")
        
        # Check if all values are zero
        if non_zero_count == 0:
            print("WARNING: All values are zero!")
        elif non_zero_count < 10:
            print("WARNING: Very few non-zero values!")
        
        # Try to store the embedding
        item_type = "debug_test"
        item_id = f"test_{i+1:03d}"
        success = embedding_service.store_embedding(item_type, item_id, text)
        print(f"Storage result: {success}")

def check_stored_embeddings():
    """Check what's actually stored in the database."""
    print("\n\n=== Checking Stored Embeddings ===")
    
    import sqlite3
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT item_type, item_id, vector_blob FROM embeddings WHERE item_type = 'debug_test'")
    results = cursor.fetchall()
    
    print(f"Found {len(results)} debug test embeddings:")
    
    for item_type, item_id, vector_blob in results:
        embedding = json.loads(vector_blob)
        non_zero_count = sum(1 for val in embedding if val != 0.0)
        zero_percentage = (1 - non_zero_count / len(embedding)) * 100
        
        print(f"- {item_type} {item_id}: {non_zero_count}/{len(embedding)} non-zero values ({zero_percentage:.1f}% zero)")
        
        if non_zero_count > 0:
            print(f"  First 5 values: {[round(val, 4) for val in embedding[:5]]}")
        else:
            print("  WARNING: All values are zero in database!")
    
    conn.close()

if __name__ == "__main__":
    debug_embedding_generation()
    check_stored_embeddings()