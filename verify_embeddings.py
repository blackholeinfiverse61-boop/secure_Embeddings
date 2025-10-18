import json
import sqlite3
from embedding_service import embedding_service

def verify_embeddings():
    """Verify that embeddings are being stored properly in the database."""
    print("Verifying embedding storage...")
    
    # Test storing an embedding
    test_text = "This is a test sentence for embedding verification."
    print(f"Test text: {test_text}")
    
    # Store embedding
    success = embedding_service.store_embedding("test", "test_001", test_text)
    print(f"Embedding storage success: {success}")
    
    # Check the database directly
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    # Get the stored embedding
    cursor.execute("SELECT vector_blob, text_content FROM embeddings WHERE item_type='test' AND item_id='test_001'")
    result = cursor.fetchone()
    
    if result:
        vector_blob, stored_text = result
        print(f"Stored text: {stored_text}")
        
        # Parse the embedding
        embedding = json.loads(vector_blob)
        print(f"Embedding length: {len(embedding)}")
        print(f"Embedding type: {type(embedding)}")
        
        # Count non-zero values
        non_zero_count = sum(1 for val in embedding if val != 0.0)
        print(f"Non-zero values: {non_zero_count}/{len(embedding)}")
        
        # Show first 10 values
        print(f"First 10 values: {[round(val, 4) for val in embedding[:10]]}")
        
        if non_zero_count > len(embedding) * 0.1:
            print("SUCCESS: Embeddings are being stored with sufficient non-zero values!")
        else:
            print("WARNING: Most embedding values are still zero.")
    else:
        print("ERROR: Embedding was not stored in database.")
    
    conn.close()

if __name__ == "__main__":
    verify_embeddings()