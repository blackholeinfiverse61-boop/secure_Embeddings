import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding_service import embedding_service
import sqlite3

def debug_embedding_storage_process():
    """Debug the entire embedding storage and retrieval process."""
    print("Debugging embedding storage process...")
    
    # Test text
    text = "This is a comprehensive test to check embedding storage"
    print(f"Test text: '{text}'")
    
    # Generate embedding
    print("\n1. Generating embedding...")
    embedding = embedding_service.generate_embedding(text)
    
    # Show statistics
    non_zero_count = sum(1 for val in embedding if val != 0.0)
    print(f"Generated embedding has {non_zero_count} non-zero values out of {len(embedding)}")
    
    # Show some non-zero values
    non_zero_values = [(i, val) for i, val in enumerate(embedding) if val != 0.0]
    print(f"First 5 non-zero values with indices: {[(i, round(val, 4)) for i, val in non_zero_values[:5]]}")
    
    # Serialize to JSON
    print("\n2. Serializing to JSON...")
    vector_blob = json.dumps(embedding)
    print(f"JSON string length: {len(vector_blob)}")
    
    # Show first part of JSON
    print(f"First 100 chars of JSON: {vector_blob[:100]}...")
    
    # Deserialize from JSON
    print("\n3. Deserializing from JSON...")
    embedding_restored = json.loads(vector_blob)
    restored_non_zero_count = sum(1 for val in embedding_restored if val != 0.0)
    print(f"Restored embedding has {restored_non_zero_count} non-zero values out of {len(embedding_restored)}")
    
    # Compare first few values
    print(f"Original first 5: {[round(val, 6) for val in embedding[:5]]}")
    print(f"Restored first 5: {[round(val, 6) for val in embedding_restored[:5]]}")
    
    # Check if they're the same
    if embedding == embedding_restored:
        print("✓ Serialization/deserialization is consistent")
    else:
        print("✗ Serialization/deserialization inconsistency detected!")
        
        # Check differences
        differences = [(i, orig, rest) for i, (orig, rest) in enumerate(zip(embedding, embedding_restored)) if abs(orig - rest) > 1e-10]
        if differences:
            print(f"First 5 differences: {[(i, round(orig, 6), round(rest, 6)) for i, orig, rest in differences[:5]]}")
    
    # Store in database
    print("\n4. Storing in database...")
    item_type = "debug_storage"
    item_id = "storage_test_001"
    
    # Direct database operation to see what's actually stored
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    params = {
        "item_type": item_type,
        "item_id": item_id,
        "vector_blob": vector_blob,
        "timestamp": "2025-01-01T00:00:00",
        "text_content": text
    }
    
    cursor.execute('''
        INSERT OR REPLACE INTO embeddings 
        (item_type, item_id, vector_blob, timestamp, text_content)
        VALUES (:item_type, :item_id, :vector_blob, :timestamp, :text_content)
        ''', params)
    
    conn.commit()
    
    # Retrieve from database
    print("\n5. Retrieving from database...")
    cursor.execute("SELECT vector_blob FROM embeddings WHERE item_type = ? AND item_id = ?", 
                   (item_type, item_id))
    result = cursor.fetchone()
    
    if result:
        db_vector_blob = result[0]
        print(f"Retrieved JSON string length: {len(db_vector_blob)}")
        print(f"Retrieved first 100 chars: {db_vector_blob[:100]}...")
        
        # Deserialize
        db_embedding = json.loads(db_vector_blob)
        db_non_zero_count = sum(1 for val in db_embedding if val != 0.0)
        print(f"Database embedding has {db_non_zero_count} non-zero values out of {len(db_embedding)}")
        
        # Compare with original
        if embedding == db_embedding:
            print("✓ Database storage/retrieval is consistent")
        else:
            print("✗ Database storage/retrieval inconsistency detected!")
            db_differences = [(i, orig, db_val) for i, (orig, db_val) in enumerate(zip(embedding, db_embedding)) if abs(orig - db_val) > 1e-10]
            if db_differences:
                print(f"First 5 database differences: {[(i, round(orig, 6), round(db_val, 6)) for i, orig, db_val in db_differences[:5]]}")
    else:
        print("✗ Failed to retrieve from database")
    
    conn.close()
    
    # Clean up test data
    print("\n6. Cleaning up...")
    embedding_service.store_embedding(item_type, item_id, "CLEANUP")

if __name__ == "__main__":
    debug_embedding_storage_process()