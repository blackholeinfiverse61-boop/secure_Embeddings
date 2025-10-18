import numpy as np
from embedding_service import embedding_service

def test_embedding_generation():
    """Test embedding generation to see if we're getting proper embeddings."""
    print("Testing embedding generation...")
    
    # Test with a sample text
    test_text = "This is a test sentence for embedding generation."
    print(f"Test text: {test_text}")
    
    # Generate embedding
    embedding = embedding_service.generate_embedding(test_text)
    
    # Check embedding properties
    print(f"Embedding length: {len(embedding)}")
    print(f"Embedding type: {type(embedding)}")
    
    # Count non-zero values
    non_zero_count = sum(1 for val in embedding if val != 0.0)
    print(f"Non-zero values: {non_zero_count}/{len(embedding)}")
    
    # Show first 10 values
    print(f"First 10 values: {embedding[:10]}")
    
    # Check if most values are zero
    if non_zero_count < len(embedding) * 0.1:
        print("WARNING: Most embedding values are zero!")
        print("This indicates a problem with embedding generation.")
    else:
        print("Embedding generation appears to be working correctly.")

if __name__ == "__main__":
    test_embedding_generation()