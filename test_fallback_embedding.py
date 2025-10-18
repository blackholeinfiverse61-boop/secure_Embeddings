#!/usr/bin/env python3
"""
Test script to verify fallback embedding functionality
"""

def test_fallback_embedding():
    """Test the fallback embedding method"""
    try:
        # Import the embedding service
        from embedding_service import embedding_service
        import numpy as np
        
        # Test text
        test_text = "This is a test sentence for embedding generation"
        
        # Generate embedding using fallback method
        embedding = embedding_service.generate_embedding(test_text)
        
        # Check if embedding is valid
        if isinstance(embedding, list) and len(embedding) == 384:
            print("✓ Fallback embedding generation successful")
            print(f"  Embedding length: {len(embedding)}")
            print(f"  Embedding type: {type(embedding)}")
            
            # Check if it's normalized
            norm = np.linalg.norm(embedding)
            print(f"  Embedding norm: {norm:.4f}")
            
            # Test similarity calculation
            similarity = embedding_service.cosine_similarity(embedding, embedding)
            print(f"  Self-similarity: {similarity:.4f}")
            
            if abs(similarity - 1.0) < 0.001:
                print("✓ Cosine similarity calculation working")
                return True
            else:
                print("✗ Cosine similarity calculation failed")
                return False
        else:
            print("✗ Fallback embedding generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_embedding_storage():
    """Test storing embeddings in database"""
    try:
        from embedding_service import embedding_service
        
        # Test storing an embedding
        result = embedding_service.store_embedding("test", "test_001", "This is a test embedding")
        
        if result:
            print("✓ Embedding storage successful")
            return True
        else:
            print("✗ Embedding storage failed")
            return False
            
    except Exception as e:
        print(f"✗ Embedding storage test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fallback Embedding Functionality")
    print("=" * 50)
    
    embedding_ok = test_fallback_embedding()
    print()
    
    storage_ok = test_embedding_storage()
    print()
    
    if embedding_ok and storage_ok:
        print("✓ All fallback embedding tests passed!")
        print("The application can run even without sentence-transformers.")
    else:
        print("✗ Some tests failed. Please check the errors above.")