import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding_service import embedding_service

def test_enhanced_embeddings():
    """Test the enhanced embeddings with similarity search."""
    print("=== Testing Enhanced Embeddings ===")
    
    # Test similarity search with different queries
    test_queries = [
        "user authentication feature",
        "database performance optimization",
        "machine learning model accuracy",
        "cloud infrastructure cost reduction",
        "security vulnerability patching"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = embedding_service.search_similar_items(query_text=query, top_k=3)
            print(f"Found {len(results)} similar items:")
            for item in results:
                print(f"  {item['item_type']} {item['item_id']}: {item['score']:.4f}")
                print(f"    Text: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
        except Exception as e:
            print(f"Search failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_enhanced_embeddings()