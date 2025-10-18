import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding_service import embedding_service
import json
import sqlite3

def comprehensive_embedding_test():
    """Comprehensive test of embedding generation, storage, and retrieval."""
    print("=== Comprehensive Embedding Test ===")
    
    # Extended test data with more diverse samples
    test_cases = [
        # Original test cases
        ("summary", "summary_001", "This is the first test summary for our application"),
        ("task", "task_001", "Create a new feature for user authentication"),
        ("response", "response_001", "I have completed the user authentication feature"),
        ("summary", "summary_002", "Machine learning models require quality training data"),
        ("task", "task_002", "Implement data preprocessing pipeline for ML models"),
        
        # Additional diverse samples for better training
        ("summary", "summary_003", "The system performance has improved significantly after the recent optimization"),
        ("summary", "summary_004", "Customer feedback indicates high satisfaction with the new interface design"),
        ("summary", "summary_005", "Database queries are taking longer than expected during peak hours"),
        ("summary", "summary_006", "The new API endpoints are functioning correctly with all test cases passing"),
        ("summary", "summary_007", "Security audit revealed several vulnerabilities that need immediate attention"),
        ("summary", "summary_008", "The deployment process completed successfully with zero downtime"),
        ("summary", "summary_009", "User engagement metrics show a 25% increase after the feature release"),
        ("summary", "summary_010", "Integration with third-party services is experiencing intermittent connection issues"),
        
        ("task", "task_003", "Optimize database queries for better performance during peak usage"),
        ("task", "task_004", "Design and implement user feedback collection mechanism"),
        ("task", "task_005", "Fix security vulnerabilities identified in the recent audit report"),
        ("task", "task_006", "Create automated deployment scripts for zero-downtime updates"),
        ("task", "task_007", "Analyze user engagement data to identify improvement opportunities"),
        ("task", "task_008", "Implement retry mechanism for third-party service connections"),
        ("task", "task_009", "Develop comprehensive API documentation for all endpoints"),
        ("task", "task_010", "Set up monitoring and alerting for system performance metrics"),
        
        ("response", "response_002", "Performance optimization completed, response times reduced by 40%"),
        ("response", "response_003", "Feedback collection form has been implemented and deployed"),
        ("response", "response_004", "All critical security vulnerabilities have been patched and tested"),
        ("response", "response_005", "Deployment scripts are ready for production use"),
        ("response", "response_006", "User engagement analysis report has been generated and shared"),
        ("response", "response_007", "Connection retry mechanism is now handling failures gracefully"),
        ("response", "response_008", "API documentation is available at the standard documentation endpoint"),
        ("response", "response_009", "Monitoring system is active and sending alerts as configured"),
        ("response", "response_010", "The issue has been resolved and system is back to normal operation"),
        
        # Additional domain-specific samples
        ("summary", "summary_011", "Natural language processing model achieved 92% accuracy on validation set"),
        ("task", "task_011", "Fine-tune the NLP model with additional domain-specific training data"),
        ("response", "response_011", "Model fine-tuning completed with improved performance on specialized tasks"),
        
        ("summary", "summary_012", "Cloud infrastructure costs increased by 15% this quarter"),
        ("task", "task_012", "Optimize cloud resource allocation to reduce infrastructure costs"),
        ("response", "response_012", "Resource optimization implemented, projected cost reduction of 20%"),
    ]
    
    # Store embeddings
    print("\n1. Storing embeddings...")
    for item_type, item_id, text in test_cases:
        success = embedding_service.store_embedding(item_type, item_id, text)
        print(f"  {item_type} {item_id}: {'✓' if success else '✗'}")
    
    # Retrieve and verify embeddings
    print("\n2. Verifying stored embeddings...")
    conn = sqlite3.connect('assistant_demo.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT item_type, item_id, vector_blob, text_content FROM embeddings")
    results = cursor.fetchall()
    
    print(f"Found {len(results)} total embeddings in database:")
    
    for item_type, item_id, vector_blob, text_content in results:
        # Parse the embedding
        embedding = json.loads(vector_blob)
        
        # Count non-zero values
        non_zero_count = sum(1 for val in embedding if val != 0.0)
        zero_percentage = (1 - non_zero_count / len(embedding)) * 100
        
        print(f"  {item_type} {item_id}: {non_zero_count}/{len(embedding)} non-zero ({zero_percentage:.1f}% zero)")
        
        # Verify text content matches
        if text_content and len(text_content) > 50:
            text_display = text_content[:47] + "..."
        else:
            text_display = text_content or "None"
            
        print(f"    Text: {text_display}")
        
        # Check if embedding is valid
        if non_zero_count > 0:
            print(f"    Status: ✓ Valid embedding")
        else:
            print(f"    Status: ✗ Invalid embedding (all zeros)")
    
    conn.close()
    
    # Test similarity search
    print("\n3. Testing similarity search...")
    try:
        results = embedding_service.search_similar_items(
            query_text="user authentication feature", 
            top_k=3
        )
        
        print(f"Found {len(results)} similar items:")
        for item in results:
            print(f"  {item['item_type']} {item['item_id']}: {item['score']:.4f}")
    except Exception as e:
        print(f"Similarity search failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    comprehensive_embedding_test()