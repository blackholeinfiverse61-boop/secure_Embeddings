#!/usr/bin/env python3
"""
Test script to verify embedding storage and retrieval functionality.
"""

import sys
import os
import requests
import json

# Add the chandresh directory to the path so we can import the service
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding_service import embedding_service

def test_embedding_storage_directly():
    """Test embedding storage directly using the service."""
    print("Testing embedding storage directly...")
    
    # Test data
    item_type = "summary"
    item_id = "test123"
    text = "This is a test summary for embedding storage"
    
    # Store embedding
    success = embedding_service.store_embedding(item_type, item_id, text)
    print(f"Embedding storage result: {success}")
    
    # Verify storage by searching
    results = embedding_service.search_similar_items(query_text=text, top_k=3)
    print(f"Search results: {results}")
    
    return success

def test_embedding_storage_via_api():
    """Test embedding storage via the API endpoint."""
    print("\nTesting embedding storage via API...")
    
    try:
        # API endpoint
        url = "http://127.0.0.1:8000/api/store_embedding"
        
        # Test data
        params = {
            "item_type": "summary",
            "item_id": "api_test123",
            "text": "This is a test summary for API embedding storage"
        }
        
        # Make request
        response = requests.post(url, params=params, timeout=10)
        print(f"API response status: {response.status_code}")
        print(f"API response body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Parsed JSON: {data}")
            return data.get("status") == "success"
        else:
            print(f"API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing API: {e}")
        return False

def test_embedding_search_via_api():
    """Test embedding search via the API endpoint."""
    print("\nTesting embedding search via API...")
    
    try:
        # API endpoint
        url = "http://127.0.0.1:8000/api/search_similar"
        
        # Test data
        payload = {
            "message_text": "test summary for embedding",
            "top_k": 3
        }
        
        # Make request
        response = requests.post(url, json=payload, timeout=10)
        print(f"Search API response status: {response.status_code}")
        print(f"Search API response body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Search results: {data}")
            return True
        else:
            print(f"Search API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing search API: {e}")
        return False

if __name__ == "__main__":
    print("Starting embedding tests...")
    
    # Test direct storage
    direct_success = test_embedding_storage_directly()
    
    # Test API storage (only if API is running)
    api_success = False
    try:
        api_success = test_embedding_storage_via_api()
    except Exception as e:
        print(f"API test skipped (server likely not running): {e}")
    
    # Test API search (only if API is running)
    search_success = False
    try:
        search_success = test_embedding_search_via_api()
    except Exception as e:
        print(f"Search API test skipped (server likely not running): {e}")
    
    print("\nTest Summary:")
    print(f"Direct storage: {'PASS' if direct_success else 'FAIL'}")
    print(f"API storage: {'PASS' if api_success else 'SKIP/FAIL'}")
    print(f"API search: {'PASS' if search_success else 'SKIP/FAIL'}")