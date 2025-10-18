#!/usr/bin/env python3
"""
Embedding Quality Check Utility

This script checks the quality of embeddings in the database to ensure they are properly generated.
"""

import sqlite3
import json
import numpy as np

def check_embedding_quality(db_path: str = "assistant_demo.db"):
    """Check the quality of embeddings in the database."""
    print("Checking embedding quality...")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all embeddings
    cursor.execute("SELECT item_type, item_id, vector_blob FROM embeddings")
    results = cursor.fetchall()
    
    if not results:
        print("No embeddings found in database.")
        return
    
    print(f"Found {len(results)} embeddings to check:")
    
    total_embeddings = len(results)
    good_embeddings = 0
    zero_embeddings = 0
    sparse_embeddings = 0
    
    for item_type, item_id, vector_blob in results:
        try:
            # Parse the embedding
            embedding = json.loads(vector_blob)
            
            # Count non-zero values
            non_zero_count = sum(1 for val in embedding if val != 0.0)
            zero_percentage = (1 - non_zero_count / len(embedding)) * 100
            
            # Classify embedding quality
            if non_zero_count == len(embedding):
                good_embeddings += 1
                quality = "GOOD"
            elif zero_percentage > 90:
                zero_embeddings += 1
                quality = "MOSTLY_ZERO"
            else:
                sparse_embeddings += 1
                quality = "SPARSE"
            
            print(f"- {item_type} {item_id}: {quality} ({zero_percentage:.1f}% zero)")
            
        except Exception as e:
            print(f"- {item_type} {item_id}: ERROR - {e}")
    
    conn.close()
    
    print(f"\n--- Quality Summary ---")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Good quality (all non-zero): {good_embeddings} ({good_embeddings/total_embeddings*100:.1f}%)")
    print(f"Sparse embeddings: {sparse_embeddings} ({sparse_embeddings/total_embeddings*100:.1f}%)")
    print(f"Mostly zero embeddings: {zero_embeddings} ({zero_embeddings/total_embeddings*100:.1f}%)")
    
    if zero_embeddings > 0:
        print("\nWARNING: Some embeddings have mostly zero values!")
        print("This may indicate issues with the embedding generation process.")
    else:
        print("\nAll embeddings appear to be of good quality!")

if __name__ == "__main__":
    check_embedding_quality()