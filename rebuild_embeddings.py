#!/usr/bin/env python3
"""
Rebuild Embeddings Script - Chandresh's reindexing utility

This script rebuilds the embedding index for all summaries and tasks in the database.
Useful for:
- Initial setup
- Model changes
- Data corruption recovery
- Performance optimization
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from embedding_service import EmbeddingService
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_existing_embeddings(db_path: str, item_type: str = None):
    """Clear existing embeddings from database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if item_type:
            cursor.execute('DELETE FROM embeddings WHERE item_type = ?', (item_type,))
            logger.info(f"Cleared embeddings for item_type: {item_type}")
        else:
            cursor.execute('DELETE FROM embeddings')
            logger.info("Cleared all embeddings")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error clearing embeddings: {e}")
        raise

def get_items_to_index(db_path: str, item_type: str):
    """Get items that need to be indexed."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if item_type == 'summary':
            cursor.execute('SELECT summary_id, summary_text FROM summaries')
        elif item_type == 'task':
            cursor.execute('SELECT task_id, task_text FROM tasks')
        elif item_type == 'response':
            cursor.execute('SELECT response_id, response_text FROM responses')
        else:
            raise ValueError(f"Unsupported item_type: {item_type}")
        
        items = cursor.fetchall()
        conn.close()
        
        return items
        
    except Exception as e:
        logger.error(f"Error getting items to index: {e}")
        raise

def rebuild_embeddings(db_path: str = "assistant_demo.db", 
                      item_types: list = None, 
                      clear_first: bool = False,
                      model_name: str = "all-MiniLM-L6-v2"):
    """
    Rebuild embeddings for specified item types.
    
    Args:
        db_path: Path to SQLite database
        item_types: List of item types to rebuild (default: ['summary', 'task'])
        clear_first: Whether to clear existing embeddings first
        model_name: SentenceTransformer model to use
    """
    
    if item_types is None:
        item_types = ['summary', 'task']
    
    logger.info(f"Starting embedding rebuild for types: {item_types}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Model: {model_name}")
    
    # Initialize embedding service
    embedding_service = EmbeddingService(db_path=db_path, model_name=model_name)
    
    total_processed = 0
    total_errors = 0
    
    for item_type in item_types:
        logger.info(f"\n--- Processing {item_type} embeddings ---")
        
        try:
            # Clear existing embeddings if requested
            if clear_first:
                clear_existing_embeddings(db_path, item_type)
            
            # Get items to index
            items = get_items_to_index(db_path, item_type)
            logger.info(f"Found {len(items)} {item_type} items to process")
            
            # Process each item
            processed = 0
            errors = 0
            
            for item_id, item_text in items:
                try:
                    if embedding_service.store_embedding(item_type, item_id, item_text):
                        processed += 1
                        if processed % 10 == 0:
                            logger.info(f"Processed {processed}/{len(items)} {item_type} items")
                    else:
                        errors += 1
                        logger.warning(f"Failed to store embedding for {item_type} {item_id}")
                        
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {item_type} {item_id}: {e}")
            
            logger.info(f"Completed {item_type}: {processed} processed, {errors} errors")
            total_processed += processed
            total_errors += errors
            
        except Exception as e:
            logger.error(f"Error processing {item_type}: {e}")
            total_errors += 1
    
    logger.info(f"\n--- Rebuild Complete ---")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total errors: {total_errors}")
    
    return total_processed, total_errors

def verify_embeddings(db_path: str = "assistant_demo.db"):
    """Verify the integrity of stored embeddings."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check embedding counts
        cursor.execute('SELECT item_type, COUNT(*) FROM embeddings GROUP BY item_type')
        counts = cursor.fetchall()
        
        logger.info("Embedding counts by type:")
        for item_type, count in counts:
            logger.info(f"  {item_type}: {count}")
        
        # Check for invalid embeddings
        cursor.execute('SELECT COUNT(*) FROM embeddings WHERE vector_blob IS NULL OR vector_blob = ""')
        invalid_count = cursor.fetchone()[0]
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid embeddings")
        else:
            logger.info("All embeddings appear valid")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Rebuild embedding index for AI Assistant")
    
    parser.add_argument(
        "--db-path", 
        default="assistant_demo.db",
        help="Path to SQLite database (default: assistant_demo.db)"
    )
    
    parser.add_argument(
        "--types",
        nargs="+",
        default=["summary", "task"],
        choices=["summary", "task", "response"],
        help="Item types to rebuild (default: summary task)"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing embeddings before rebuilding"
    )
    
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing embeddings, don't rebuild"
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            logger.info("Verifying existing embeddings...")
            verify_embeddings(args.db_path)
        else:
            # Rebuild embeddings
            processed, errors = rebuild_embeddings(
                db_path=args.db_path,
                item_types=args.types,
                clear_first=args.clear,
                model_name=args.model
            )
            
            # Verify after rebuild
            logger.info("\nVerifying rebuilt embeddings...")
            verify_embeddings(args.db_path)
            
            # Exit with error code if there were errors
            if errors > 0:
                logger.warning(f"Completed with {errors} errors")
                sys.exit(1)
            else:
                logger.info("Rebuild completed successfully!")
                sys.exit(0)
                
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()