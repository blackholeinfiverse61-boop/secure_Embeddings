import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time
import asyncio

# Import production components
from database_prod import db_manager
from cache import cache_manager, cached
from vector_db import vector_db
from config import config
from circuit_breaker import CircuitBreakerOpenException

# Try to import sentence_transformers, with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using fallback embedding method.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for handling embeddings and similarity search - Chandresh's core work."""
    
    def __init__(self, db_path: str = "assistant_demo.db", model_name: Optional[str] = None):
        self.db_path = db_path
        # Use config model name if not provided
        self.model_name = model_name or config.MODEL_NAME
        self._model = None
        # Initialize production components
        db_manager.initialize()
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
            
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            # Use cache directory from config
            self._model = SentenceTransformer(  # type: ignore
                self.model_name, 
                cache_folder=config.MODEL_CACHE_DIR
            )
        return self._model
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except CircuitBreakerOpenException:
                # Don't retry if circuit breaker is open
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")
    
    @cached(ttl=3600)  # Cache for 1 hour
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text."""
        def _generate():
            try:
                # Use sentence-transformers if available
                if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
                    # Type ignore for the possibly unbound variable warning
                    embedding = self.model.encode([text])[0]  # type: ignore
                    return embedding.tolist()
                else:
                    # Fallback to improved hash-based embedding
                    logger.warning("Using fallback embedding method due to missing sentence-transformers")
                    vec = [0.0] * 384  # Standard size for sentence transformers
                    
                    # Split text into words and process each word
                    words = text.lower().split()
                    for i, word in enumerate(words[:100]):  # Limit to first 100 words
                        # Use multiple hash functions to distribute values better
                        h1 = hash(word) % 384
                        h2 = hash(word + "salt1") % 384
                        h3 = hash(word + "salt2") % 384
                        
                        # Add values to multiple positions
                        vec[h1] += 1.0
                        vec[h2] += 0.5
                        vec[h3] += 0.25
                    
                    # Also add positional information
                    for i, word in enumerate(words[:50]):  # First 50 words get positional weight
                        pos_hash = hash(f"pos_{i}_{word}") % 384
                        vec[pos_hash] += 0.1 * (1.0 - i/50.0)  # Decreasing weight with position
                    
                    # Add character-level n-grams for better representation
                    text_lower = text.lower()
                    for i in range(len(text_lower) - 2):
                        if i < 200:  # Limit character n-grams
                            trigram = text_lower[i:i+3]
                            char_hash = hash(trigram) % 384
                            vec[char_hash] += 0.05
                    
                    # L2 normalize
                    norm = sum(v * v for v in vec) ** 0.5
                    if norm > 0:
                        vec = [v / norm for v in vec]
                    else:
                        # If all values are zero, create a small random vector
                        vec = np.random.random(384).tolist()
                        # Normalize the random vector
                        norm = sum(v * v for v in vec) ** 0.5
                        vec = [v / norm for v in vec]
                    return vec
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback to random vector for testing
                return np.random.random(384).tolist()
        
        return self._retry_with_backoff(_generate)
    
    def store_embedding(self, item_type: str, item_id: str, text: str) -> bool:
        """Store embedding for an item in the database."""
        def _store():
            try:
                # Generate embedding
                embedding = self.generate_embedding(text)
                vector_blob = json.dumps(embedding)
                
                # Store in relational database using named parameters
                params = {
                    "item_type": item_type,
                    "item_id": item_id,
                    "vector_blob": vector_blob,
                    "timestamp": datetime.now().isoformat(),
                    "text_content": text
                }
                
                db_manager.execute_update(
                    '''
                    INSERT OR REPLACE INTO embeddings 
                    (item_type, item_id, vector_blob, timestamp, text_content)
                    VALUES (:item_type, :item_id, :vector_blob, :timestamp, :text_content)
                    ''', 
                    params
                )
                
                # Also store in vector database for scalable search
                if vector_db.enabled:
                    metadata = {
                        "item_type": item_type,
                        "item_id": item_id,
                        "text_content": text[:1000],  # Limit metadata size
                        "timestamp": datetime.now().isoformat()
                    }
                    vector_db.upsert_embedding(f"{item_type}_{item_id}", embedding, metadata)
                
                logger.info(f"Stored embedding for {item_type} {item_id}")
                return True
                
            except CircuitBreakerOpenException:
                logger.error("Circuit breaker is open, cannot store embedding")
                raise
            except Exception as e:
                logger.error(f"Error storing embedding: {e}")
                raise
        
        try:
            return self._retry_with_backoff(_store)
        except CircuitBreakerOpenException:
            return False
        except Exception:
            return False
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def search_similar_items(self, query_text: Optional[str] = None, summary_id: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar items based on query text or summary_id."""
        def _search():
            try:
                # Determine query embedding
                if query_text:
                    query_embedding = self.generate_embedding(query_text)
                elif summary_id:
                    # Get the summary text for the given summary_id
                    results = db_manager.execute_query_all(
                        'SELECT text_content FROM embeddings WHERE item_id = :item_id AND item_type = :item_type', 
                        {"item_id": summary_id, "item_type": "summary"}
                    )
                    
                    if not results or not results[0] or not results[0][0]:
                        return []
                    
                    query_embedding = self.generate_embedding(results[0][0])
                else:
                    return []
                
                # Use vector database for scalable search if available
                if vector_db.enabled:
                    filter_dict = None
                    if summary_id:
                        # Exclude the query item itself
                        filter_dict = {"item_id": {"$ne": summary_id}}
                    
                    vector_results = vector_db.query_similar(query_embedding, top_k, filter_dict)
                    
                    # Format results
                    similarities = []
                    for result in vector_results:
                        similarities.append({
                            'item_type': result['metadata'].get('item_type', 'unknown'),
                            'item_id': result['metadata'].get('item_id', 'unknown'),
                            'score': result['score'],
                            'text': result['metadata'].get('text_content', '')[:200] + '...' if len(result['metadata'].get('text_content', '')) > 200 else result['metadata'].get('text_content', '')
                        })
                    return similarities
                
                # Fallback to database search
                # Get all embeddings from database
                results = db_manager.execute_query_all('''
                    SELECT item_type, item_id, vector_blob, text_content
                    FROM embeddings
                ''')
                
                # Calculate similarities
                similarities = []
                for item_type, item_id, vector_blob, text_content in results:
                    try:
                        # Skip items with no text
                        if not text_content:
                            continue
                        
                        # Skip the query item itself if searching by summary_id
                        if summary_id and item_id == summary_id and item_type == 'summary':
                            continue
                        
                        item_embedding = json.loads(vector_blob)
                        similarity = self.cosine_similarity(query_embedding, item_embedding)
                        
                        similarities.append({
                            'item_type': item_type,
                            'item_id': item_id,
                            'score': similarity,
                            'text': text_content[:200] + '...' if len(text_content) > 200 else text_content  # Truncate for display
                        })
                    except Exception as e:
                        logger.error(f"Error processing item {item_id}: {e}")
                        continue
                
                # Sort by similarity score and return top_k
                similarities.sort(key=lambda x: x['score'], reverse=True)
                return similarities[:top_k]
                
            except CircuitBreakerOpenException:
                logger.error("Circuit breaker is open, cannot search similar items")
                raise
            except Exception as e:
                logger.error(f"Error searching similar items: {e}")
                raise
        
        try:
            return self._retry_with_backoff(_search)
        except CircuitBreakerOpenException:
            return []
        except Exception:
            return []
    
    def index_existing_summaries(self) -> int:
        """Index all existing summaries that don't have embeddings yet."""
        def _index_summaries():
            try:
                # Find summaries without embeddings
                results = db_manager.execute_query_all('''
                    SELECT s.summary_id, s.summary_text 
                    FROM summaries s
                    LEFT JOIN embeddings e ON s.summary_id = e.item_id AND e.item_type = 'summary'
                    WHERE e.id IS NULL
                ''')
                
                unindexed_summaries = results
                indexed_count = 0
                for summary_id, summary_text in unindexed_summaries:
                    if self.store_embedding('summary', summary_id, summary_text):
                        indexed_count += 1
                
                logger.info(f"Indexed {indexed_count} summaries")
                return indexed_count
                
            except CircuitBreakerOpenException:
                logger.error("Circuit breaker is open, cannot index summaries")
                raise
            except Exception as e:
                logger.error(f"Error indexing existing summaries: {e}")
                raise
        
        try:
            return self._retry_with_backoff(_index_summaries)
        except CircuitBreakerOpenException:
            return 0
        except Exception:
            return 0
    
    def index_existing_tasks(self) -> int:
        """Index all existing tasks that don't have embeddings yet."""
        def _index_tasks():
            try:
                # Find tasks without embeddings
                results = db_manager.execute_query_all('''
                    SELECT t.task_id, t.task_text 
                    FROM tasks t
                    LEFT JOIN embeddings e ON t.task_id = e.item_id AND e.item_type = 'task'
                    WHERE e.id IS NULL
                ''')
                
                unindexed_tasks = results
                indexed_count = 0
                for task_id, task_text in unindexed_tasks:
                    if self.store_embedding('task', task_id, task_text):
                        indexed_count += 1
                
                logger.info(f"Indexed {indexed_count} tasks")
                return indexed_count
                
            except CircuitBreakerOpenException:
                logger.error("Circuit breaker is open, cannot index tasks")
                raise
            except Exception as e:
                logger.error(f"Error indexing existing tasks: {e}")
                raise
        
        try:
            return self._retry_with_backoff(_index_tasks)
        except CircuitBreakerOpenException:
            return 0
        except Exception:
            return 0

# Global instance for use in API
embedding_service = EmbeddingService()