import os
import logging
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class VectorDB:
    """Vector database interface for scalable similarity search."""

    def __init__(self):
        self.enabled = False
        self._client = None
        self._index = None

        # Try to initialize Pinecone if configured
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "assistant-embeddings")

            if pinecone_api_key:
                import pinecone
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

                # Check if index exists, create if not
                if pinecone_index not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=pinecone_index,
                        dimension=384,  # Standard for sentence-transformers
                        metric="cosine"
                    )

                self._index = pinecone.Index(pinecone_index)
                self.enabled = True
                logger.info(f"Pinecone vector database initialized: {pinecone_index}")
            else:
                logger.info("Pinecone not configured, using SQLite fallback")
        except ImportError:
            logger.warning("Pinecone not available, using SQLite fallback")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")

    def upsert_embedding(self, item_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Upsert an embedding into the vector database."""
        if not self.enabled or not self._index:
            return False

        try:
            self._index.upsert(vectors=[{
                "id": item_id,
                "values": vector,
                "metadata": metadata
            }])
            return True
        except Exception as e:
            logger.error(f"Failed to upsert embedding {item_id}: {e}")
            return False

    def query_similar(self, vector: List[float], top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict]:
        """Query for similar embeddings."""
        if not self.enabled or not self._index:
            return []

        try:
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )

            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {}
                })

            return results
        except Exception as e:
            logger.error(f"Failed to query similar vectors: {e}")
            return []

    def delete_embedding(self, item_id: str) -> bool:
        """Delete an embedding from the vector database."""
        if not self.enabled or not self._index:
            return False

        try:
            self._index.delete(ids=[item_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding {item_id}: {e}")
            return False

    def health_check(self) -> bool:
        """Check if vector database is healthy."""
        if not self.enabled:
            return False

        try:
            # Simple health check - try to get index stats
            stats = self._index.describe_index_stats()
            return True
        except Exception:
            return False

# Global instance
vector_db = VectorDB()