import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import logging

from .utils import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for creating text embeddings"""
    
    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        try:
            if not texts:
                return np.array([])
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            # Ensure embeddings are 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Create embedding for a single text"""
        return self.embed_texts([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model is None:
            self._load_model()
        
        # Create a test embedding to get dimension
        test_embedding = self.model.encode(["test"], normalize_embeddings=True)
        return test_embedding.shape[1] 