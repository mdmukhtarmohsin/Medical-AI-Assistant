"""
Vector store service for document embeddings and similarity search.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config.settings import settings


class VectorStore:
    """Service for managing document embeddings and similarity search."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_store_dir = settings.VECTOR_STORE_DIR
        self.index_file = os.path.join(self.vector_store_dir, "faiss_index.pkl")
        self.metadata_file = os.path.join(self.vector_store_dir, "metadata.json")
        
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # Initialize or load index
        self.index = None
        self.metadata = []
        self.dimension = None
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                with open(self.index_file, 'rb') as f:
                    self.index = pickle.load(f)
                
                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Get dimension from index
                if self.index is not None:
                    self.dimension = self.index.d
                    print(f"Loaded vector store with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self._initialize_empty_index()
    
    def _initialize_empty_index(self):
        """Initialize an empty FAISS index."""
        # Get embedding dimension by encoding a test string
        test_embedding = self.embedding_model.encode(["test"])
        self.dimension = test_embedding.shape[1]
        
        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        print(f"Initialized empty vector store with dimension {self.dimension}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Saved vector store with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def add_document_chunks(self, document_id: str, filename: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            chunks: List of text chunks with metadata
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Ensure index is initialized
        if self.index is None:
            self._initialize_empty_index()
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Add metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "char_count": chunk["char_count"],
                "vector_id": len(self.metadata)  # Current position in index
            }
            self.metadata.append(chunk_metadata)
        
        # Save updated index
        self._save_index()
        
        return len(chunks)
    
    def search_similar_chunks(self, query: str, k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            document_id: Optional document ID to filter results
            
        Returns:
            List of similar chunks with relevance scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search for similar vectors
        # Search more than k to allow for filtering
        search_k = min(k * 3, self.index.ntotal) if document_id else k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                chunk_metadata = self.metadata[idx].copy()
                
                # Filter by document_id if specified
                if document_id and chunk_metadata["document_id"] != document_id:
                    continue
                
                # Convert distance to similarity score (higher is better)
                # For L2 distance, smaller distance means higher similarity
                similarity_score = 1.0 / (1.0 + distance)
                
                chunk_metadata["relevance_score"] = float(similarity_score)
                chunk_metadata["distance"] = float(distance)
                
                results.append(chunk_metadata)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        return [chunk for chunk in self.metadata if chunk["document_id"] == document_id]
    
    def remove_document(self, document_id: str) -> int:
        """
        Remove all chunks for a document from the vector store.
        Note: This is a simplified implementation. In production, consider using a more efficient approach.
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            Number of chunks removed
        """
        # Find chunks to remove
        chunks_to_remove = [i for i, chunk in enumerate(self.metadata) if chunk["document_id"] == document_id]
        
        if not chunks_to_remove:
            return 0
        
        # Remove metadata entries
        self.metadata = [chunk for chunk in self.metadata if chunk["document_id"] != document_id]
        
        # For FAISS, we need to rebuild the index without the removed vectors
        # This is inefficient but necessary for this implementation
        if self.metadata:
            # Get all remaining texts
            remaining_texts = [chunk["text"] for chunk in self.metadata]
            
            # Regenerate embeddings
            embeddings = self.embedding_model.encode(remaining_texts)
            
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Update vector_ids
            for i, chunk in enumerate(self.metadata):
                chunk["vector_id"] = i
        else:
            # If no metadata left, reinitialize empty index
            self._initialize_empty_index()
        
        # Save updated index
        self._save_index()
        
        return len(chunks_to_remove)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        total_vectors = self.index.ntotal if self.index else 0
        unique_documents = len(set(chunk["document_id"] for chunk in self.metadata))
        
        return {
            "total_vectors": total_vectors,
            "unique_documents": unique_documents,
            "dimension": self.dimension,
            "embedding_model": settings.EMBEDDING_MODEL
        } 