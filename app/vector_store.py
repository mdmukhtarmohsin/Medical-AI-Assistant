import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
import logging

from .utils import Config
from .embed import EmbeddingService

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self):
        self.index_dir = Config.FAISS_INDEX_DIR
        self.index_dir.mkdir(exist_ok=True)
        
        self.embedding_service = EmbeddingService()
        self.dimension = self.embedding_service.get_embedding_dimension()
        
        self.index = None
        self.document_chunks = {}  # Maps chunk_id to {document_id, chunk_text, chunk_index}
        self.document_metadata = {}  # Maps document_id to metadata
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.pkl"
        chunks_path = self.index_dir / "chunks.pkl"
        
        try:
            if index_path.exists() and metadata_path.exists() and chunks_path.exists():
                logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, "rb") as f:
                    self.document_metadata = pickle.load(f)
                
                with open(chunks_path, "rb") as f:
                    self.document_chunks = pickle.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            else:
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
                self.document_chunks = {}
                self.document_metadata = {}
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            logger.info("Creating new index due to error")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.document_chunks = {}
            self.document_metadata = {}
    
    def add_document(self, document_id: str, chunks: List[str], metadata: Dict):
        """Add document chunks to the vector store"""
        try:
            logger.info(f"Adding document {document_id} with {len(chunks)} chunks")
            
            # Create embeddings for all chunks
            embeddings = self.embedding_service.embed_texts(chunks)
            
            # Get current index size to generate chunk IDs
            start_chunk_id = self.index.ntotal
            
            # Add embeddings to FAISS index
            self.index.add(embeddings)
            
            # Store chunk metadata
            for i, chunk_text in enumerate(chunks):
                chunk_id = start_chunk_id + i
                self.document_chunks[chunk_id] = {
                    "document_id": document_id,
                    "chunk_text": chunk_text,
                    "chunk_index": i
                }
            
            # Store document metadata
            self.document_metadata[document_id] = {
                **metadata,
                "chunk_start_id": start_chunk_id,
                "chunk_count": len(chunks)
            }
            
            # Save updated index and metadata
            self._save_index()
            
            logger.info(f"Successfully added document {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def search(self, query: str, document_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant chunks in a specific document"""
        try:
            if self.index.ntotal == 0:
                logger.warning("No documents in vector store")
                return []
            
            # Create query embedding
            query_embedding = self.embedding_service.embed_single_text(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(k * 3, self.index.ntotal))
            
            # Filter results for the specific document and prepare response
            results = []
            doc_metadata = self.document_metadata.get(document_id)
            
            if not doc_metadata:
                logger.warning(f"Document {document_id} not found in vector store")
                return []
            
            chunk_start_id = doc_metadata["chunk_start_id"]
            chunk_end_id = chunk_start_id + doc_metadata["chunk_count"]
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for padding
                    continue
                
                # Check if this chunk belongs to the requested document
                if chunk_start_id <= idx < chunk_end_id:
                    chunk_info = self.document_chunks[idx]
                    results.append((chunk_info["chunk_text"], float(score)))
                
                if len(results) >= k:
                    break
            
            logger.info(f"Found {len(results)} relevant chunks for query in document {document_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """Get document information"""
        return self.document_metadata.get(document_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the vector store"""
        return list(self.document_metadata.keys())
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = self.index_dir / "faiss_index.bin"
            metadata_path = self.index_dir / "metadata.pkl"
            chunks_path = self.index_dir / "chunks.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            with open(metadata_path, "wb") as f:
                pickle.dump(self.document_metadata, f)
            
            # Save chunks
            with open(chunks_path, "wb") as f:
                pickle.dump(self.document_chunks, f)
                
            logger.info("FAISS index and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def delete_document(self, document_id: str):
        """Delete a document from the vector store (note: FAISS doesn't support deletion, so we rebuild)"""
        if document_id not in self.document_metadata:
            logger.warning(f"Document {document_id} not found")
            return
        
        logger.info(f"Deleting document {document_id} (rebuilding index)")
        
        # Remove from metadata
        del self.document_metadata[document_id]
        
        # Remove chunks
        chunks_to_remove = [chunk_id for chunk_id, chunk_info in self.document_chunks.items() 
                          if chunk_info["document_id"] == document_id]
        
        for chunk_id in chunks_to_remove:
            del self.document_chunks[chunk_id]
        
        # Rebuild index (FAISS doesn't support deletion)
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild FAISS index after deletion"""
        logger.info("Rebuilding FAISS index")
        
        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if not self.document_chunks:
            self._save_index()
            return
        
        # Collect all chunks and recreate embeddings
        all_chunks = []
        new_chunk_mapping = {}
        
        for doc_id, doc_metadata in self.document_metadata.items():
            doc_chunks = []
            for chunk_id, chunk_info in self.document_chunks.items():
                if chunk_info["document_id"] == doc_id:
                    doc_chunks.append((chunk_info["chunk_index"], chunk_info["chunk_text"]))
            
            # Sort by chunk index
            doc_chunks.sort(key=lambda x: x[0])
            chunk_texts = [chunk[1] for chunk in doc_chunks]
            
            start_id = len(all_chunks)
            all_chunks.extend(chunk_texts)
            
            # Update metadata
            self.document_metadata[doc_id]["chunk_start_id"] = start_id
            
            # Create new chunk mapping
            for i, chunk_text in enumerate(chunk_texts):
                new_chunk_mapping[start_id + i] = {
                    "document_id": doc_id,
                    "chunk_text": chunk_text,
                    "chunk_index": i
                }
        
        # Create embeddings and add to index
        if all_chunks:
            embeddings = self.embedding_service.embed_texts(all_chunks)
            self.index.add(embeddings)
        
        # Update chunk mapping
        self.document_chunks = new_chunk_mapping
        
        # Save updated index
        self._save_index()
        
        logger.info("Index rebuilt successfully") 