"""
Main application core for the Medical AI Assistant.
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.models.models import (
    DocumentUploadResponse, UploadStatus, AnswerResponse, 
    DocumentListResponse, DocumentListItem
)


class MedicalAIAssistant:
    """Main application class that coordinates all services."""
    
    def __init__(self):
        """Initialize the Medical AI Assistant."""
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        
        # In-memory storage for document metadata (in production, use a database)
        self.documents_registry: Dict[str, Dict[str, Any]] = {}
    
    async def upload_document(self, file_content: bytes, filename: str) -> DocumentUploadResponse:
        """
        Upload and process a PDF document.
        
        Args:
            file_content: Binary content of the PDF file
            filename: Original filename
            
        Returns:
            DocumentUploadResponse with upload status and metadata
        """
        try:
            # Save the uploaded file
            document_id = await self.document_processor.save_uploaded_file(file_content, filename)
            
            # Register document with pending status
            self.documents_registry[document_id] = {
                "filename": filename,
                "status": UploadStatus.PENDING,
                "upload_timestamp": datetime.now(),
                "processing_error": None
            }
            
            # Start processing in background (for demo, we'll do it synchronously)
            await self._process_document(document_id, filename)
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                status=self.documents_registry[document_id]["status"],
                message="Document uploaded and processed successfully",
                upload_timestamp=self.documents_registry[document_id]["upload_timestamp"]
            )
            
        except Exception as e:
            return DocumentUploadResponse(
                document_id="",
                filename=filename,
                status=UploadStatus.FAILED,
                message=f"Upload failed: {str(e)}",
                upload_timestamp=datetime.now()
            )
    
    async def _process_document(self, document_id: str, filename: str):
        """Process a document: extract text, chunk it, and add to vector store."""
        try:
            # Update status to processing
            self.documents_registry[document_id]["status"] = UploadStatus.PROCESSING
            
            # Extract text from PDF
            extracted_data = self.document_processor.extract_text_from_pdf(document_id, filename)
            
            # Chunk the text
            chunks = self.document_processor.chunk_text(extracted_data["full_text"])
            
            if not chunks:
                raise Exception("No text content found in the document")
            
            # Add chunks to vector store
            chunks_added = self.vector_store.add_document_chunks(document_id, filename, chunks)
            
            # Update status to completed
            self.documents_registry[document_id].update({
                "status": UploadStatus.COMPLETED,
                "page_count": extracted_data["metadata"]["page_count"],
                "chunks_count": chunks_added,
                "metadata": extracted_data["metadata"]
            })
            
        except Exception as e:
            # Update status to failed
            self.documents_registry[document_id].update({
                "status": UploadStatus.FAILED,
                "processing_error": str(e)
            })
    
    async def ask_question(
        self, 
        question: str, 
        document_id: Optional[str] = None,
        include_sources: bool = True,
        k: int = 5
    ) -> AnswerResponse:
        """
        Ask a question and get an answer based on uploaded documents.
        
        Args:
            question: The question to ask
            document_id: Optional specific document to search in
            include_sources: Whether to include source references
            k: Number of relevant chunks to retrieve
            
        Returns:
            AnswerResponse with the generated answer
        """
        try:
            # Search for relevant chunks
            relevant_chunks = self.vector_store.search_similar_chunks(
                query=question,
                k=k,
                document_id=document_id
            )
            
            # Generate answer using LLM
            answer = self.llm_service.generate_answer(
                question=question,
                relevant_chunks=relevant_chunks,
                include_sources=include_sources
            )
            
            return answer
            
        except Exception as e:
            return AnswerResponse(
                question=question,
                answer=f"Error processing question: {str(e)}",
                confidence_score=0.0,
                sources=[],
                response_timestamp=datetime.now(),
                processing_time_ms=0
            )
    
    def get_documents(self) -> DocumentListResponse:
        """Get list of all uploaded documents."""
        documents = []
        
        for doc_id, doc_info in self.documents_registry.items():
            # Get file size from document processor
            try:
                metadata = self.document_processor.get_document_metadata(doc_id, doc_info["filename"])
                file_size = metadata.file_size
            except Exception:
                file_size = 0
            
            doc_item = DocumentListItem(
                document_id=doc_id,
                filename=doc_info["filename"],
                upload_timestamp=doc_info["upload_timestamp"],
                status=doc_info["status"],
                page_count=doc_info.get("page_count"),
                file_size=file_size
            )
            documents.append(doc_item)
        
        # Sort by upload timestamp (most recent first)
        documents.sort(key=lambda x: x.upload_timestamp, reverse=True)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document."""
        if document_id not in self.documents_registry:
            return None
        
        doc_info = self.documents_registry[document_id].copy()
        
        # Add vector store information
        chunks = self.vector_store.get_document_chunks(document_id)
        doc_info["chunks_in_vector_store"] = len(chunks)
        
        return doc_info
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its associated data."""
        try:
            if document_id not in self.documents_registry:
                return False
            
            doc_info = self.documents_registry[document_id]
            filename = doc_info["filename"]
            
            # Remove from vector store
            self.vector_store.remove_document(document_id)
            
            # Delete physical file
            self.document_processor.delete_document(document_id, filename)
            
            # Remove from registry
            del self.documents_registry[document_id]
            
            return True
            
        except Exception:
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        vector_stats = self.vector_store.get_index_stats()
        
        # Document status counts
        status_counts = {}
        for doc_info in self.documents_registry.values():
            status = doc_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_documents": len(self.documents_registry),
            "document_status_counts": status_counts,
            "vector_store_stats": vector_stats,
            "llm_healthy": self.llm_service.health_check()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        return {
            "status": "healthy",
            "services": {
                "document_processor": True,  # Always available
                "vector_store": self.vector_store.index is not None,
                "llm_service": self.llm_service.health_check()
            },
            "timestamp": datetime.now().isoformat()
        } 