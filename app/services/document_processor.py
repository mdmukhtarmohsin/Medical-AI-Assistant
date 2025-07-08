"""
Document processor service for extracting text from PDF files.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
import pdfplumber
from datetime import datetime

from config.settings import settings
from app.models.models import DocumentMetadata, UploadStatus


class DocumentProcessor:
    """Service for processing PDF documents and extracting text."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.upload_dir = settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return document ID."""
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create safe filename
        safe_filename = f"{document_id}_{filename}"
        file_path = os.path.join(self.upload_dir, safe_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return document_id
    
    def extract_text_from_pdf(self, document_id: str, filename: str) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            extracted_data = {
                "document_id": document_id,
                "filename": filename,
                "pages": [],
                "full_text": "",
                "metadata": {}
            }
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    page_text = page.extract_text() or ""
                    
                    # Store page information
                    page_info = {
                        "page_number": page_num,
                        "text": page_text,
                        "char_count": len(page_text)
                    }
                    extracted_data["pages"].append(page_info)
                    extracted_data["full_text"] += page_text + "\n\n"
                
                # Extract metadata
                metadata = pdf.metadata or {}
                extracted_data["metadata"] = {
                    "page_count": len(pdf.pages),
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "subject": metadata.get("Subject", ""),
                    "creator": metadata.get("Creator", ""),
                    "creation_date": str(metadata.get("CreationDate", "")),
                    "modification_date": str(metadata.get("ModDate", ""))
                }
            
            return extracted_data
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector store processing.
        
        Args:
            text: Full text to chunk
            chunk_size: Size of each chunk (defaults to settings)
            overlap: Overlap between chunks (defaults to settings)
            
        Returns:
            List of text chunks with metadata
        """
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP
        
        # Simple text chunking by characters
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:  # At least half the chunk size
                    end = start + boundary + 1
                    chunk_text = text[start:end]
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    "chunk_id": str(chunk_id),
                    "text": chunk_text.strip(),
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(chunk_text.strip())
                })
                chunk_id += 1
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def get_document_metadata(self, document_id: str, filename: str) -> DocumentMetadata:
        """Get metadata for a document."""
        file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {document_id}")
        
        file_stats = os.stat(file_path)
        
        # Try to get page count
        page_count = None
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
        except Exception:
            pass  # Page count will remain None
        
        return DocumentMetadata(
            filename=filename,
            file_size=file_stats.st_size,
            upload_timestamp=datetime.fromtimestamp(file_stats.st_ctime),
            content_type="application/pdf",
            page_count=page_count
        )
    
    def delete_document(self, document_id: str, filename: str) -> bool:
        """Delete a document file."""
        file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False 