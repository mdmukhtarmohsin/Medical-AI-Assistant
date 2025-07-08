import os
import shutil
from pathlib import Path
from typing import List, Dict
import pdfplumber
from fastapi import UploadFile, HTTPException
import logging

from .utils import Config, generate_document_id
from .models import DocumentUploadResponse

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document upload and processing"""
    
    def __init__(self):
        self.upload_dir = Config.UPLOAD_DIR
        self.upload_dir.mkdir(exist_ok=True)
    
    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        """Upload and process a PDF document"""
        try:
            # Validate file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            if file.size and file.size > Config.MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            
            # Generate document ID and save file
            document_id = generate_document_id(file.filename)
            file_path = self.upload_dir / f"{document_id}.pdf"
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text and create chunks
            text_content = self._extract_text_from_pdf(file_path)
            chunks = self._create_chunks(text_content)
            
            # Save document metadata
            metadata = {
                "document_id": document_id,
                "filename": file.filename,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "chunk_count": len(chunks),
                "text_content": text_content,
                "chunks": chunks
            }
            
            self._save_metadata(document_id, metadata)
            
            logger.info(f"Document uploaded successfully: {document_id}")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                status="success",
                message="Document uploaded and processed successfully",
                chunks_created=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text_content = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = Config.CHUNK_SIZE
        chunk_overlap = Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence ending punctuation
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - chunk_overlap
            if start < 0:
                start = end
        
        return chunks
    
    def _save_metadata(self, document_id: str, metadata: Dict):
        """Save document metadata to JSON file"""
        metadata_path = self.upload_dir / f"{document_id}_metadata.json"
        
        import json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_document_metadata(self, document_id: str) -> Dict:
        """Load document metadata"""
        metadata_path = self.upload_dir / f"{document_id}_metadata.json"
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_documents(self) -> List[Dict]:
        """List all uploaded documents"""
        documents = []
        
        for metadata_file in self.upload_dir.glob("*_metadata.json"):
            try:
                import json
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    documents.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {str(e)}")
        
        return documents 