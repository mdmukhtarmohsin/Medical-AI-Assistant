from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from .utils import setup_logging, ensure_directories
from .models import (
    DocumentUploadResponse, QueryRequest, QueryResponse, 
    DocumentListResponse, DocumentMetadata
)
from .upload import DocumentProcessor
from .query import QueryService

# Setup logging and directories
setup_logging()
ensure_directories()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Assistant",
    description="A RAG-based medical document QA system with RAGAS evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
query_service = QueryService()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Medical AI Assistant...")
    logger.info("Services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Medical AI Assistant...")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Medical AI Assistant API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a medical PDF document for processing
    
    - **file**: PDF file to upload
    
    Returns document ID and processing status
    """
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Process the document
        upload_response = await document_processor.upload_document(file)
        
        # Get document metadata for vector store
        metadata = document_processor.get_document_metadata(upload_response.document_id)
        
        # Add to vector store
        query_service.add_document_to_vector_store(
            upload_response.document_id,
            metadata["chunks"],
            metadata
        )
        
        logger.info(f"Document uploaded and indexed successfully: {upload_response.document_id}")
        return upload_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


@app.post("/query", response_model=QueryResponse)
async def query_document(query_request: QueryRequest):
    """
    Ask a medical question about an uploaded document
    
    - **document_id**: ID of the uploaded document
    - **question**: Medical question to ask
    
    Returns answer with RAGAS evaluation metrics
    """
    try:
        logger.info(f"Processing query for document {query_request.document_id}")
        
        # Verify document exists
        try:
            document_processor.get_document_metadata(query_request.document_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            raise
        
        # Process the query
        response = await query_service.answer_question(query_request)
        
        logger.info(f"Query processed successfully for document {query_request.document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    Get list of all uploaded documents
    
    Returns list of document metadata
    """
    try:
        logger.info("Retrieving document list")
        
        documents = document_processor.list_documents()
        
        # Convert to response format
        document_list = []
        for doc in documents:
            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                filename=doc["filename"],
                upload_date=datetime.fromtimestamp(doc.get("upload_timestamp", 0)) 
                           if "upload_timestamp" in doc 
                           else datetime.now(),
                file_size=doc["file_size"],
                chunk_count=doc["chunk_count"],
                status="processed"
            ))
        
        return DocumentListResponse(
            documents=document_list,
            total_count=len(document_list)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving document list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving documents")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated data
    
    - **document_id**: ID of the document to delete
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Verify document exists
        try:
            document_processor.get_document_metadata(document_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            raise
        
        # Delete from vector store
        query_service.vector_store.delete_document(document_id)
        
        # Delete files (PDF and metadata)
        import os
        from .utils import Config
        
        pdf_path = Config.UPLOAD_DIR / f"{document_id}.pdf"
        metadata_path = Config.UPLOAD_DIR / f"{document_id}_metadata.json"
        
        if pdf_path.exists():
            os.remove(pdf_path)
        
        if metadata_path.exists():
            os.remove(metadata_path)
        
        logger.info(f"Document deleted successfully: {document_id}")
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during deletion")


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Check vector store
        vector_store_status = "healthy"
        try:
            doc_count = len(query_service.vector_store.list_documents())
        except:
            vector_store_status = "error"
            doc_count = 0
        
        # Check document processor
        processor_status = "healthy"
        try:
            uploaded_docs = len(document_processor.list_documents())
        except:
            processor_status = "error"
            uploaded_docs = 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "vector_store": vector_store_status,
                "document_processor": processor_status
            },
            "stats": {
                "documents_in_vector_store": doc_count,
                "uploaded_documents": uploaded_docs
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 