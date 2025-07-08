"""
FastAPI routes for the Medical AI Assistant.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional
import aiofiles
from datetime import datetime

from app.core.app import MedicalAIAssistant
from app.models.models import (
    QuestionRequest, AnswerResponse, DocumentListResponse,
    DocumentUploadResponse, HealthCheckResponse, ErrorResponse
)
from config.settings import settings

# Initialize router
router = APIRouter()

# Global application instance (in production, use dependency injection)
app_instance = None

def get_app() -> MedicalAIAssistant:
    """Get the application instance."""
    global app_instance
    if app_instance is None:
        app_instance = MedicalAIAssistant()
    return app_instance


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Upload a PDF document for processing.
    
    Args:
        file: PDF file to upload
        
    Returns:
        DocumentUploadResponse with upload status
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    # Validate file size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file is not allowed"
        )
    
    try:
        # Upload and process document
        response = await app.upload_document(content, file.filename)
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Ask a question and get an answer based on uploaded documents.
    
    Args:
        request: Question request with query parameters
        
    Returns:
        AnswerResponse with generated answer and sources
    """
    try:
        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Get answer
        answer = await app.ask_question(
            question=request.question,
            document_id=request.document_id,
            include_sources=request.include_sources
        )
        
        return answer
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(app: MedicalAIAssistant = Depends(get_app)):
    """
    Get list of all uploaded documents.
    
    Returns:
        DocumentListResponse with list of documents
    """
    try:
        return app.get_documents()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


@router.get("/documents/{document_id}")
async def get_document_info(
    document_id: str,
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Get detailed information about a specific document.
    
    Args:
        document_id: Document ID to retrieve
        
    Returns:
        Document information
    """
    try:
        doc_info = app.get_document_info(document_id)
        if doc_info is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        return doc_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document info: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Delete a document and all its associated data.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Success message
    """
    try:
        success = await app.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found or could not be deleted"
            )
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(app: MedicalAIAssistant = Depends(get_app)):
    """
    Perform a health check of the application.
    
    Returns:
        HealthCheckResponse with system status
    """
    try:
        health_info = app.health_check()
        
        return HealthCheckResponse(
            status=health_info["status"],
            version=settings.APP_VERSION,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            timestamp=datetime.now()
        )


@router.get("/stats")
async def get_system_stats(app: MedicalAIAssistant = Depends(get_app)):
    """
    Get system statistics and metrics.
    
    Returns:
        System statistics
    """
    try:
        return app.get_system_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system stats: {str(e)}"
        )


# Quick question endpoint for simple queries without file upload
@router.post("/quick-ask")
async def quick_ask(
    question: str = Form(...),
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Quick question endpoint for simple form-based queries.
    
    Args:
        question: Question as form data
        
    Returns:
        Answer response
    """
    try:
        if not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        answer = await app.ask_question(
            question=question,
            document_id=None,
            include_sources=True
        )
        
        return answer
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        ) 