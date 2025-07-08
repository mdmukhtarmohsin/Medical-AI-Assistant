from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    chunks_created: int


class QueryRequest(BaseModel):
    """Request model for medical question"""
    document_id: str = Field(..., description="ID of the uploaded document")
    question: str = Field(..., min_length=1, description="Medical question to ask")


class RAGASMetrics(BaseModel):
    """RAGAS evaluation metrics"""
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response model for medical question answering"""
    answer: str
    document_id: str
    question: str
    ragas_metrics: RAGASMetrics
    context_used: List[str]
    warning: Optional[str] = None
    response_time: float


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    document_id: str
    filename: str
    upload_date: datetime
    file_size: int
    chunk_count: int
    status: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    documents: List[DocumentMetadata]
    total_count: int


class EvaluationResult(BaseModel):
    """Model for batch evaluation results"""
    document_id: str
    question: str
    answer: str
    ragas_metrics: RAGASMetrics
    passed_thresholds: bool 