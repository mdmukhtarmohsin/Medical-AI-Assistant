"""
FastAPI routes for the Medical AI Assistant.
"""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import aiofiles
from datetime import datetime

from app.core.app import MedicalAIAssistant
from app.models.models import (
    QuestionRequest, AnswerResponse, DocumentListResponse,
    DocumentUploadResponse, HealthCheckResponse, ErrorResponse,
    EvaluationRequest, EvaluationResponse, BatchEvaluationReport
)
from config.settings import settings
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.document_processor import DocumentProcessor
from app.services.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global application instance (in production, use dependency injection)
app_instance = None

# Initialize services
vector_store = VectorStore()
llm_service = LLMService()
doc_processor = DocumentProcessor()
# evaluation_service is imported directly as a singleton

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
    max_file_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
    if len(content) > max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
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


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_responses(
    request: EvaluationRequest,
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Evaluate a batch of questions and answers using RAGAS metrics.
    """
    try:
        logger.info(f"Starting RAGAS evaluation for {len(request.questions)} questions")
        
        if not evaluation_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Evaluation service is not available. Please check GEMINI_API_KEY configuration."
            )
        
        # Validate input
        if not request.questions or not request.generated_answers or not request.contexts:
            raise HTTPException(status_code=400, detail="Questions, answers, and contexts are required")
        
        if len(request.questions) != len(request.generated_answers) or len(request.questions) != len(request.contexts):
            raise HTTPException(status_code=400, detail="Questions, answers, and contexts must have the same length")
        
        # Evaluate using the evaluation service
        evaluation_result = await evaluation_service.evaluate_rag_responses(request)
        
        logger.info(f"RAGAS evaluation completed successfully")
        return evaluation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/evaluate/single")
async def evaluate_single_response(
    question: str = Form(...),
    answer: str = Form(...),
    contexts: List[str] = Form(...),
    ground_truth: str = Form(None),
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Evaluate a single question-answer pair using RAGAS metrics.
    
    Args:
        question: The user question
        answer: The generated answer  
        contexts: List of context strings (one per line)
        ground_truth: Optional ground truth answer
    
    Returns:
        Evaluation metrics for the single response
    """
    try:
        logger.info("Starting single response evaluation")
        
        if not evaluation_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Evaluation service is not available. Please check GEMINI_API_KEY configuration."
            )
        
        # Evaluate the single response
        metrics = await evaluation_service.evaluate_single_response(
            question=question,
            generated_answer=answer,
            context=contexts,
            ground_truth=ground_truth
        )
        
        # Return metrics as dict
        result = {
            "faithfulness": metrics.faithfulness,
            "answer_relevancy": metrics.answer_relevancy, 
            "context_relevancy": metrics.context_relevancy,
            "context_recall": metrics.context_recall,
            "overall_score": metrics.overall_score
        }
        
        logger.info("Single response evaluation completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Single evaluation failed: {str(e)}")


@router.post("/evaluate/document/{document_id}")
async def evaluate_document_responses(
    document_id: str,
    questions: List[str] = Form(...),
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Evaluate responses for a specific document using RAGAS metrics.
    
    Args:
        document_id: Document ID to evaluate
        questions: List of questions to ask about the document
    
    Returns:
        Aggregated evaluation metrics for all questions
    """
    try:
        logger.info(f"Starting document evaluation for {document_id} with {len(questions)} questions")
        
        if not evaluation_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Evaluation service is not available. Please check GEMINI_API_KEY configuration."
            )
        
        # Get answers for each question from the document
        results = []
        successful_evaluations = 0
        failed_evaluations = 0
        all_scores = []
        
        for question in questions:
            try:
                # Get answer from document
                response = await app.ask_question(question, document_id)
                
                if response.get("answer"):
                    # Extract contexts from sources
                    contexts = []
                    if response.get("sources"):
                        contexts = [source.get("content_preview", "") for source in response["sources"]]
                    
                    # Evaluate this single response
                    metrics = await evaluation_service.evaluate_single_response(
                        question=question,
                        generated_answer=response["answer"],
                        context=contexts
                    )
                    
                    scores = {
                        "faithfulness": metrics.faithfulness,
                        "answer_relevancy": metrics.answer_relevancy,
                        "context_relevancy": metrics.context_relevancy,
                        "context_recall": metrics.context_recall,
                        "overall_score": metrics.overall_score
                    }
                    
                    all_scores.append(scores)
                    successful_evaluations += 1
                    
                    results.append({
                        "question": question,
                        "answer": response["answer"],
                        "metrics": scores
                    })
                else:
                    failed_evaluations += 1
                    logger.warning(f"No answer generated for question: {question}")
                    
            except Exception as e:
                failed_evaluations += 1
                logger.error(f"Failed to evaluate question '{question}': {str(e)}")
        
        if successful_evaluations == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to evaluate any questions"
            )
        
        # Calculate average metrics
        avg_metrics = {}
        if all_scores:
            for metric in ["faithfulness", "answer_relevancy", "context_relevancy", "context_recall", "overall_score"]:
                avg_metrics[metric] = sum(score[metric] for score in all_scores) / len(all_scores)
        
        # Calculate threshold pass rate (assuming 0.7 threshold)
        threshold_passes = sum(1 for score in all_scores if score["overall_score"] >= 0.7)
        threshold_pass_rate = threshold_passes / len(all_scores) if all_scores else 0.0
        
        result = {
            "document_id": document_id,
            "total_evaluations": len(questions),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "average_metrics": avg_metrics,
            "threshold_pass_rate": threshold_pass_rate,
            "detailed_results": results
        }
        
        logger.info(f"Document evaluation completed: {successful_evaluations}/{len(questions)} successful")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document evaluation failed: {str(e)}") 