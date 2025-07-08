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
from app.services.evaluation_service import RAGASEvaluationService

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global application instance (in production, use dependency injection)
app_instance = None

# Initialize evaluation service
evaluation_service = RAGASEvaluationService()

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
    
    Args:
        request: Evaluation request with questions, answers, and contexts
        
    Returns:
        EvaluationResponse with RAGAS metrics
    """
    try:
        # Validate request
        if not request.questions or not request.answers or not request.contexts_list:
            raise HTTPException(
                status_code=400,
                detail="Questions, answers, and contexts are required"
            )
        
        if not (len(request.questions) == len(request.answers) == len(request.contexts_list)):
            raise HTTPException(
                status_code=400,
                detail="Questions, answers, and contexts must have the same length"
            )
        
        # Run evaluation
        evaluation_result = await evaluation_service.evaluate_batch(
            questions=request.questions,
            answers=request.answers,
            contexts_list=request.contexts_list,
            ground_truths=request.ground_truth_answers
        )
        
        return evaluation_result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/evaluate/single")
async def evaluate_single_response(
    question: str = Form(...),
    answer: str = Form(...),
    contexts: List[str] = Form(...),
    ground_truth: Optional[str] = Form(None),
    app: MedicalAIAssistant = Depends(get_app)
):
    """
    Evaluate a single question-answer pair using RAGAS metrics.
    
    Args:
        question: The question
        answer: The generated answer
        contexts: List of context chunks
        ground_truth: Optional ground truth answer
        
    Returns:
        EvaluationMetrics
    """
    try:
        if not question.strip() or not answer.strip():
            raise HTTPException(
                status_code=400,
                detail="Question and answer cannot be empty"
            )
        
        if not contexts:
            raise HTTPException(
                status_code=400,
                detail="At least one context is required"
            )
        
        # Run single evaluation
        metrics = await evaluation_service.evaluate_single_response(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single evaluation failed: {str(e)}"
        )


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
        BatchEvaluationReport with detailed results
    """
    try:
        # Validate document exists
        doc_info = app.get_document_info(document_id)
        if doc_info is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        if not questions:
            raise HTTPException(
                status_code=400,
                detail="At least one question is required"
            )
        
        # Generate answers for all questions
        answers = []
        contexts_list = []
        results = []
        
        for question in questions:
            try:
                # Get answer from the assistant
                answer_response = await app.ask_question(
                    question=question,
                    document_id=document_id,
                    include_sources=True
                )
                
                answers.append(answer_response.answer)
                
                # Extract contexts from sources
                contexts = [source.content_preview for source in answer_response.sources]
                contexts_list.append(contexts)
                
                # Evaluate this single response
                metrics = await evaluation_service.evaluate_single_response(
                    question=question,
                    answer=answer_response.answer,
                    contexts=contexts
                )
                
                # Check if it meets thresholds
                meets_thresholds = evaluation_service._check_thresholds(metrics)
                
                results.append({
                    'question': question,
                    'answer': answer_response.answer,
                    'contexts': contexts,
                    'document_id': document_id,
                    'metrics': metrics,
                    'meets_thresholds': meets_thresholds
                })
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="Failed to evaluate any questions"
            )
        
        # Calculate aggregate metrics
        successful_evaluations = len(results)
        total_evaluations = len(questions)
        failed_evaluations = total_evaluations - successful_evaluations
        
        # Calculate averages
        avg_faithfulness = sum(r['metrics'].faithfulness for r in results) / successful_evaluations
        avg_answer_relevancy = sum(r['metrics'].answer_relevancy for r in results) / successful_evaluations
        avg_context_relevancy = sum(r['metrics'].context_relevancy for r in results) / successful_evaluations
        avg_context_recall = sum(r['metrics'].context_recall for r in results) / successful_evaluations
        avg_overall_score = sum(r['metrics'].overall_score for r in results) / successful_evaluations
        
        threshold_pass_rate = sum(1 for r in results if r['meets_thresholds']) / successful_evaluations
        
        # Create report
        from app.models.models import EvaluationMetrics, BatchEvaluationResult, BatchEvaluationReport
        
        average_metrics = EvaluationMetrics(
            faithfulness=avg_faithfulness,
            answer_relevancy=avg_answer_relevancy,
            context_relevancy=avg_context_relevancy,
            context_recall=avg_context_recall,
            overall_score=avg_overall_score
        )
        
        detailed_results = [
            BatchEvaluationResult(
                question=r['question'],
                answer=r['answer'],
                contexts=r['contexts'],
                document_id=r['document_id'],
                metrics=r['metrics'],
                meets_thresholds=r['meets_thresholds']
            )
            for r in results
        ]
        
        report = BatchEvaluationReport(
            total_evaluations=total_evaluations,
            successful_evaluations=successful_evaluations,
            failed_evaluations=failed_evaluations,
            average_metrics=average_metrics,
            threshold_pass_rate=threshold_pass_rate,
            report_timestamp=datetime.now(),
            detailed_results=detailed_results
        )
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document evaluation failed: {str(e)}"
        ) 