"""Evaluation service using RAGAS with Google Gemini."""
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import time
from datetime import datetime

from config.settings import settings
from ..models.models import EvaluationMetrics, EvaluationRequest, EvaluationResponse

# Import custom Gemini RAGAS evaluator
if TYPE_CHECKING:
    from ..evaluation.custom_ragas_config import GeminiRagasEvaluator

try:
    from ..evaluation.custom_ragas_config import create_gemini_evaluator, GeminiRagasEvaluator
    GEMINI_RAGAS_AVAILABLE = True
except ImportError:
    GEMINI_RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvaluationService:
    """Service for evaluating RAG responses using RAGAS with Google Gemini."""
    
    def __init__(self):
        """Initialize the evaluation service."""
        self.evaluator: Optional["GeminiRagasEvaluator"] = None
        self._initialize_evaluator()
    
    def _initialize_evaluator(self):
        """Initialize the Gemini RAGAS evaluator."""
        if not GEMINI_RAGAS_AVAILABLE:
            logger.warning("Gemini RAGAS evaluator not available. Evaluation will be disabled.")
            return
            
        try:
            # Get Google API key from settings
            google_api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not google_api_key:
                logger.warning("GEMINI_API_KEY not found in settings. Evaluation will be disabled.")
                return
                
            # Create the evaluator
            self.evaluator = create_gemini_evaluator(gemini_api_key=google_api_key)
            
            # Validate configuration
            if self.evaluator and self.evaluator.validate_configuration():
                logger.info("✅ Gemini RAGAS evaluator initialized successfully")
            else:
                logger.error("❌ Gemini RAGAS evaluator validation failed")
                self.evaluator = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini RAGAS evaluator: {e}")
            self.evaluator = None
    
    def is_available(self) -> bool:
        """Check if evaluation service is available."""
        return self.evaluator is not None
    
    async def evaluate_single_response(
        self,
        question: str,
        generated_answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a single RAG response.
        
        Args:
            question: The user question
            generated_answer: The generated answer
            context: List of context strings used for generation
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationMetrics with RAGAS scores
        """
        if not self.is_available() or not self.evaluator:
            logger.warning("Evaluation service not available, returning zero scores")
            return EvaluationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_relevancy=0.0,
                context_recall=0.0,
                overall_score=0.0
            )
        
        try:
            # Perform evaluation
            scores = self.evaluator.evaluate_single(
                question=question,
                answer=generated_answer,
                contexts=context,
                ground_truth=ground_truth
            )
            
            # Map RAGAS scores to our metrics model
            faithfulness = scores.get('faithfulness', 0.0)
            answer_relevancy = scores.get('answer_relevancy', 0.0)
            context_relevancy = scores.get('context_relevancy', 0.0)
            context_recall = scores.get('context_recall', 0.0) if ground_truth else 0.0
            
            # Calculate overall score as average of available metrics
            available_scores = [faithfulness, answer_relevancy, context_relevancy]
            if ground_truth:
                available_scores.append(context_recall)
            
            overall_score = sum(available_scores) / len(available_scores) if available_scores else 0.0
            
            return EvaluationMetrics(
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_relevancy=context_relevancy,
                context_recall=context_recall,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error during response evaluation: {e}")
            return EvaluationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_relevancy=0.0,
                context_recall=0.0,
                overall_score=0.0
            )
    
    async def evaluate_rag_responses(
        self,
        evaluation_request: EvaluationRequest
    ) -> EvaluationResponse:
        """
        Evaluate multiple RAG responses and return aggregated metrics.
        
        Args:
            evaluation_request: Request containing questions, answers, and contexts
            
        Returns:
            EvaluationResponse with aggregated metrics
        """
        if not self.is_available() or not self.evaluator:
            logger.warning("Evaluation service not available, returning zero scores")
            return EvaluationResponse(
                metrics=EvaluationMetrics(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_relevancy=0.0,
                    context_recall=0.0,
                    overall_score=0.0
                ),
                evaluation_timestamp=datetime.now(),
                sample_size=0,
                meets_quality_thresholds=False,
                processing_time_ms=0
            )
        
        start_time = time.time()
        
        try:
            # Perform batch evaluation
            scores = self.evaluator.evaluate_batch(
                questions=evaluation_request.questions,
                answers=evaluation_request.generated_answers,
                contexts=evaluation_request.contexts,
                ground_truths=evaluation_request.ground_truths
            )
            
            # Map RAGAS scores to our metrics model
            faithfulness = scores.get('faithfulness', 0.0)
            answer_relevancy = scores.get('answer_relevancy', 0.0)
            context_relevancy = scores.get('context_relevancy', 0.0)
            context_recall = scores.get('context_recall', 0.0)
            
            # Calculate overall score
            available_scores = [faithfulness, answer_relevancy, context_relevancy]
            if evaluation_request.ground_truths:
                available_scores.append(context_recall)
            
            overall_score = sum(available_scores) / len(available_scores) if available_scores else 0.0
            
            # Create metrics
            metrics = EvaluationMetrics(
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_relevancy=context_relevancy,
                context_recall=context_recall,
                overall_score=overall_score
            )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Check quality thresholds (adjust these as needed)
            meets_thresholds = (
                faithfulness >= 0.7 and
                answer_relevancy >= 0.7 and
                context_relevancy >= 0.7 and
                overall_score >= 0.7
            )
            
            return EvaluationResponse(
                metrics=metrics,
                evaluation_timestamp=datetime.now(),
                sample_size=len(evaluation_request.questions),
                meets_quality_thresholds=meets_thresholds,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return EvaluationResponse(
                metrics=EvaluationMetrics(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_relevancy=0.0,
                    context_recall=0.0,
                    overall_score=0.0
                ),
                evaluation_timestamp=datetime.now(),
                sample_size=len(evaluation_request.questions),
                meets_quality_thresholds=False,
                processing_time_ms=processing_time_ms
            )
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """
        Get information about available evaluation metrics.
        
        Returns:
            Dictionary containing metrics information
        """
        if not self.is_available():
            return {
                "available": False,
                "error": "Evaluation service not available",
                "metrics": []
            }
        
        metrics_info = {
            "available": True,
            "model_used": getattr(self.evaluator, 'llm_model', 'gemini-2.0-flash-exp'),
            "embedding_model": getattr(self.evaluator, 'embedding_model', 'models/embedding-001'),
            "metrics": [
                {
                    "name": "Faithfulness",
                    "description": "Measures factual consistency of the answer with the given context",
                    "requires_ground_truth": False
                },
                {
                    "name": "AnswerRelevancy", 
                    "description": "Assesses how pertinent the answer is to the given question",
                    "requires_ground_truth": False
                },
                {
                    "name": "ContextPrecision",
                    "description": "Evaluates the precision of the retrieved context",
                    "requires_ground_truth": False
                },
                {
                    "name": "ContextRelevancy",
                    "description": "Measures how relevant the context is to the question",
                    "requires_ground_truth": False
                },
                {
                    "name": "ContextRecall",
                    "description": "Assesses the extent to which relevant context is retrieved",
                    "requires_ground_truth": True
                },
                {
                    "name": "AnswerSimilarity",
                    "description": "Quantifies semantic similarity between generated and expected answers",
                    "requires_ground_truth": True
                },
                {
                    "name": "AnswerCorrectness",
                    "description": "Focuses on factual accuracy of the generated answer",
                    "requires_ground_truth": True
                }
            ]
        }
        
        return metrics_info


# Global evaluation service instance
evaluation_service = EvaluationService() 