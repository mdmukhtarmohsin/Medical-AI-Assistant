"""
RAGAS evaluation service for the Medical AI Assistant.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import pandas as pd

from config.settings import settings
from app.models.models import EvaluationMetrics, EvaluationRequest, EvaluationResponse

logger = logging.getLogger(__name__)


class RAGASEvaluationService:
    """Service for evaluating responses using RAGAS metrics."""
    
    def __init__(self):
        """Initialize the RAGAS evaluation service."""
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
    async def evaluate_single_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a single Q&A response using RAGAS metrics.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of context chunks used for generation
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationMetrics with RAGAS scores
        """
        try:
            # Use the generated answer as ground truth if not provided
            if ground_truth is None:
                ground_truth = answer
            
            # Prepare data for RAGAS evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Run RAGAS evaluation in a separate thread to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, evaluate, dataset, self.metrics
            )
            
            # Extract metrics
            metrics = EvaluationMetrics(
                faithfulness=float(result.get('faithfulness', 0.0)),
                answer_relevancy=float(result.get('answer_relevancy', 0.0)),
                context_relevancy=float(result.get('context_precision', 0.0)),
                context_recall=float(result.get('context_recall', 0.0)),
                overall_score=self._calculate_overall_score(result)
            )
            
            logger.info(f"RAGAS evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {str(e)}")
            # Return default metrics if RAGAS fails
            return EvaluationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_relevancy=0.0,
                context_recall=0.0,
                overall_score=0.0
            )
    
    async def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> EvaluationResponse:
        """
        Evaluate a batch of Q&A responses.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists for each question
            ground_truths: Optional list of ground truth answers
            
        Returns:
            EvaluationResponse with aggregated metrics
        """
        start_time = time.time()
        
        try:
            # Validate input lengths
            if not (len(questions) == len(answers) == len(contexts_list)):
                raise ValueError("All input lists must have the same length")
            
            if ground_truths is None:
                ground_truths = answers.copy()
            elif len(ground_truths) != len(questions):
                raise ValueError("Ground truths list must have the same length as questions")
            
            # Prepare data for RAGAS evaluation
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths
            }
            
            dataset = Dataset.from_dict(data)
            
            # Run RAGAS evaluation
            result = await asyncio.get_event_loop().run_in_executor(
                None, evaluate, dataset, self.metrics
            )
            
            # Calculate aggregated metrics
            metrics = EvaluationMetrics(
                faithfulness=float(result.get('faithfulness', 0.0)),
                answer_relevancy=float(result.get('answer_relevancy', 0.0)),
                context_relevancy=float(result.get('context_precision', 0.0)),
                context_recall=float(result.get('context_recall', 0.0)),
                overall_score=self._calculate_overall_score(result)
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Check if metrics meet thresholds
            meets_thresholds = self._check_thresholds(metrics)
            
            return EvaluationResponse(
                metrics=metrics,
                evaluation_timestamp=datetime.now(),
                sample_size=len(questions),
                meets_quality_thresholds=meets_thresholds,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return EvaluationResponse(
                metrics=EvaluationMetrics(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_relevancy=0.0,
                    context_recall=0.0,
                    overall_score=0.0
                ),
                evaluation_timestamp=datetime.now(),
                sample_size=len(questions),
                meets_quality_thresholds=False,
                processing_time_ms=processing_time
            )
    
    def _calculate_overall_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall score from RAGAS metrics."""
        metrics = [
            result.get('faithfulness', 0.0),
            result.get('answer_relevancy', 0.0),
            result.get('context_precision', 0.0),
            result.get('context_recall', 0.0)
        ]
        
        # Calculate weighted average (you can adjust weights based on importance)
        weights = [0.3, 0.25, 0.25, 0.2]  # Slightly more weight on faithfulness
        
        overall = sum(m * w for m, w in zip(metrics, weights))
        return float(overall)
    
    def _check_thresholds(self, metrics: EvaluationMetrics) -> bool:
        """Check if metrics meet quality thresholds."""
        return (
            metrics.faithfulness >= settings.RAGAS_FAITHFULNESS_THRESHOLD and
            metrics.context_relevancy >= settings.RAGAS_CONTEXT_PRECISION_THRESHOLD
        )
    
    def create_evaluation_report(self, results: List[Dict[str, Any]]) -> str:
        """Create a detailed evaluation report."""
        if not results:
            return "No evaluation results to report."
        
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'total_questions': len(results),
            'avg_faithfulness': df['faithfulness'].mean(),
            'avg_answer_relevancy': df['answer_relevancy'].mean(),
            'avg_context_relevancy': df['context_relevancy'].mean(),
            'avg_context_recall': df['context_recall'].mean(),
            'avg_overall_score': df['overall_score'].mean(),
            'meets_thresholds_count': df['meets_thresholds'].sum(),
            'meets_thresholds_percent': (df['meets_thresholds'].sum() / len(results)) * 100
        }
        
        report = f"""
=== RAGAS EVALUATION REPORT ===

Total Questions Evaluated: {summary['total_questions']}
Questions Meeting Thresholds: {summary['meets_thresholds_count']} ({summary['meets_thresholds_percent']:.1f}%)

AVERAGE METRICS:
- Faithfulness: {summary['avg_faithfulness']:.3f} (threshold: {settings.RAGAS_FAITHFULNESS_THRESHOLD})
- Answer Relevancy: {summary['avg_answer_relevancy']:.3f}
- Context Relevancy: {summary['avg_context_relevancy']:.3f} (threshold: {settings.RAGAS_CONTEXT_PRECISION_THRESHOLD})
- Context Recall: {summary['avg_context_recall']:.3f}
- Overall Score: {summary['avg_overall_score']:.3f}

QUALITY ASSESSMENT:
"""
        
        if summary['meets_thresholds_percent'] >= 80:
            report += "✅ EXCELLENT: System meets quality thresholds for most questions\n"
        elif summary['meets_thresholds_percent'] >= 60:
            report += "⚠️  GOOD: System meets quality thresholds for majority of questions\n"
        elif summary['meets_thresholds_percent'] >= 40:
            report += "⚠️  FAIR: System needs improvement in quality metrics\n"
        else:
            report += "❌ POOR: System requires significant improvement\n"
        
        report += "\n" + "="*50
        
        return report 