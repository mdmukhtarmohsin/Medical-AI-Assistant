"""
Custom RAGAS Configuration for Google Gemini
This module provides configuration for RAGAS evaluation using Google Gemini models
instead of OpenAI, allowing for cost-effective and powerful evaluation.
"""

import os
from typing import Optional, List, Dict, Any
try:
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        ContextPrecision,
        ContextRecall,
        ContextRelevancy,
        AnswerSimilarity,
        AnswerCorrectness
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
    from ragas import evaluate
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)

class GeminiRagasEvaluator:
    """
    RAGAS evaluator configured to use Google Gemini models instead of OpenAI.
    
    This class provides a complete solution for evaluating RAG pipelines using
    Google's Gemini models for both LLM-based metrics and embeddings.
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        llm_model: str = "gemini-2.0-flash-exp",
        embedding_model: str = "models/embedding-001",
        temperature: float = 0.0
    ):
        """
        Initialize the Gemini RAGAS evaluator.
        
        Args:
            gemini_api_key: Google AI API key
            llm_model: Gemini model to use for LLM-based evaluations
            embedding_model: Gemini embedding model for similarity calculations
            temperature: Temperature for LLM generation (0.0 for deterministic)
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS dependencies not available. Please install ragas and related packages.")
            
        self.gemini_api_key = gemini_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        
        # Initialize LLM and embeddings
        self._init_models()
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_models(self):
        """Initialize Gemini LLM and embedding models."""
        try:
            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
                convert_system_message_to_human=True  # Required for Gemini
            )
            
            # Initialize Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=self.gemini_api_key
            )
            
            # Wrap for RAGAS compatibility
            self.ragas_llm = LangchainLLMWrapper(self.llm)
            self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
            
            logger.info(f"Successfully initialized Gemini models: {self.llm_model}, {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise
    
    def _init_metrics(self):
        """Initialize RAGAS metrics with Gemini models."""
        # Define the metrics we want to use
        self.metrics = [
            Faithfulness(),           # Factual consistency with context
            AnswerRelevancy(),       # Relevance to the question
            ContextPrecision(),      # Precision of retrieved context
            ContextRelevancy(),      # Relevance of context to question
            AnswerSimilarity(),      # Semantic similarity (requires ground truth)
            AnswerCorrectness()      # Overall correctness (requires ground truth)
        ]
        
        # Initialize each metric with our Gemini models
        for metric in self.metrics:
            self._configure_metric(metric)
    
    def _configure_metric(self, metric):
        """Configure a single metric with Gemini models."""
        try:
            # Assign LLM if the metric uses it
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.ragas_llm
            
            # Assign embeddings if the metric uses them
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = self.ragas_embeddings
            
            # Initialize the metric
            run_config = RunConfig()
            metric.init(run_config)
            
        except Exception as e:
            logger.warning(f"Failed to configure metric {metric.__class__.__name__}: {e}")
    
    def evaluate_single(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of context strings used for generation
            ground_truth: Optional ground truth answer for similarity metrics
        
        Returns:
            Dictionary of metric scores
        """
        # Prepare data
        data: Dict[str, List[Any]] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        
        # Add ground truth if provided
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        selected_metrics = self._select_metrics(has_ground_truth=ground_truth is not None)
        
        try:
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            # Convert to simple dictionary
            scores = {}
            for metric_name, score in result.items():
                if hasattr(score, 'item'):  # Convert numpy types
                    scores[metric_name] = float(score.item())
                else:
                    scores[metric_name] = float(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error during single evaluation: {e}")
            return {}
    
    def evaluate_batch(
        self, 
        questions: List[str], 
        answers: List[str], 
        contexts: List[List[str]], 
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of question-answer pairs.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional list of ground truth answers
        
        Returns:
            Dictionary of aggregated metric scores
        """
        # Validate input lengths
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError("All input lists must have the same length")
        
        if ground_truths is not None and len(ground_truths) != len(questions):
            raise ValueError("Ground truths list must match other inputs length")
        
        # Prepare data
        data: Dict[str, List[Any]] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        
        # Add ground truths if provided
        if ground_truths is not None:
            data["ground_truth"] = ground_truths
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        selected_metrics = self._select_metrics(has_ground_truth=ground_truths is not None)
        
        try:
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            # Convert to simple dictionary
            scores = {}
            for metric_name, score in result.items():
                if hasattr(score, 'item'):  # Convert numpy types
                    scores[metric_name] = float(score.item())
                else:
                    scores[metric_name] = float(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            return {}
    
    def _select_metrics(self, has_ground_truth: bool = False):
        """Select appropriate metrics based on available data."""
        # Metrics that don't require ground truth
        reference_free_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRelevancy()
        ]
        
        # Metrics that require ground truth
        reference_based_metrics = [
            ContextRecall(),  # Requires ground truth
            AnswerSimilarity(),  # Requires ground truth
            AnswerCorrectness()  # Requires ground truth
        ]
        
        # Configure selected metrics
        if has_ground_truth:
            selected = reference_free_metrics + reference_based_metrics
        else:
            selected = reference_free_metrics
        
        # Configure each metric
        for metric in selected:
            self._configure_metric(metric)
        
        return selected
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names."""
        return [metric.__class__.__name__ for metric in self.metrics]
    
    def validate_configuration(self) -> bool:
        """Validate that the evaluator is properly configured."""
        try:
            # Test LLM
            test_response = self.llm.invoke("Test message")
            
            # Test embeddings
            test_embedding = self.embeddings.embed_query("Test query")
            
            logger.info("Gemini RAGAS evaluator configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def create_gemini_evaluator(gemini_api_key: Optional[str] = None) -> GeminiRagasEvaluator:
    """
    Factory function to create a Gemini RAGAS evaluator.
    
    Args:
        gemini_api_key: Google AI API key (if not provided, will use GOOGLE_API_KEY env var)
    
    Returns:
        Configured GeminiRagasEvaluator instance
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        
    if not gemini_api_key:
        raise ValueError(
            "Gemini API key is required. Either pass it directly or set GOOGLE_API_KEY environment variable."
        )
    
    return GeminiRagasEvaluator(gemini_api_key=gemini_api_key)


# Example usage
if __name__ == "__main__":
    # Example of how to use the Gemini RAGAS evaluator
    
    # Create evaluator
    evaluator = create_gemini_evaluator()
    
    # Validate configuration
    if evaluator.validate_configuration():
        print("✅ Gemini RAGAS evaluator is ready!")
        print(f"Available metrics: {evaluator.get_available_metrics()}")
        
        # Example evaluation
        question = "What are the benefits of using RAG?"
        answer = "RAG provides better context-aware responses by retrieving relevant information."
        contexts = ["RAG combines retrieval and generation for improved AI responses."]
        
        scores = evaluator.evaluate_single(question, answer, contexts)
        print(f"Evaluation scores: {scores}")
    else:
        print("❌ Configuration validation failed") 