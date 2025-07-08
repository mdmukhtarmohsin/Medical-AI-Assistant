import time
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from .utils import Config, log_ragas_metrics, check_ragas_thresholds
from .vector_store import FAISSVectorStore
from .models import QueryRequest, QueryResponse, RAGASMetrics

logger = logging.getLogger(__name__)


class QueryService:
    """Service for handling medical question answering with RAG and RAGAS evaluation"""
    
    def __init__(self):
        # Configure Gemini
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore()
        
        # System prompt for medical assistant
        self.system_prompt = """You are a medical assistant. Use only the provided document excerpts to answer the following medical question.
If the document doesn't contain the information, reply: "This information is not available in the uploaded document."
Be precise, factual, and cite specific information from the provided context.
Do not make assumptions or provide information not explicitly stated in the context."""
    
    async def answer_question(self, query_request: QueryRequest) -> QueryResponse:
        """Answer a medical question using RAG with RAGAS evaluation"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query for document {query_request.document_id}")
            
            # Retrieve relevant context
            context_chunks = self.vector_store.search(
                query_request.question, 
                query_request.document_id, 
                k=5
            )
            
            if not context_chunks:
                raise Exception("No relevant context found in the document")
            
            # Prepare context for generation
            context_texts = [chunk[0] for chunk in context_chunks]
            context_str = "\n\n".join([f"Context {i+1}:\n{text}" for i, text in enumerate(context_texts)])
            
            # Generate answer using Gemini
            answer = await self._generate_answer(query_request.question, context_str)
            
            # Evaluate with RAGAS
            ragas_metrics = await self._evaluate_with_ragas(
                query_request.question, 
                answer, 
                context_texts
            )
            
            # Check thresholds and generate warning if needed
            passed_thresholds, warning = check_ragas_thresholds(ragas_metrics.__dict__)
            
            # Log metrics
            log_ragas_metrics(
                query_request.document_id, 
                query_request.question, 
                answer, 
                ragas_metrics.__dict__
            )
            
            response_time = time.time() - start_time
            
            return QueryResponse(
                answer=answer,
                document_id=query_request.document_id,
                question=query_request.question,
                ragas_metrics=ragas_metrics,
                context_used=context_texts,
                warning=warning if not passed_thresholds else None,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini 2.5 Pro"""
        try:
            prompt = f"""{self.system_prompt}

Context:
{context}

Question:
{question}

Answer:"""
            
            response = self.model.generate_content(prompt)
            
            if not response.text:
                raise Exception("No response generated from Gemini")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    async def _evaluate_with_ragas(self, question: str, answer: str, contexts: List[str]) -> RAGASMetrics:
        """Evaluate the RAG response using RAGAS metrics"""
        try:
            # Prepare data for RAGAS evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [answer]  # Using generated answer as ground truth for now
            }
            
            dataset = Dataset.from_dict(data)
            
            # Run RAGAS evaluation
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            result = evaluate(dataset, metrics=metrics)
            
            # Extract metric scores
            ragas_metrics = RAGASMetrics(
                faithfulness=float(result.get('faithfulness', 0.0)),
                context_precision=float(result.get('context_precision', 0.0)),
                context_recall=float(result.get('context_recall', 0.0)),
                answer_relevancy=float(result.get('answer_relevancy', 0.0))
            )
            
            logger.info(f"RAGAS evaluation completed: {ragas_metrics}")
            return ragas_metrics
            
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {str(e)}")
            # Return default metrics if RAGAS fails
            return RAGASMetrics(
                faithfulness=0.0,
                context_precision=0.0,
                context_recall=0.0,
                answer_relevancy=0.0
            )
    
    def add_document_to_vector_store(self, document_id: str, chunks: List[str], metadata: Dict):
        """Add document to vector store"""
        self.vector_store.add_document(document_id, chunks, metadata) 