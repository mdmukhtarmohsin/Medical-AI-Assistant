"""
LLM service for generating answers using Google Gemini.
"""
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from config.settings import settings
from app.models.models import AnswerResponse, SourceReference


class LLMService:
    """Service for generating answers using Google Gemini."""
    
    def __init__(self):
        """Initialize the LLM service."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Generation configuration
        self.generation_config = {
            "temperature": settings.GEMINI_TEMPERATURE,
            "max_output_tokens": settings.GEMINI_MAX_TOKENS,
        }
    
    def generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]], 
        include_sources: bool = True
    ) -> AnswerResponse:
        """
        Generate an answer based on the question and relevant document chunks.
        
        Args:
            question: The user's question
            relevant_chunks: List of relevant document chunks
            include_sources: Whether to include source references
            
        Returns:
            AnswerResponse with the generated answer and metadata
        """
        start_time = time.time()
        
        # Prepare context from relevant chunks
        context = self._prepare_context(relevant_chunks)
        
        # Create the prompt
        prompt = self._create_prompt(question, context)
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            answer_text = response.text if response.text else "I couldn't generate an answer for this question."
            
            # Calculate confidence score based on relevance scores
            confidence_score = self._calculate_confidence(relevant_chunks)
            
            # Prepare source references
            sources = []
            if include_sources:
                sources = self._create_source_references(relevant_chunks)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return AnswerResponse(
                question=question,
                answer=answer_text,
                confidence_score=confidence_score,
                sources=sources,
                response_timestamp=datetime.now(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            # Return error response
            processing_time = int((time.time() - start_time) * 1000)
            
            return AnswerResponse(
                question=question,
                answer=f"Error generating answer: {str(e)}",
                confidence_score=0.0,
                sources=[],
                response_timestamp=datetime.now(),
                processing_time_ms=processing_time
            )
    
    def _prepare_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from relevant chunks."""
        if not relevant_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_part = f"[Context {i}]\n"
            context_part += f"Source: {chunk.get('filename', 'Unknown')}\n"
            context_part += f"Content: {chunk.get('text', '')}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a structured prompt for the LLM."""
        prompt = f"""You are a medical AI assistant helping healthcare professionals answer questions based on medical documents.

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. Be accurate and precise in your medical terminology
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide specific references to the source material when possible
5. Use a professional, medical tone
6. If there are any contradictions in the context, mention them

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def _calculate_confidence(self, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on relevance scores of chunks."""
        if not relevant_chunks:
            return 0.0
        
        # Get relevance scores
        relevance_scores = [chunk.get('relevance_score', 0.0) for chunk in relevant_chunks]
        
        # Calculate weighted average (giving more weight to top results)
        weights = [1.0 / (i + 1) for i in range(len(relevance_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(relevance_scores, weights))
        weight_sum = sum(weights)
        
        confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, confidence))
    
    def _create_source_references(self, relevant_chunks: List[Dict[str, Any]]) -> List[SourceReference]:
        """Create source references from relevant chunks."""
        sources = []
        
        for chunk in relevant_chunks:
            # Extract page number from chunk metadata (if available)
            page_number = self._extract_page_number(chunk)
            
            # Create content preview (first 200 characters)
            content_preview = chunk.get('text', '')[:200]
            if len(chunk.get('text', '')) > 200:
                content_preview += "..."
            
            source = SourceReference(
                document_id=chunk.get('document_id', ''),
                filename=chunk.get('filename', 'Unknown'),
                page_number=page_number,
                chunk_id=chunk.get('chunk_id', ''),
                relevance_score=chunk.get('relevance_score', 0.0),
                content_preview=content_preview
            )
            sources.append(source)
        
        return sources
    
    def _extract_page_number(self, chunk: Dict[str, Any]) -> int:
        """Extract page number from chunk metadata."""
        # This is a simplified implementation
        # In a real scenario, you'd store page numbers with each chunk
        return 1  # Default page number
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            # Test with a simple prompt
            response = self.model.generate_content(
                "Say 'OK' if you can respond.",
                generation_config={"temperature": 0, "max_output_tokens": 10}
            )
            return response.text is not None
        except Exception:
            return False 