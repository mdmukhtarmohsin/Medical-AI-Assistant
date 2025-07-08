#!/usr/bin/env python3
"""
Batch evaluation script for Medical AI Assistant using RAGAS
"""

import os
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from .utils import Config, setup_logging
from .query import QueryService
from .models import QueryRequest, EvaluationResult

# Setup logging
logger = setup_logging()


class BatchEvaluator:
    """Batch evaluation using RAGAS metrics"""
    
    def __init__(self):
        self.query_service = QueryService()
        self.results = []
    
    async def evaluate_document(self, document_id: str, questions: List[str]) -> List[EvaluationResult]:
        """Evaluate a document with multiple questions"""
        results = []
        
        logger.info(f"Starting evaluation for document {document_id} with {len(questions)} questions")
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)}")
                
                # Create query request
                query_request = QueryRequest(
                    document_id=document_id,
                    question=question
                )
                
                # Get answer and metrics
                response = await self.query_service.answer_question(query_request)
                
                # Check if thresholds are met
                passed_thresholds = (
                    response.ragas_metrics.faithfulness >= Config.RAGAS_FAITHFULNESS_THRESHOLD and
                    response.ragas_metrics.context_precision >= Config.RAGAS_CONTEXT_PRECISION_THRESHOLD
                )
                
                # Create evaluation result
                result = EvaluationResult(
                    document_id=document_id,
                    question=question,
                    answer=response.answer,
                    ragas_metrics=response.ragas_metrics,
                    passed_thresholds=passed_thresholds
                )
                
                results.append(result)
                
                logger.info(f"Question {i} processed. Passed thresholds: {passed_thresholds}")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                continue
        
        return results
    
    def load_evaluation_questions(self, questions_file: str) -> Dict[str, List[str]]:
        """Load questions from JSON file"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded questions from {questions_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading questions file: {str(e)}")
            raise
    
    async def run_evaluation(self, questions_file: str, output_file: str = None):
        """Run complete evaluation"""
        try:
            # Load questions
            questions_data = self.load_evaluation_questions(questions_file)
            
            all_results = []
            
            # Process each document
            for document_id, questions in questions_data.items():
                logger.info(f"Evaluating document: {document_id}")
                
                doc_results = await self.evaluate_document(document_id, questions)
                all_results.extend(doc_results)
            
            # Save results
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"evaluation_results_{timestamp}.csv"
            
            self.save_results_to_csv(all_results, output_file)
            
            # Print summary
            self.print_evaluation_summary(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_results_to_csv(self, results: List[EvaluationResult], output_file: str):
        """Save evaluation results to CSV"""
        try:
            data = []
            
            for result in results:
                data.append({
                    'document_id': result.document_id,
                    'question': result.question,
                    'answer': result.answer,
                    'faithfulness': result.ragas_metrics.faithfulness,
                    'context_precision': result.ragas_metrics.context_precision,
                    'context_recall': result.ragas_metrics.context_recall,
                    'answer_relevancy': result.ragas_metrics.answer_relevancy,
                    'passed_thresholds': result.passed_thresholds
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def print_evaluation_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        if not results:
            logger.warning("No results to summarize")
            return
        
        total_questions = len(results)
        passed_count = sum(1 for r in results if r.passed_thresholds)
        
        # Calculate average metrics
        avg_faithfulness = sum(r.ragas_metrics.faithfulness for r in results) / total_questions
        avg_context_precision = sum(r.ragas_metrics.context_precision for r in results) / total_questions
        avg_context_recall = sum(r.ragas_metrics.context_recall for r in results) / total_questions
        avg_answer_relevancy = sum(r.ragas_metrics.answer_relevancy for r in results) / total_questions
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {total_questions}")
        print(f"Passed Thresholds: {passed_count} ({passed_count/total_questions*100:.1f}%)")
        print(f"Failed Thresholds: {total_questions - passed_count} ({(total_questions-passed_count)/total_questions*100:.1f}%)")
        print("\nAverage Metrics:")
        print(f"  Faithfulness: {avg_faithfulness:.3f}")
        print(f"  Context Precision: {avg_context_precision:.3f}")
        print(f"  Context Recall: {avg_context_recall:.3f}")
        print(f"  Answer Relevancy: {avg_answer_relevancy:.3f}")
        print("\nThresholds:")
        print(f"  Faithfulness >= {Config.RAGAS_FAITHFULNESS_THRESHOLD}")
        print(f"  Context Precision >= {Config.RAGAS_CONTEXT_PRECISION_THRESHOLD}")
        print("="*60)


def create_sample_questions_file():
    """Create a sample questions file for testing"""
    sample_questions = {
        "example_doc_id": [
            "What are the side effects mentioned in this document?",
            "What is the recommended dosage?",
            "Are there any contraindications listed?",
            "What is the mechanism of action described?",
            "What are the clinical trial results mentioned?"
        ]
    }
    
    with open("sample_questions.json", "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, indent=2, ensure_ascii=False)
    
    print("Sample questions file created: sample_questions.json")
    print("Update the document_id and questions before running evaluation.")


async def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluation for Medical AI Assistant")
    parser.add_argument("--questions", type=str, help="Path to questions JSON file")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--create-sample", action="store_true", help="Create sample questions file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_questions_file()
        return
    
    if not args.questions:
        print("Please provide a questions file with --questions or create a sample with --create-sample")
        return
    
    evaluator = BatchEvaluator()
    await evaluator.run_evaluation(args.questions, args.output)


if __name__ == "__main__":
    asyncio.run(main()) 