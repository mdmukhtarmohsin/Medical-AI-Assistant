#!/usr/bin/env python3
"""
Test script for RAGAS evaluation in the Medical AI Assistant.
This script demonstrates how to test your Medical AI Assistant using RAGAS metrics.
"""

import asyncio
import json
import requests
import time
from typing import List, Dict, Any
import argparse


class RAGASEvaluationTester:
    """Test class for RAGAS evaluation functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the tester with API base URL."""
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test if the API is responding."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API is healthy and responding")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {str(e)}")
            return False
    
    def upload_test_document(self, file_path: str) -> str:
        """
        Upload a test document and return document ID.
        
        Args:
            file_path: Path to the PDF file to upload
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = self.session.post(f"{self.base_url}/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    document_id = result.get('document_id')
                    print(f"✅ Document uploaded successfully: {document_id}")
                    return document_id
                else:
                    print(f"❌ Document upload failed: {response.status_code}")
                    print(f"Error: {response.text}")
                    return None
        except Exception as e:
            print(f"❌ Error uploading document: {str(e)}")
            return None
    
    def ask_question(self, document_id: str, question: str) -> Dict[str, Any]:
        """
        Ask a question about a document.
        
        Args:
            document_id: ID of the uploaded document
            question: Question to ask
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            payload = {
                "question": question,
                "document_id": document_id,
                "include_sources": True,
                "temperature": 0.1
            }
            
            response = self.session.post(f"{self.base_url}/ask", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Question answered successfully")
                return result
            else:
                print(f"❌ Question failed: {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error asking question: {str(e)}")
            return None
    
    def evaluate_single_response(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A response using RAGAS.
        
        Args:
            question: The question
            answer: The generated answer
            contexts: List of context chunks
            ground_truth: Optional ground truth answer
            
        Returns:
            Evaluation metrics
        """
        try:
            payload = {
                "question": question,
                "answer": answer,
                "contexts": contexts
            }
            
            if ground_truth:
                payload["ground_truth"] = ground_truth
            
            response = self.session.post(f"{self.base_url}/evaluate/single", data=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Single evaluation completed")
                return result
            else:
                print(f"❌ Single evaluation failed: {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error in single evaluation: {str(e)}")
            return None
    
    def evaluate_document_responses(
        self, 
        document_id: str, 
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate responses for a document using RAGAS.
        
        Args:
            document_id: ID of the document
            questions: List of questions to evaluate
            
        Returns:
            Batch evaluation report
        """
        try:
            # Use form data for list of questions
            data = {"questions": questions}
            
            response = self.session.post(
                f"{self.base_url}/evaluate/document/{document_id}", 
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document evaluation completed")
                return result
            else:
                print(f"❌ Document evaluation failed: {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error in document evaluation: {str(e)}")
            return None
    
    def run_comprehensive_test(self, pdf_file_path: str = None):
        """
        Run a comprehensive RAGAS evaluation test.
        
        Args:
            pdf_file_path: Path to a PDF file for testing (optional)
        """
        print("🚀 Starting RAGAS Evaluation Test")
        print("=" * 50)
        
        # 1. Health check
        print("\n1. Testing API Health...")
        if not self.test_health_check():
            print("❌ Cannot proceed without healthy API")
            return
        
        # 2. Document upload (if file provided)
        document_id = None
        if pdf_file_path:
            print(f"\n2. Uploading test document: {pdf_file_path}")
            document_id = self.upload_test_document(pdf_file_path)
        
        # 3. Test single evaluation (with dummy data)
        print("\n3. Testing Single Response Evaluation...")
        test_question = "What are the side effects of acetaminophen?"
        test_answer = "Common side effects include nausea, vomiting, and stomach pain. Serious side effects may include liver damage."
        test_contexts = [
            "Acetaminophen may cause side effects such as nausea and vomiting.",
            "Overdose of acetaminophen can lead to serious liver damage.",
            "Common gastrointestinal side effects include stomach pain and discomfort."
        ]
        
        metrics = self.evaluate_single_response(
            question=test_question,
            answer=test_answer,
            contexts=test_contexts
        )
        
        if metrics:
            print("📊 RAGAS Metrics:")
            print(f"   - Faithfulness: {metrics.get('faithfulness', 'N/A'):.3f}")
            print(f"   - Answer Relevancy: {metrics.get('answer_relevancy', 'N/A'):.3f}")
            print(f"   - Context Relevancy: {metrics.get('context_relevancy', 'N/A'):.3f}")
            print(f"   - Context Recall: {metrics.get('context_recall', 'N/A'):.3f}")
            print(f"   - Overall Score: {metrics.get('overall_score', 'N/A'):.3f}")
        
        # 4. Test document evaluation (if document uploaded)
        if document_id:
            print(f"\n4. Testing Document Evaluation for {document_id}...")
            test_questions = [
                "What are the main side effects mentioned?",
                "What is the recommended dosage?",
                "Are there any contraindications?",
                "What should patients do in case of overdose?"
            ]
            
            report = self.evaluate_document_responses(document_id, test_questions)
            
            if report:
                print("📋 Evaluation Report Summary:")
                print(f"   - Total Questions: {report.get('total_evaluations', 'N/A')}")
                print(f"   - Successful: {report.get('successful_evaluations', 'N/A')}")
                print(f"   - Failed: {report.get('failed_evaluations', 'N/A')}")
                print(f"   - Threshold Pass Rate: {report.get('threshold_pass_rate', 'N/A'):.1%}")
                
                avg_metrics = report.get('average_metrics', {})
                print("   📊 Average Metrics:")
                print(f"     - Faithfulness: {avg_metrics.get('faithfulness', 'N/A'):.3f}")
                print(f"     - Answer Relevancy: {avg_metrics.get('answer_relevancy', 'N/A'):.3f}")
                print(f"     - Context Relevancy: {avg_metrics.get('context_relevancy', 'N/A'):.3f}")
                print(f"     - Context Recall: {avg_metrics.get('context_recall', 'N/A'):.3f}")
                print(f"     - Overall Score: {avg_metrics.get('overall_score', 'N/A'):.3f}")
        
        print("\n" + "=" * 50)
        print("🎉 RAGAS Evaluation Test Completed!")


def create_sample_questions_file():
    """Create a sample questions file for batch testing."""
    sample_questions = {
        "medical_questions": [
            "What are the common side effects of this medication?",
            "What is the recommended dosage for adults?",
            "Are there any contraindications or warnings?",
            "What should patients do if they miss a dose?",
            "Can this medication be taken with food?",
            "What are the signs of an allergic reaction?",
            "Is this medication safe during pregnancy?",
            "What are the storage requirements?",
            "How long does it take for the medication to work?",
            "What should be done in case of overdose?"
        ]
    }
    
    with open('sample_evaluation_questions.json', 'w') as f:
        json.dump(sample_questions, f, indent=2)
    
    print("✅ Created sample_evaluation_questions.json")
    print("You can use this file to test batch evaluation with your documents")


def main():
    """Main function for running RAGAS evaluation tests."""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Tester for Medical AI Assistant")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                       help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--pdf", type=str, help="Path to PDF file for testing")
    parser.add_argument("--create-sample", action="store_true", 
                       help="Create sample questions file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_questions_file()
        return
    
    # Initialize tester
    tester = RAGASEvaluationTester(base_url=args.url)
    
    # Run comprehensive test
    tester.run_comprehensive_test(pdf_file_path=args.pdf)


if __name__ == "__main__":
    main() 