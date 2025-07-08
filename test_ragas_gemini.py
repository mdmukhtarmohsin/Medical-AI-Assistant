#!/usr/bin/env python3
"""
Test script for RAGAS with Google Gemini integration.

This script validates the configuration and demonstrates usage of
RAGAS evaluation metrics with Google Gemini models instead of OpenAI.

Usage:
    python test_ragas_gemini.py

Environment Variables Required:
    GOOGLE_API_KEY: Your Google AI API key

Author: Medical AI Assistant Development Team
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

def test_environment():
    """Test environment setup and dependencies."""
    print("🔍 Testing Environment Setup...")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("   Please set it: export GOOGLE_API_KEY='your-api-key'")
        return False
    else:
        print(f"✅ GOOGLE_API_KEY found (length: {len(api_key)})")
    
    # Check dependencies
    dependencies = [
        "ragas",
        "langchain_google_genai", 
        "datasets",
        "pandas"
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n📦 Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def test_manual_ragas_config():
    """Test manual RAGAS configuration with Google Gemini."""
    print("\n🔧 Testing Manual RAGAS Configuration...")
    
    try:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRelevancy
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import evaluate
        from datasets import Dataset
        
        print("✅ All RAGAS imports successful")
        
        # Initialize Gemini models
        api_key = os.getenv("GOOGLE_API_KEY")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.0,
            convert_system_message_to_human=True
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        print("✅ Gemini models initialized")
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        print("✅ RAGAS wrappers created")
        
        # Initialize metrics
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRelevancy()
        ]
        
        # Configure each metric
        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = ragas_embeddings
        
        print("✅ Metrics configured successfully")
        
        # Test with sample data
        sample_data = {
            "question": ["What are the symptoms of diabetes?"],
            "answer": ["Common symptoms include increased thirst, frequent urination, and unexplained weight loss."],
            "contexts": [["Diabetes symptoms include polydipsia (excessive thirst), polyuria (frequent urination), polyphagia (excessive hunger), and unexplained weight loss."]],
        }
        
        dataset = Dataset.from_dict(sample_data)
        print("✅ Sample dataset created")
        
        # Run evaluation
        print("🔄 Running evaluation (this may take a moment)...")
        result = evaluate(dataset, metrics=metrics)
        
        print("✅ Manual RAGAS evaluation successful!")
        print(f"📊 Results: {dict(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual configuration failed: {str(e)}")
        return False

def test_custom_evaluator():
    """Test our custom Gemini RAGAS evaluator."""
    print("\n🎯 Testing Custom Gemini RAGAS Evaluator...")
    
    try:
        # Import custom evaluator
        from app.evaluation.custom_ragas_config import create_gemini_evaluator, GeminiRagasEvaluator
        
        print("✅ Custom evaluator imports successful")
        
        # Create evaluator
        evaluator = create_gemini_evaluator()
        print("✅ Evaluator created")
        
        # Validate configuration
        if evaluator.validate_configuration():
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        # Check available metrics
        metrics = evaluator.get_available_metrics()
        print(f"✅ Available metrics: {metrics}")
        
        # Test single evaluation
        print("🔄 Testing single evaluation...")
        
        question = "What are the main symptoms of hypertension?"
        answer = "High blood pressure often has no symptoms, but severe cases may cause headaches, shortness of breath, and nosebleeds."
        contexts = [
            "Hypertension is often called a 'silent killer' because it typically has no symptoms.",
            "Severe hypertension may cause headaches, difficulty breathing, and nosebleeds.",
            "Regular blood pressure monitoring is important for early detection."
        ]
        
        scores = evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts
        )
        
        print("✅ Single evaluation successful!")
        print(f"📊 Scores: {json.dumps(scores, indent=2)}")
        
        # Test batch evaluation
        print("🔄 Testing batch evaluation...")
        
        questions = [
            "What causes diabetes?",
            "How is diabetes diagnosed?",
            "What are the complications of diabetes?"
        ]
        
        answers = [
            "Diabetes is caused by insulin resistance or insufficient insulin production by the pancreas.",
            "Diabetes is diagnosed through blood tests measuring glucose levels, including fasting glucose and HbA1c tests.",
            "Diabetes complications include cardiovascular disease, kidney damage, nerve damage, and eye problems."
        ]
        
        contexts = [
            ["Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells. Type 2 diabetes results from insulin resistance and relative insulin deficiency."],
            ["Diagnostic criteria include fasting plasma glucose ≥126 mg/dL, random glucose ≥200 mg/dL with symptoms, or HbA1c ≥6.5%."],
            ["Long-term diabetes complications affect multiple systems: cardiovascular (heart disease, stroke), renal (nephropathy), neurological (neuropathy), and ocular (retinopathy)."]
        ]
        
        batch_scores = evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts=contexts
        )
        
        print("✅ Batch evaluation successful!")
        print(f"📊 Batch Scores: {json.dumps(batch_scores, indent=2)}")
        
        # Test with ground truth
        print("🔄 Testing evaluation with ground truth...")
        
        ground_truths = [
            "Diabetes mellitus is caused by defects in insulin production, insulin action, or both.",
            "Diabetes diagnosis is based on plasma glucose criteria: fasting ≥126 mg/dL or HbA1c ≥6.5%.",
            "Diabetes complications include macrovascular (cardiovascular) and microvascular (retinopathy, nephropathy, neuropathy) diseases."
        ]
        
        enhanced_scores = evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
        
        print("✅ Enhanced evaluation with ground truth successful!")
        print(f"📊 Enhanced Scores: {json.dumps(enhanced_scores, indent=2)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error (expected if custom evaluator not available): {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Custom evaluator test failed: {str(e)}")
        return False

async def test_evaluation_service():
    """Test the evaluation service integration."""
    print("\n🌐 Testing Evaluation Service Integration...")
    
    try:
        from app.services.evaluation_service import evaluation_service
        
        print("✅ Evaluation service imported")
        
        # Check if service is available
        if evaluation_service.is_available():
            print("✅ Evaluation service is available")
            
            # Test single response evaluation
            print("🔄 Testing single response evaluation...")
            
            metrics = await evaluation_service.evaluate_single_response(
                question="What is the normal range for blood pressure?",
                generated_answer="Normal blood pressure is typically less than 120/80 mmHg.",
                context=["Blood pressure categories: Normal <120/80, Elevated 120-129/<80, Stage 1 HTN 130-139/80-89, Stage 2 HTN ≥140/90."]
            )
            
            print("✅ Single response evaluation successful!")
            print(f"📊 Metrics: {metrics}")
            
            # Test batch evaluation
            print("🔄 Testing batch evaluation...")
            
            batch_metrics = await evaluation_service.evaluate_rag_responses(
                questions=["What is cholesterol?", "What causes high cholesterol?"],
                generated_answers=[
                    "Cholesterol is a waxy substance found in blood that's needed for building cells.",
                    "High cholesterol can be caused by diet, genetics, lifestyle factors, and certain medical conditions."
                ],
                contexts=[
                    ["Cholesterol is a lipid molecule essential for cell membrane structure and hormone synthesis."],
                    ["Cholesterol levels are influenced by dietary intake, genetic factors, physical activity, and underlying health conditions."]
                ]
            )
            
            print("✅ Batch evaluation successful!")
            print(f"📊 Batch Metrics: {batch_metrics}")
            
            return True
        else:
            print("❌ Evaluation service is not available")
            print("   This might be due to missing API key or dependencies")
            return False
            
    except ImportError as e:
        print(f"❌ Import error (expected if service not available): {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Evaluation service test failed: {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoints for evaluation."""
    print("\n🔗 Testing API Endpoints...")
    
    try:
        import requests
        
        # Test evaluation endpoint
        base_url = "http://localhost:8000"
        
        test_data = {
            "questions": ["What is asthma?"],
            "generated_answers": ["Asthma is a chronic respiratory condition characterized by inflammation and narrowing of airways."],
            "contexts": [["Asthma is a chronic inflammatory disorder of the airways characterized by variable airflow obstruction and airway hyperresponsiveness."]],
            "ground_truths": ["Asthma is a chronic inflammatory disease of the airways."]
        }
        
        response = requests.post(
            f"{base_url}/evaluation/evaluate",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ API endpoint test successful!")
            result = response.json()
            print(f"📊 API Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ API endpoint returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API (server might not be running)")
        print("   Start server with: uvicorn app.main:app --reload")
        return False
    except ImportError:
        print("❌ requests library not available")
        print("   Install with: pip install requests")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

def generate_test_report(results: Dict[str, bool]):
    """Generate a comprehensive test report."""
    print("\n📋 Test Report")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("-" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Generate recommendations
    print("\n💡 Recommendations:")
    
    if not results.get("Environment Setup", False):
        print("1. Set up your Google API key: export GOOGLE_API_KEY='your-key'")
        print("2. Install missing dependencies: pip install ragas langchain-google-genai datasets")
    
    if not results.get("Manual RAGAS Config", False):
        print("3. Check internet connection and API key permissions")
    
    if not results.get("Custom Evaluator", False):
        print("4. Ensure custom evaluator code is properly implemented")
    
    if not results.get("Evaluation Service", False):
        print("5. Check service configuration and dependencies")
    
    if not results.get("API Endpoints", False):
        print("6. Start the FastAPI server: uvicorn app.main:app --reload")
    
    # Save report to file
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "summary": {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests/total_tests)*100
        }
    }
    
    with open("test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📁 Detailed report saved to: test_report.json")

async def main():
    """Main test runner."""
    print("🚀 RAGAS with Google Gemini Integration Test Suite")
    print("=" * 60)
    
    # Store test results
    results = {}
    
    # Test 1: Environment Setup
    results["Environment Setup"] = test_environment()
    
    if not results["Environment Setup"]:
        print("\n⚠️  Environment setup failed. Please fix the issues before continuing.")
        generate_test_report(results)
        return
    
    # Test 2: Manual RAGAS Configuration
    results["Manual RAGAS Config"] = test_manual_ragas_config()
    
    # Test 3: Custom Evaluator
    results["Custom Evaluator"] = test_custom_evaluator()
    
    # Test 4: Evaluation Service
    results["Evaluation Service"] = await test_evaluation_service()
    
    # Test 5: API Endpoints
    results["API Endpoints"] = test_api_endpoints()
    
    # Generate final report
    generate_test_report(results)
    
    # Success message
    if all(results.values()):
        print("\n🎉 All tests passed! Your RAGAS-Gemini integration is working correctly.")
        print("   You can now use Google Gemini for RAG evaluation instead of OpenAI.")
    else:
        print("\n⚠️  Some tests failed. Please review the recommendations above.")

if __name__ == "__main__":
    asyncio.run(main()) 