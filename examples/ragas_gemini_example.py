#!/usr/bin/env python3
"""
Simple example demonstrating RAGAS evaluation with Google Gemini.

This example shows how to:
1. Set up RAGAS with Google Gemini models
2. Evaluate RAG responses using various metrics
3. Interpret the results

Prerequisites:
    - Set GEMINI_API_KEY environment variable
    - Install: pip install ragas langchain-google-genai datasets

Usage:
    export GEMINI_API_KEY="your-google-ai-api-key"
    python examples/ragas_gemini_example.py
"""

import os
import json
from typing import List, Dict, Any

def check_setup():
    """Check if environment is properly set up."""
    print("🔍 Checking setup...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("   Please set it: export GEMINI_API_KEY='your-api-key'")
        print("   Get your key from: https://aistudio.google.com/")
        return False
    
    print(f"✅ API key found (length: {len(api_key)})")
    
    # Check dependencies
    required_packages = ["ragas", "langchain_google_genai", "datasets"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def setup_ragas_with_gemini():
    """Set up RAGAS metrics with Google Gemini models."""
    print("\n🔧 Setting up RAGAS with Google Gemini...")
    
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRelevancy
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    
    # Initialize Gemini models
    api_key = os.getenv("GEMINI_API_KEY")
    
    # LLM for evaluation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True
    )
    
    # Embeddings for similarity calculations
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # Initialize metrics
    metrics = [
        Faithfulness(),           # Factual consistency
        AnswerRelevancy(),        # How relevant is the answer
        ContextPrecision(),       # Quality of retrieved context
        ContextRelevancy()        # Relevance of context to question
    ]
    
    # Configure metrics with Gemini models
    for metric in metrics:
        if hasattr(metric, 'llm'):
            metric.llm = ragas_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_embeddings
    
    print("✅ RAGAS configured with Gemini models")
    return metrics

def evaluate_medical_examples(metrics):
    """Evaluate medical RAG examples."""
    print("\n🏥 Evaluating Medical RAG Examples...")
    
    from ragas import evaluate
    from datasets import Dataset
    
    # Medical examples
    examples = {
        "question": [
            "What are the symptoms of type 2 diabetes?",
            "How is hypertension diagnosed?",
            "What are the risk factors for heart disease?"
        ],
        "answer": [
            "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision. Some people may have no symptoms initially.",
            "Hypertension is diagnosed by measuring blood pressure on multiple occasions. A reading of 140/90 mmHg or higher on two separate occasions indicates high blood pressure.",
            "Risk factors for heart disease include high cholesterol, high blood pressure, smoking, diabetes, obesity, family history, and sedentary lifestyle."
        ],
        "contexts": [
            [
                "Type 2 diabetes mellitus symptoms include polydipsia (excessive thirst), polyuria (frequent urination), polyphagia (increased appetite), fatigue, and blurred vision.",
                "Many individuals with type 2 diabetes may be asymptomatic in early stages.",
                "Weight loss despite increased appetite can also occur in diabetes."
            ],
            [
                "Hypertension diagnosis requires blood pressure measurements ≥140/90 mmHg on two or more separate occasions.",
                "Blood pressure should be measured using proper technique with appropriate cuff size.",
                "Ambulatory blood pressure monitoring may be used to confirm diagnosis."
            ],
            [
                "Cardiovascular disease risk factors include modifiable factors like hypertension, dyslipidemia, smoking, diabetes mellitus, and obesity.",
                "Non-modifiable risk factors include age, gender, and family history of cardiovascular disease.",
                "Sedentary lifestyle and poor diet are additional modifiable risk factors."
            ]
        ]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(examples)
    
    # Run evaluation
    print("🔄 Running evaluation (this may take 30-60 seconds)...")
    result = evaluate(dataset, metrics=metrics)
    
    # Display results
    print("\n📊 Evaluation Results:")
    print("=" * 40)
    
    for metric_name, score in result.items():
        print(f"{metric_name}: {score:.3f}")
        
        # Interpretation
        if score >= 0.8:
            interpretation = "Excellent ✅"
        elif score >= 0.6:
            interpretation = "Good 👍"
        elif score >= 0.4:
            interpretation = "Fair ⚠️"
        else:
            interpretation = "Needs Improvement ❌"
        
        print(f"  → {interpretation}")
    
    return result

def interpret_results(results: Dict[str, float]):
    """Provide detailed interpretation of results."""
    print("\n💡 Result Interpretation:")
    print("=" * 40)
    
    interpretations = {
        "faithfulness": {
            "description": "Measures factual consistency between answer and context",
            "good_threshold": 0.8,
            "advice_low": "Answer contains information not supported by context (hallucination)",
            "advice_high": "Answer is factually consistent with provided context"
        },
        "answer_relevancy": {
            "description": "Measures how well the answer addresses the question",
            "good_threshold": 0.7,
            "advice_low": "Answer doesn't directly address the question asked",
            "advice_high": "Answer is highly relevant to the question"
        },
        "context_precision": {
            "description": "Measures quality of context ranking/ordering",
            "good_threshold": 0.7,
            "advice_low": "Irrelevant contexts ranked highly, affecting answer quality",
            "advice_high": "Most relevant contexts are ranked appropriately"
        },
        "context_relevancy": {
            "description": "Measures relevance of retrieved context to the question",
            "good_threshold": 0.7,
            "advice_low": "Retrieved contexts don't relate well to the question",
            "advice_high": "Retrieved contexts are highly relevant to the question"
        }
    }
    
    for metric_name, score in results.items():
        if metric_name in interpretations:
            info = interpretations[metric_name]
            print(f"\n📈 {metric_name.replace('_', ' ').title()}: {score:.3f}")
            print(f"   {info['description']}")
            
            if score >= info['good_threshold']:
                print(f"   ✅ {info['advice_high']}")
            else:
                print(f"   ⚠️  {info['advice_low']}")

def save_results(results: Dict[str, float]):
    """Save results to file."""
    output = {
        "timestamp": "2024-01-01T00:00:00Z",  # Would use datetime.utcnow().isoformat()
        "model": "gemini-2.0-flash-exp",
        "evaluation_framework": "RAGAS",
        "metrics": results,
        "summary": {
            "average_score": sum(results.values()) / len(results),
            "total_metrics": len(results),
            "scores_above_0_7": sum(1 for score in results.values() if score >= 0.7)
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: evaluation_results.json")

def main():
    """Main function."""
    print("🚀 RAGAS + Google Gemini: Medical RAG Evaluation")
    print("=" * 50)
    
    # Step 1: Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        return
    
    try:
        # Step 2: Configure RAGAS with Gemini
        metrics = setup_ragas_with_gemini()
        
        # Step 3: Run evaluation
        results = evaluate_medical_examples(metrics)
        
        # Step 4: Interpret results
        interpret_results(dict(results))
        
        # Step 5: Save results
        save_results(dict(results))
        
        print("\n🎉 Evaluation completed successfully!")
        print("\nNext steps:")
        print("1. Review the scores and interpretations above")
        print("2. Improve low-scoring areas (context retrieval, answer generation)")
        print("3. Run regular evaluations to monitor performance")
        print("4. Consider adding ground truth data for more comprehensive metrics")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your Google API key is valid")
        print("3. Ensure you have sufficient API quota")
        print("4. Try running the test again in a few minutes")

if __name__ == "__main__":
    main() 