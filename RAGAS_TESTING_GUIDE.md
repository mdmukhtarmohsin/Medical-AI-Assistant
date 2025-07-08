# 🧪 RAGAS Testing Guide for Medical AI Assistant

This guide explains how to test your Medical AI Assistant using RAGAS (RAG Assessment) metrics to evaluate the quality of your AI responses.

## 📋 What is RAGAS?

RAGAS evaluates RAG (Retrieval-Augmented Generation) systems using four key metrics:

1. **Faithfulness** (0-1): How factually accurate is the answer based on the given context?
2. **Answer Relevancy** (0-1): How relevant is the answer to the question?
3. **Context Precision** (0-1): How precise and relevant are the retrieved contexts?
4. **Context Recall** (0-1): How well does the context cover the ground truth answer?

## 🚀 Quick Start

### 1. Ensure Your Application is Running

First, make sure your Medical AI Assistant is running:

```bash
# Start the application using the provided script
./start.sh

# Or manually start it
source venv/bin/activate
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run Basic RAGAS Test

```bash
# Run the test script without uploading a document (uses dummy data)
python3 test_ragas_evaluation.py

# Or with a specific PDF document
python3 test_ragas_evaluation.py --pdf /path/to/your/medical_document.pdf

# Test against a different URL
python3 test_ragas_evaluation.py --url http://localhost:8001
```

## 📊 Testing Methods

### Method 1: Single Response Evaluation

Test individual question-answer pairs:

```bash
curl -X POST "http://localhost:8000/evaluate/single" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What are the side effects of acetaminophen?" \
  -d "answer=Common side effects include nausea and liver damage." \
  -d "contexts=Acetaminophen can cause nausea" \
  -d "contexts=Overdose may lead to liver damage"
```

**Python Example:**

```python
import requests

data = {
    "question": "What are the side effects?",
    "answer": "Side effects include nausea and headache.",
    "contexts": ["Nausea is a common side effect", "Headaches may occur"]
}

response = requests.post("http://localhost:8000/evaluate/single", data=data)
metrics = response.json()
print(f"Faithfulness: {metrics['faithfulness']:.3f}")
```

### Method 2: Document-Based Evaluation

Upload a document and evaluate multiple questions:

```bash
# First upload a document
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@medical_document.pdf"

# Use the returned document_id for evaluation
curl -X POST "http://localhost:8000/evaluate/document/YOUR_DOCUMENT_ID" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "questions=What are the side effects?" \
  -d "questions=What is the dosage?" \
  -d "questions=Are there contraindications?"
```

### Method 3: Batch Evaluation with Ground Truth

For more advanced testing with known correct answers:

```python
import requests

evaluation_data = {
    "questions": [
        "What are the side effects?",
        "What is the dosage?"
    ],
    "answers": [
        "Side effects include nausea and dizziness.",
        "The recommended dosage is 500mg twice daily."
    ],
    "contexts_list": [
        ["Nausea is common", "Dizziness may occur"],
        ["Standard dose is 500mg", "Take twice per day"]
    ],
    "ground_truth_answers": [
        "Common side effects are nausea and dizziness.",
        "Take 500mg two times daily."
    ]
}

response = requests.post("http://localhost:8000/evaluate", json=evaluation_data)
results = response.json()
```

## 🎯 Interpreting RAGAS Scores

### Score Ranges and Quality Assessment

| Metric                | Excellent (0.8-1.0)                | Good (0.6-0.8)  | Fair (0.4-0.6)          | Poor (<0.4)         |
| --------------------- | ---------------------------------- | --------------- | ----------------------- | ------------------- |
| **Faithfulness**      | Highly accurate, no hallucinations | Mostly accurate | Some inaccuracies       | Many hallucinations |
| **Answer Relevancy**  | Directly answers question          | Mostly relevant | Partially relevant      | Off-topic           |
| **Context Precision** | Perfect context retrieval          | Good context    | Some irrelevant context | Poor context        |
| **Context Recall**    | Complete coverage                  | Good coverage   | Partial coverage        | Poor coverage       |

### Quality Thresholds (Configurable in .env)

```env
RAGAS_FAITHFULNESS_THRESHOLD=0.90
RAGAS_CONTEXT_PRECISION_THRESHOLD=0.85
```

Your system will flag responses that fall below these thresholds as potentially unreliable.

## 🔧 Advanced Testing Scenarios

### Scenario 1: Test Different Question Types

```python
test_questions = {
    "factual": "What is the active ingredient in this medication?",
    "dosage": "How much should I take and how often?",
    "safety": "What are the contraindications?",
    "procedural": "How should this medication be stored?",
    "comparative": "How does this compare to other treatments?"
}
```

### Scenario 2: Test with Medical Documents

Upload different types of medical documents:

- Drug information leaflets
- Clinical trial reports
- Medical research papers
- Patient information guides

### Scenario 3: Ground Truth Validation

Create a test dataset with known correct answers:

```json
{
  "test_cases": [
    {
      "question": "What is the maximum daily dose?",
      "ground_truth": "The maximum daily dose is 4000mg",
      "document_id": "drug_info_123"
    }
  ]
}
```

## 📈 Performance Monitoring

### Setting Up Continuous Evaluation

1. **Create a test suite** with representative questions
2. **Run periodic evaluations** to monitor quality
3. **Track metrics over time** to detect degradation
4. **Set up alerts** for threshold violations

### Example Monitoring Script

```python
import schedule
import time

def run_daily_evaluation():
    """Run daily RAGAS evaluation on production data."""
    tester = RAGASEvaluationTester()
    results = tester.evaluate_document_responses(
        document_id="production_doc",
        questions=STANDARD_TEST_QUESTIONS
    )

    # Log results and alert if quality drops
    if results['threshold_pass_rate'] < 0.8:
        send_alert(f"Quality degradation detected: {results['threshold_pass_rate']:.1%}")

# Schedule daily evaluation
schedule.every().day.at("02:00").do(run_daily_evaluation)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Low Faithfulness Scores**

   - Check if your LLM is hallucinating
   - Verify context quality and relevance
   - Adjust retrieval parameters

2. **Low Answer Relevancy**

   - Review question preprocessing
   - Check if answers are too generic
   - Improve prompt engineering

3. **Low Context Precision**

   - Tune vector similarity thresholds
   - Improve document chunking strategy
   - Review embedding model performance

4. **Low Context Recall**
   - Increase number of retrieved chunks
   - Improve document coverage
   - Check if ground truth is in the documents

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed RAGAS evaluation steps
```

## 📝 Best Practices

1. **Regular Testing**: Run RAGAS evaluation regularly, not just during development
2. **Diverse Questions**: Test with various question types and complexities
3. **Ground Truth**: When possible, use verified medical information as ground truth
4. **Threshold Tuning**: Adjust quality thresholds based on your specific use case
5. **Documentation**: Keep track of evaluation results and improvements

## 🎓 Example Test Session

Here's a complete example of testing your Medical AI Assistant:

```bash
# 1. Start the application
./start.sh

# 2. Create sample questions
python3 test_ragas_evaluation.py --create-sample

# 3. Run comprehensive test with a medical PDF
python3 test_ragas_evaluation.py --pdf medical_drug_info.pdf

# 4. Check the results and iterate on improvements
```

Expected output:

```
🚀 Starting RAGAS Evaluation Test
==================================================

1. Testing API Health...
✅ API is healthy and responding

2. Uploading test document: medical_drug_info.pdf
✅ Document uploaded successfully: abc123-def456

3. Testing Single Response Evaluation...
✅ Single evaluation completed
📊 RAGAS Metrics:
   - Faithfulness: 0.892
   - Answer Relevancy: 0.856
   - Context Relevancy: 0.734
   - Context Recall: 0.823
   - Overall Score: 0.826

4. Testing Document Evaluation for abc123-def456...
✅ Document evaluation completed
📋 Evaluation Report Summary:
   - Total Questions: 4
   - Successful: 4
   - Failed: 0
   - Threshold Pass Rate: 75.0%
   📊 Average Metrics:
     - Faithfulness: 0.887
     - Answer Relevancy: 0.831
     - Context Relevancy: 0.798
     - Context Recall: 0.756
     - Overall Score: 0.818

==================================================
🎉 RAGAS Evaluation Test Completed!
```

This comprehensive evaluation gives you insights into your system's performance and helps identify areas for improvement.

Happy testing! 🧪✨
