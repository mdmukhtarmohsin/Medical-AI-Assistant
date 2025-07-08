# RAGAS with Google Gemini Tutorial

This tutorial explains how to configure and use RAGAS (RAG Assessment) with Google Gemini models instead of OpenAI for evaluating your RAG (Retrieval-Augmented Generation) pipeline.

## Why Use RAGAS with Google Gemini?

### Benefits:

- **Cost-effective**: Google Gemini API pricing is competitive
- **Powerful evaluation**: Gemini 2.0 Flash provides excellent evaluation capabilities
- **No OpenAI dependency**: Complete independence from OpenAI services
- **Multimodal support**: Gemini supports both text and image evaluation
- **Better integration**: Native Google ecosystem integration

### RAGAS Metrics Available:

1. **Faithfulness**: Factual consistency with retrieved context
2. **Answer Relevancy**: How well the answer addresses the question
3. **Context Precision**: Quality of retrieved context ranking
4. **Context Relevancy**: Relevance of context to the question
5. **Context Recall**: Coverage of relevant information (requires ground truth)
6. **Answer Similarity**: Semantic similarity to ground truth (requires ground truth)
7. **Answer Correctness**: Overall factual accuracy (requires ground truth)

## Prerequisites

### 1. Install Dependencies

```bash
pip install ragas>=0.1.0 langchain-google-genai>=1.0.0 datasets>=2.0.0
```

### 2. Get Google AI API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your-google-ai-api-key"
```

## Configuration

### Option 1: Using Our Custom Evaluator

The Medical AI Assistant includes a pre-configured Gemini RAGAS evaluator:

```python
from app.evaluation.custom_ragas_config import create_gemini_evaluator

# Create evaluator
evaluator = create_gemini_evaluator()

# Validate configuration
if evaluator.validate_configuration():
    print("✅ Gemini RAGAS evaluator is ready!")
else:
    print("❌ Configuration validation failed")
```

### Option 2: Manual Configuration

For custom setups, you can configure RAGAS manually:

```python
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

# Initialize Gemini models
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key="your-api-key",
    temperature=0.0,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="your-api-key"
)

# Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

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
```

## Usage Examples

### Single Response Evaluation

```python
from app.evaluation.custom_ragas_config import create_gemini_evaluator

# Create evaluator
evaluator = create_gemini_evaluator()

# Evaluate single response
question = "What are the symptoms of diabetes?"
answer = "Common symptoms include increased thirst, frequent urination, and fatigue."
contexts = [
    "Diabetes symptoms include polydipsia (excessive thirst), polyuria (frequent urination), and fatigue.",
    "Type 2 diabetes often presents with gradual onset of symptoms."
]

scores = evaluator.evaluate_single(
    question=question,
    answer=answer,
    contexts=contexts
)

print(f"Evaluation scores: {scores}")
```

### Batch Evaluation

```python
# Prepare batch data
questions = [
    "What causes diabetes?",
    "How is diabetes treated?",
    "What are diabetes complications?"
]

answers = [
    "Diabetes is caused by insulin resistance or insufficient insulin production.",
    "Treatment includes lifestyle changes, medication, and blood sugar monitoring.",
    "Complications include cardiovascular disease, kidney damage, and nerve damage."
]

contexts = [
    ["Diabetes results from the body's inability to properly use or produce insulin."],
    ["Management involves diet, exercise, medication, and regular monitoring."],
    ["Long-term complications affect multiple organ systems."]
]

# Run batch evaluation
scores = evaluator.evaluate_batch(
    questions=questions,
    answers=answers,
    contexts=contexts
)

print(f"Batch evaluation scores: {scores}")
```

### With Ground Truth (Enhanced Metrics)

```python
# Include ground truth for more comprehensive evaluation
ground_truths = [
    "Diabetes is caused by insulin resistance or beta cell dysfunction.",
    "Treatment involves lifestyle modifications, medications, and monitoring.",
    "Complications include retinopathy, nephropathy, and neuropathy."
]

scores = evaluator.evaluate_batch(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)

# This will include additional metrics like ContextRecall, AnswerSimilarity, etc.
print(f"Enhanced evaluation scores: {scores}")
```

## Integration with Your Application

### Using the Evaluation Service

```python
from app.services.evaluation_service import evaluation_service

# Check if service is available
if evaluation_service.is_available():
    # Evaluate single response
    metrics = await evaluation_service.evaluate_single_response(
        question="What is hypertension?",
        generated_answer="Hypertension is high blood pressure.",
        context=["High blood pressure, or hypertension, is a common condition."]
    )

    print(f"Faithfulness: {metrics.faithfulness}")
    print(f"Answer Relevancy: {metrics.answer_relevancy}")
    print(f"Overall Score: {metrics.overall_score}")
```

### API Integration

The Medical AI Assistant includes API endpoints for evaluation:

```bash
# Test evaluation endpoint
curl -X POST "http://localhost:8000/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is diabetes?"],
    "generated_answers": ["Diabetes is a metabolic disorder."],
    "contexts": [["Diabetes affects blood sugar regulation."]],
    "ground_truths": ["Diabetes is a chronic metabolic condition."]
  }'
```

## Interpreting Results

### Score Ranges

- **0.0 - 0.4**: Poor quality, significant improvement needed
- **0.4 - 0.6**: Fair quality, some improvements recommended
- **0.6 - 0.8**: Good quality, minor optimizations possible
- **0.8 - 1.0**: Excellent quality, well-performing system

### Metric-Specific Interpretation

**Faithfulness (0.0 - 1.0)**

- Measures factual consistency with context
- Low scores indicate hallucination or contradiction
- Target: > 0.8 for reliable systems

**Answer Relevancy (0.0 - 1.0)**

- Measures how well answer addresses question
- Low scores indicate off-topic responses
- Target: > 0.7 for good user experience

**Context Relevancy (0.0 - 1.0)**

- Measures relevance of retrieved context
- Low scores indicate poor retrieval
- Target: > 0.7 for effective RAG

## Troubleshooting

### Common Issues

1. **API Key Error**

```
ValueError: Gemini API key is required
```

**Solution**: Set `GEMINI_API_KEY` environment variable

2. **Import Errors**

```
ImportError: RAGAS dependencies not available
```

**Solution**: Install required packages:

```bash
pip install ragas langchain-google-genai datasets
```

3. **Zero Scores**

```
All metrics return 0.0
```

**Solution**: Check API key, internet connection, and input data format

4. **Rate Limiting**

```
429 Too Many Requests
```

**Solution**: Implement retry logic with exponential backoff

### Debugging Tips

1. **Validate Configuration**

```python
if evaluator.validate_configuration():
    print("✅ Configuration valid")
else:
    print("❌ Configuration invalid")
```

2. **Check Available Metrics**

```python
print(f"Available metrics: {evaluator.get_available_metrics()}")
```

3. **Test with Simple Examples**

```python
# Start with basic example
scores = evaluator.evaluate_single(
    question="What is 2+2?",
    answer="2+2 equals 4",
    contexts=["Basic arithmetic: 2+2=4"]
)
```

## Best Practices

### 1. Data Quality

- Ensure contexts are relevant and informative
- Use clear, well-formed questions
- Provide accurate ground truth when available

### 2. Batch Processing

- Process in batches of 10-50 for efficiency
- Implement retry logic for failed evaluations
- Monitor API usage and costs

### 3. Metric Selection

- Use reference-free metrics for production evaluation
- Include ground truth metrics for development/testing
- Focus on metrics most relevant to your use case

### 4. Performance Optimization

- Cache evaluation results when possible
- Use async processing for large batches
- Monitor evaluation latency and costs

## Advanced Configuration

### Custom Models

```python
evaluator = GeminiRagasEvaluator(
    gemini_api_key="your-key",
    llm_model="gemini-1.5-pro",  # Use different model
    embedding_model="models/text-embedding-004",  # Use different embeddings
    temperature=0.1  # Adjust creativity
)
```

### Metric Customization

```python
# Use specific metrics only
reference_free_metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextRelevancy()
]

# Configure evaluator with custom metrics
for metric in reference_free_metrics:
    evaluator._configure_metric(metric)
```

## Monitoring and Analytics

### Tracking Evaluation Results

```python
import json
from datetime import datetime

# Log evaluation results
result = {
    "timestamp": datetime.utcnow().isoformat(),
    "question": question,
    "scores": scores,
    "model_used": "gemini-2.0-flash-exp"
}

# Save to file or database
with open("evaluation_log.jsonl", "a") as f:
    f.write(json.dumps(result) + "\n")
```

### Performance Metrics

- Track average scores over time
- Monitor evaluation latency
- Analyze score distributions
- Identify low-performing question types

## Cost Optimization

### Strategies

1. **Selective Evaluation**: Evaluate samples, not all responses
2. **Metric Selection**: Use fewer metrics for routine evaluation
3. **Batch Processing**: Combine evaluations for efficiency
4. **Caching**: Store results for repeated evaluations

### Cost Monitoring

```python
# Track API calls and estimate costs
evaluation_count = 0
estimated_cost = 0.0

def track_evaluation(num_questions):
    global evaluation_count, estimated_cost
    evaluation_count += num_questions
    # Estimate based on Gemini pricing
    estimated_cost += num_questions * 0.001  # Example rate

    print(f"Total evaluations: {evaluation_count}")
    print(f"Estimated cost: ${estimated_cost:.3f}")
```

## Conclusion

Using RAGAS with Google Gemini provides a powerful, cost-effective solution for evaluating RAG systems. The combination offers:

- **Comprehensive evaluation metrics**
- **Cost-effective pricing**
- **Easy integration**
- **Reliable performance**

Start with basic single evaluations, then scale to batch processing as your needs grow. Monitor results and optimize based on your specific requirements.

For more advanced use cases, consider implementing custom metrics or integrating with MLOps platforms for continuous evaluation and monitoring.
