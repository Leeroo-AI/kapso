# QA Model - Modal Deployment

Question answering model using transformers, deployed on Modal.com with GPU support.

## Files

- `qa_model.py` - Core QA model logic
- `main.py` - Entry point with predict() function
- `modal_app.py` - Modal deployment configuration
- `requirements.txt` - Python dependencies

## Local Setup

```bash
pip install -r requirements.txt
```

## Local Testing

Test the predict function:

```bash
echo '{"question": "What is AI?", "context": "Artificial Intelligence is the simulation of human intelligence."}' | python main.py
```

Or run directly:

```bash
python main.py
```

## Modal Deployment

### Prerequisites

```bash
pip install modal
modal token new  # First time only
```

### Test Locally on Modal

```bash
modal run modal_app.py
```

### Deploy to Modal

```bash
modal deploy modal_app.py
```

This deploys the QA model with:
- T4 GPU
- 16GB memory
- Auto-scaling
- HTTP endpoint

### Call the Deployed Function

#### Via Python SDK

```python
import modal

predict = modal.Function.lookup("qa-model", "predict")
result = predict.remote({
    "question": "What is the capital of France?",
    "context": "Paris is the capital and most populous city of France."
})
print(result)
```

#### Via HTTP Endpoint

After deployment, you'll get a URL. Call it with:

```bash
curl -X POST "https://[your-username]--qa-model-web-predict.modal.run" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is AI?", "context": "Artificial Intelligence is..."}'
```

## Input Format

```json
{
  "question": "Your question here",
  "context": "The context containing the answer"
}
```

## Output Format

Success:
```json
{
  "status": "success",
  "output": {
    "answer": "extracted answer",
    "score": 0.9876
  }
}
```

Error:
```json
{
  "status": "error",
  "error": "error message"
}
```
