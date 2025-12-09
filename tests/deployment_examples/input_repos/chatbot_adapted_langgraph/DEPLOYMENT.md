# LangGraph Chatbot - Deployment Guide

## ✅ Deployment Files Created

All necessary files for LangGraph deployment have been created:

- **main.py** - Main entry point with `predict()` function
- **agent.py** - LangGraph agent with compiled graph
- **langgraph.json** - LangGraph configuration file
- **requirements.txt** - All dependencies including LangGraph SDK
- **.env** - Environment variables template

## 🔧 Local Testing

The chatbot works locally and can be tested:

```bash
# Test the predict function
python -c "from main import predict; print(predict({'text': 'Hello!'}))"

# Test via main.py
echo '{"text": "Hello!"}' | python main.py
```

## 🚀 Deployment Options

### Option 1: LangGraph Cloud (Recommended for Production)

LangGraph Cloud requires:
- LangSmith API key with LangGraph Cloud access, OR
- Production license key (LANGGRAPH_CLOUD_LICENSE_KEY)

To deploy to LangGraph Cloud:

1. Ensure you have a LangSmith account with LangGraph Cloud access
2. Set environment variables in `.env`:
   ```
   LANGSMITH_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   ```
3. Run deployment command:
   ```bash
   langgraph up --port 8123
   ```

### Option 2: Local Development Server

Run locally for testing without cloud access:

```bash
# Option A: Using LangGraph CLI (if license permits)
langgraph up --port 8123

# Option B: Direct Python import
python main.py
```

### Option 3: Custom HTTP Server (No License Required)

Create a simple FastAPI/Flask wrapper around the `predict()` function:

```python
from fastapi import FastAPI
from main import predict

app = FastAPI()

@app.post("/predict")
def endpoint(inputs: dict):
    return predict(inputs)
```

Then deploy using any standard Python hosting service (Heroku, AWS, etc.).

## 📦 Run Interface

The chatbot is accessible via the `predict()` function in `main.py`:

```python
from main import predict

result = predict({
    "text": "Hello! What's your name?"
})
# Returns: {"status": "success", "output": {...}}
```

## 🔑 Environment Variables

Required environment variables:
- `ANTHROPIC_API_KEY` - Your Anthropic API key for Claude
- `LANGSMITH_API_KEY` - Your LangSmith API key (for cloud deployment)

## 📝 Notes

- The agent uses Claude Sonnet 4 (claude-sonnet-4-20250514)
- Conversation memory is handled automatically by LangGraph's state management
- All deployment files follow LangGraph Platform best practices
