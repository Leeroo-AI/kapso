# LangGraph Chatbot

A conversational AI chatbot built with LangGraph and Claude, featuring persistent memory and stateful conversations.

## Features

- Stateful conversation management using LangGraph
- Claude Sonnet 4 integration for intelligent responses
- Automatic message history accumulation
- Ready for deployment to LangGraph Platform

## Structure

```
.
├── agent.py           # LangGraph agent definition with compiled graph
├── main.py            # Standardized entry point with predict() function
├── langgraph.json     # LangGraph Platform configuration
├── deploy.py          # Deployment script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Local Setup

### Prerequisites

- Python 3.8+
- Anthropic API key

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY='your-api-key'
```

### Local Testing

Test the chatbot using the main.py entry point:

```bash
# Interactive test
python main.py

# Test with stdin
echo '{"text": "Hello!"}' | python main.py

# Test from Python
python -c "from main import predict; print(predict({'text': 'Hello!'}))"
```

## Deployment to LangGraph Platform

### Prerequisites

1. Install LangGraph CLI:
   ```bash
   pip install langgraph-cli
   ```

2. Get your LangSmith API key from https://smith.langchain.com/

3. Set environment variable:
   ```bash
   export LANGSMITH_API_KEY='your-langsmith-api-key'
   ```

### Deploy

```bash
# Using the deployment script
python deploy.py

# Or directly with CLI
langgraph deploy
```

### Using the Deployed Agent

After deployment, interact with your agent using the LangGraph SDK:

```python
from langgraph_sdk import get_client

# Connect to your deployment
client = get_client(url="YOUR_DEPLOYMENT_URL")

# Create a conversation thread
thread = await client.threads.create()

# Send a message
result = await client.runs.create(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]}
)

print(result)
```

### Streaming Responses

```python
async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Tell me a story"}]}
):
    print(event)
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for Claude API access (local and deployed)
- `LANGSMITH_API_KEY` - Required for LangGraph Platform deployment
- `LANGGRAPH_API_URL` - Optional, defaults to https://api.smith.langchain.com

## API

### predict(inputs)

Main prediction function for the chatbot.

**Arguments:**
- `inputs` (dict | str): Input data with one of:
  - `{"messages": [...]}` - List of message objects
  - `{"text": "..."}` - Simple text input
  - `{"content": "..."}` - Content string
  - Or a plain string

**Returns:**
- `dict`: Response with status and output
  ```python
  {
      "status": "success",
      "output": {
          "messages": [...]
      }
  }
  ```

## Architecture

The chatbot uses a simple LangGraph state graph:

1. **AgentState**: Tracks message history with automatic accumulation
2. **chatbot_node**: Processes messages using Claude Sonnet 4
3. **Graph Flow**: START → chatbot → END

The `add_messages` annotation ensures conversation context is preserved across turns.

## LangGraph Platform Features

When deployed to LangGraph Platform, you get:

- **Persistent Threads**: Conversations maintain state across requests
- **Streaming**: Token-by-token response streaming
- **Checkpointing**: Automatic state snapshots for recovery
- **Human-in-the-loop**: Pause execution for human input
- **Cron Jobs**: Schedule agent execution
- **Scalability**: Auto-scaling based on demand

## Troubleshooting

**Import Error:**
```bash
pip install -r requirements.txt
```

**API Key Error:**
```bash
export ANTHROPIC_API_KEY='your-key'
export LANGSMITH_API_KEY='your-key'
```

**Deployment Failed:**
- Verify `langgraph.json` is valid
- Check that `agent.py:graph` exports a compiled graph
- Ensure LangSmith API key is set

## License

This project is provided as-is for deployment demonstration purposes.
