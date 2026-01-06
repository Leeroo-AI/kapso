# Claude Code with AWS Bedrock

Run Claude Code via AWS Bedrock instead of direct Anthropic API.

---

## Setup

### 1. Install CLI
```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Set Credentials (one of)

```bash
# Option A: Bedrock bearer token (simplest)
export AWS_BEARER_TOKEN_BEDROCK="ABSKQmVkcm9ja..."

# Option B: IAM access keys
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."

# Option C: SSO profile
aws sso login --profile my-profile
export AWS_PROFILE="my-profile"
```

### 3. Set Region
```bash
export AWS_REGION="us-east-1"
```

---

## Model IDs

| Direct API | Bedrock |
|------------|---------|
| `claude-opus-4-5` | `us.anthropic.claude-opus-4-5-20251101-v1:0` |

---

## Usage

```python
from src.execution.coding_agents.base import CodingAgentConfig
from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent

config = CodingAgentConfig(
    agent_type="claude_code",
    model="us.anthropic.claude-opus-4-5-20251101-v1:0",
    debug_model="us.anthropic.claude-opus-4-5-20251101-v1:0",
    agent_specific={
        "use_bedrock": True,
        "aws_region": "us-east-1",
    }
)

agent = ClaudeCodeCodingAgent(config)
agent.initialize("/path/to/workspace")
result = agent.generate_code("Create hello.py that prints 'Hello!'")
```

---

## Test

```bash
# Set credentials, then:
python tests/test_claude_code_agent_bedrock.py
```

Tests: CLI connects to Bedrock, creates a file, verifies success.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| No AWS credentials | Set `AWS_BEARER_TOKEN_BEDROCK` or access keys |
| AWS_REGION not set | `export AWS_REGION="us-east-1"` |
| Model not found | Enable model in AWS Bedrock console, check IAM permissions |
