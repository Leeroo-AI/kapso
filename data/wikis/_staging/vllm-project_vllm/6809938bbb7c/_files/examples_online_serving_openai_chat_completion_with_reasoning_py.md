# File: `examples/online_serving/openai_chat_completion_with_reasoning.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 65 |
| Functions | `main` |
| Imports | openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Basic reasoning model usage example (DeepSeek-R1, etc.)

**Mechanism:** Shows non-streaming chat completion with reasoning-capable models. Requires --reasoning-parser flag (deepseek_r1, qwen3, etc.). Response includes both 'reasoning' and 'content' fields - reasoning contains the model's thinking process, content has the final answer. Demonstrates multi-turn conversations where reasoning is preserved between turns.

**Significance:** Essential starting point for using reasoning models with vLLM. Shows how to access the model's internal thought process (chain-of-thought) through the reasoning field. Critical for interpretable AI applications, education, debugging model behavior, and building transparent systems. Demonstrates vLLM's support for the emerging reasoning model paradigm.
