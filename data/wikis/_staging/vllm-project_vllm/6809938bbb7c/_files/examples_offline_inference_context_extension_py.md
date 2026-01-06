# File: `examples/offline_inference/context_extension.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 68 |
| Functions | `create_llm`, `run_llm_chat`, `print_outputs`, `main` |
| Imports | vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates extending model context length beyond pretrained limits using RoPE scaling techniques.

**Mechanism:** Uses LLM with extended max_model_len (32768 tokens) and rope_scaling configuration (type="dynamic", factor=2.0) to process longer sequences than the model's original context window. Tests with a 22K+ token prompt to show context extension in action.

**Significance:** Shows how to use RoPE (Rotary Position Embedding) scaling to handle longer contexts than models were trained for. Essential for applications requiring extended context windows like long document processing.
