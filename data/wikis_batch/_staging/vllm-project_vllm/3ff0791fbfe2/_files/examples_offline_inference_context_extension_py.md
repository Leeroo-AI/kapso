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

**Purpose:** Demonstrates context length extension using YARN.

**Mechanism:** Uses hf_overrides with rope_parameters to configure YARN (Yet Another RoPE extensioN) method for Qwen model, setting rope_theta, rope_type="yarn", scaling factor, and extended max_model_len (4x original_max_position_embeddings). Runs simple chat example to demonstrate extended context capability.

**Significance:** Example showing how to extend model context length beyond original training limits using RoPE scaling techniques.
