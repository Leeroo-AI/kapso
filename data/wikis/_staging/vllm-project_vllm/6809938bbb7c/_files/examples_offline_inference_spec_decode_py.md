# File: `examples/offline_inference/spec_decode.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 234 |
| Functions | `get_custom_mm_prompts`, `parse_args`, `main` |
| Imports | transformers, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive example of speculative decoding using draft models to speed up generation latency.

**Mechanism:** Configures LLM with speculative_config containing draft_model parameter. Draft model generates candidate tokens quickly, which target model verifies in parallel. Supports various draft model types (smaller models, n-gram, MLPSpeculator) and handles both text-only and multimodal inputs. Compares latency with and without speculation.

**Significance:** Critical optimization technique for latency-sensitive applications. Shows how speculative decoding can significantly reduce time-to-first-token and overall generation latency by parallelizing verification of draft predictions.
