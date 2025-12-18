# File: `examples/offline_inference/structured_outputs.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 113 |
| Classes | `CarType`, `CarDescription` |
| Functions | `format_output`, `generate_output`, `main` |
| Imports | enum, pydantic, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates constrained generation using guided_decoding to force outputs conforming to JSON schemas.

**Mechanism:** Defines Pydantic models (CarDescription with nested CarType enum) and uses guided_json parameter in SamplingParams to constrain generation. vLLM enforces JSON schema during decoding, guaranteeing parseable structured outputs. Shows both generate() and chat() APIs with guided decoding.

**Significance:** Essential for applications requiring structured data extraction or API responses. Ensures model outputs are always valid JSON matching specified schemas, eliminating parsing errors and validation failures.
