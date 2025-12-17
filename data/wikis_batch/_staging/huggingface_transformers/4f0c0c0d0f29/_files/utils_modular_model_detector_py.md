# File: `utils/modular_model_detector.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 913 |
| Classes | `CodeSimilarityAnalyzer` |
| Functions | `build_date_data`, `main` |
| Imports | argparse, ast, datetime, functools, huggingface_hub, json, logging, numpy, os, pathlib, ... +5 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects code similarity between model implementations to identify modularization opportunities using semantic embeddings.

**Mechanism:** `CodeSimilarityAnalyzer` builds an index by parsing all `modeling_*.py` files, extracting class/function definitions, sanitizing model-specific names (e.g., replacing "Llama" with "Model"), and encoding with Qwen3-Embedding-4B. For queries, computes cosine similarity between embeddings and optionally Jaccard similarity on token sets. Results include embedding scores, Jaccard scores, and intersection. Index stored in Hub dataset with `embeddings.safetensors`, `code_index_map.json`, and `code_index_tokens.json`. Includes release date correlation via `build_date_data()`.

**Significance:** Strategic tool for library architecture evolution, helping maintainers identify similar implementations that can be refactored into modular patterns. Accelerates modularization efforts by automatically finding the best parent model for new implementations, potentially saving significant development and maintenance time.
