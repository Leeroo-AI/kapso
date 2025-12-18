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

**Purpose:** Detects code similarities between model implementations to identify modularization opportunities.

**Mechanism:** Builds a searchable index by extracting class and function definitions from all modeling files, sanitizes code by replacing model-specific names with generic placeholders, generates embeddings using Qwen3-Embedding-4B transformer model, and computes both embedding-based cosine similarity and Jaccard token overlap for robust similarity detection. Provides CLI for analyzing new model files against the index with release date context.

**Significance:** Development tool for modular architecture expansion that helps identify which existing model classes can serve as good parent classes for new models. Accelerates modular model development by suggesting the most similar existing implementations to inherit from, considering both semantic similarity (embeddings) and structural similarity (token overlap). Supports the goal of converting more models to modular format.
