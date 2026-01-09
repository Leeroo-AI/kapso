# File: `unsloth/dataprep/synthetic.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 465 |
| Classes | `PipeCapture`, `SyntheticDataKit` |
| Functions | `terminate_tree` |
| Imports | collections, gc, numpy, os, re, requests, subprocess, synthetic_configs, threading, time, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generate synthetic Q&A training data from raw text using vLLM inference server

**Mechanism:** SyntheticDataKit spawns vLLM server subprocess with optimized parameters (GPU memory utilization, float8 KV cache, compilation), uses PipeCapture for non-blocking output monitoring with regex-based readiness detection, provides chunk_data() to split documents by token limits with overlap, prepare_qa_generation() configures YAML-based pipeline for Q&A extraction with cleanup and rating stages

**Significance:** Enables automated creation of instruction-following datasets from unstructured text, leveraging vLLM for fast inference to generate question-answer pairs at scale. Critical for users who need to create training data from domain-specific documents without manual annotation, integrating with Unsloth's training pipeline for end-to-end dataset preparation
