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

**Purpose:** Provides the `SyntheticDataKit` class for generating synthetic training data (e.g., QA pairs) using a vLLM-served language model.

**Mechanism:** The class launches a vLLM inference server as a subprocess using configurable parameters (model name, max sequence length, GPU memory utilization, etc.). It patches vLLM via `unsloth_zoo.vllm_utils` and monitors server startup using the `PipeCapture` class which performs non-blocking pipe reading with regex-based readiness detection. Once the server is ready (detected via stdout regex matching "Starting vLLM API server"), it can be used for synthetic data generation. The `prepare_qa_generation` method creates a YAML configuration file from the `synthetic_qa_config` template with parameters for QA generation. The `chunk_data` method splits large text files into token-bounded chunks with overlap for processing. Context manager support (`__enter__`/`__exit__`) ensures proper cleanup including subprocess termination and CUDA memory clearing.

**Significance:** Enables automated synthetic dataset creation by leveraging LLM inference, which is valuable for data augmentation and creating training datasets from raw documents. The subprocess-based vLLM integration allows efficient batch generation while maintaining isolation from the training process.
