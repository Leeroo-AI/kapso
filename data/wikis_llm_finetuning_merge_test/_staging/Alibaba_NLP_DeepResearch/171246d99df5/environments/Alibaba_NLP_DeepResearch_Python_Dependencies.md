# Environment: Python_Dependencies

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|README|https://github.com/Alibaba-NLP/DeepResearch/blob/main/README.md]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::LLM_Agents]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Python package dependencies required for running DeepResearch inference and evaluation pipelines.

=== Description ===
DeepResearch is built on the qwen-agent framework and requires multiple Python packages for LLM inference, web scraping, document parsing, and benchmark evaluation. The stack includes transformers for tokenization, openai for API clients, requests for HTTP operations, and various specialized libraries for multimodal processing.

=== Usage ===
Install these dependencies before running any DeepResearch workflow. The packages are required for the ReAct agent, web search tools, webpage visitation, file parsing, code execution, and benchmark evaluation.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Python || >= 3.9 || Python 3.10+ recommended
|-
| OS || Linux (Ubuntu 20.04+) || macOS supported for development
|-
| Hardware || 16GB RAM minimum || More for large model inference
|}

== Dependencies ==

=== System Packages ===
* `python3-dev` (Python development headers)
* `ffmpeg` (for video/audio processing)
* `git-lfs` (for model downloads)

=== Python Packages ===
* `transformers` >= 4.37.0 (AutoTokenizer, AutoProcessor)
* `openai` >= 1.0.0 (OpenAI-compatible API client)
* `requests` >= 2.28.0 (HTTP requests)
* `json5` (relaxed JSON parsing)
* `tiktoken` (token counting for context management)
* `pillow` (PIL for image processing)
* `qwen-agent` (Qwen agent framework)
* `litellm` (multi-provider LLM client for evaluation)
* `serpapi` (SerpAPI client for reverse image search)
* `oss2` (Alibaba Cloud OSS for image upload)
* `dashscope` (Dashscope API client)
* `vllm` (optional, for local model serving)
* `gradio` (optional, for demo UI)
* `streamlit` (optional, for WebWalker demo)
* `tenacity` (retry with exponential backoff)

== Credentials ==
No credentials required for package installation. See [[Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]] for API credentials.

== Quick Install ==
<syntaxhighlight lang="bash">
# Core dependencies
pip install transformers>=4.37.0 openai>=1.0.0 requests json5 tiktoken pillow

# Agent framework
pip install qwen-agent

# Evaluation dependencies
pip install litellm serpapi tenacity

# Optional: for multimodal processing
pip install oss2 dashscope

# Optional: for local inference
pip install vllm

# Optional: for demo UIs
pip install gradio streamlit
</syntaxhighlight>

== Code Evidence ==

Transformers import for tokenization from `react_agent.py:8`:
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
</syntaxhighlight>

OpenAI client usage from `react_agent.py:7`:
<syntaxhighlight lang="python">
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
</syntaxhighlight>

Tiktoken for token counting from `tool_visit.py:24-32`:
<syntaxhighlight lang="python">
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
</syntaxhighlight>

JSON5 for relaxed parsing from `react_agent.py:2`:
<syntaxhighlight lang="python">
import json5
</syntaxhighlight>

Tenacity for retry logic from `rag_system.py:7`:
<syntaxhighlight lang="python">
from tenacity import retry, stop_after_attempt, wait_exponential
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'transformers'` || transformers not installed || `pip install transformers>=4.37.0`
|-
|| `ImportError: cannot import name 'OpenAI' from 'openai'` || Old openai package version || `pip install openai>=1.0.0`
|-
|| `ModuleNotFoundError: No module named 'qwen_agent'` || qwen-agent not installed || `pip install qwen-agent`
|-
|| `ModuleNotFoundError: No module named 'tiktoken'` || tiktoken not installed || `pip install tiktoken`
|}

== Compatibility Notes ==

* **transformers >= 4.37.0:** Required for latest AutoProcessor features used in multimodal agent
* **openai >= 1.0.0:** Uses new client API; not compatible with openai < 1.0
* **vllm:** Optional but recommended for local high-performance inference
* **qwen-agent:** Core dependency that provides BaseTool, FnCallAgent base classes

== Related Pages ==
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__init__]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_process_image]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_run_main]]
