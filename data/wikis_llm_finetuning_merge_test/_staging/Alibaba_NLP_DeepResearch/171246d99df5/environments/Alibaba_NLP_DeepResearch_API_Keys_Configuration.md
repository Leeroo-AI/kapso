# Environment: API_Keys_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|.env.example|https://github.com/Alibaba-NLP/DeepResearch/blob/main/.env.example]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Web_Research]], [[domain::LLM_Agents]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Environment configuration for external API services required by DeepResearch web agents and evaluation pipelines.

=== Description ===
DeepResearch requires multiple external API keys for web search, webpage reading, file parsing, and Python code execution. These keys enable the agent to interact with web services (Serper, Jina), document parsing services (Dashscope), and sandboxed code execution (SandboxFusion). The environment is designed for distributed multi-worker setups with configurable NCCL settings for multi-GPU inference.

=== Usage ===
Use this environment configuration when running any DeepResearch inference pipeline, including the ReAct agent, multimodal OmniSearch, or benchmark evaluation workflows. All web search and webpage visitation tools require these API keys to function.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Tested on Ubuntu 20.04 LTS
|-
| Hardware || NVIDIA GPU (Optional) || Required for local vLLM inference
|-
| Network || Internet access || Required for external API calls
|}

== Dependencies ==

=== System Packages ===
* `python` >= 3.9
* `curl` (for API testing)

=== Python Packages ===
* `requests` (HTTP client for API calls)
* `openai` >= 1.0.0 (OpenAI-compatible API client)
* `http.client` (standard library for Serper API)

== Credentials ==
The following environment variables must be set in `.env`:

* `SERPER_KEY_ID`: Serper.dev API key for Google Search and Google Scholar queries
* `JINA_API_KEYS`: Jina.ai API key for webpage content extraction via r.jina.ai
* `API_KEY`: OpenAI-compatible API key for page summarization LLM
* `API_BASE`: Base URL for OpenAI-compatible summarization API
* `SUMMARY_MODEL_NAME`: Model name for page summarization (e.g., gpt-4o-mini)
* `DASHSCOPE_API_KEY`: Alibaba Dashscope API key for document parsing (PDF, Office files)
* `SANDBOX_FUSION_ENDPOINT`: Endpoint URL(s) for SandboxFusion code execution service
* `OPENAI_API_KEY`: (Optional) OpenAI API key for LLM judge evaluation
* `OPENAI_API_BASE`: (Optional) OpenAI API base URL for evaluation

== Quick Install ==
<syntaxhighlight lang="bash">
# Copy the example environment file
cp .env.example .env

# Edit .env and fill in your API keys:
# - SERPER_KEY_ID: Get from https://serper.dev/
# - JINA_API_KEYS: Get from https://jina.ai/api-dashboard/
# - API_KEY/API_BASE: Your OpenAI-compatible API credentials
# - DASHSCOPE_API_KEY: Get from https://dashscope.aliyun.com/
# - SANDBOX_FUSION_ENDPOINT: Your SandboxFusion deployment URL

# Load environment variables
source .env
</syntaxhighlight>

== Code Evidence ==

Environment variable loading from `tool_search.py:15`:
<syntaxhighlight lang="python">
SERPER_KEY=os.environ.get('SERPER_KEY_ID')
</syntaxhighlight>

Multiple API keys loaded in `tool_visit.py:17-21`:
<syntaxhighlight lang="python">
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
</syntaxhighlight>

Sandbox endpoint requirement from `tool_python.py:25`:
<syntaxhighlight lang="python">
SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')
</syntaxhighlight>

LLM API configuration from `evaluate_deepsearch_official.py:20-23`:
<syntaxhighlight lang="python">
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
os.environ['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE","")
API_KEY= os.getenv("API_KEY","")
BASE_URL=os.getenv("BASE_URL","")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Google search Timeout, return None` || SERPER_KEY_ID not set or invalid || Set valid SERPER_KEY_ID in environment
|-
|| `[visit] Failed to read page.` || JINA_API_KEYS not set or invalid || Set valid JINA_API_KEYS in environment
|-
|| `KeyError: 'SANDBOX_FUSION_ENDPOINT'` || Sandbox endpoint not configured || Set SANDBOX_FUSION_ENDPOINT with valid SandboxFusion URL
|-
|| `vllm server error!!!` || LLM API unreachable or misconfigured || Verify API_KEY and API_BASE settings
|}

== Compatibility Notes ==

* **Multiple Sandbox Endpoints:** SANDBOX_FUSION_ENDPOINT supports comma-separated URLs for load balancing
* **API Key Rotation:** JINA_API_KEYS can contain multiple keys for rate limit management
* **OpenAI Compatibility:** API_KEY/API_BASE work with any OpenAI-compatible endpoint (vLLM, OpenRouter, etc.)
* **Dashscope Models:** Supports qwen-omni-turbo, qwen-plus-latest for document parsing

== Related Pages ==
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_Search_call]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_Visit_call]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_PythonInterpreter_call]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_Call_Llm_Judge]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_WebSearch_call]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_Visit_call_multimodal]]
