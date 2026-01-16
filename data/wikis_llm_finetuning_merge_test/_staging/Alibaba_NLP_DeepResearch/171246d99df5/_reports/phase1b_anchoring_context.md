# Phase 1b: WorkflowIndex Enrichment Report

## Summary
- **Workflows enriched:** 3
- **Steps with detailed tables:** 23
- **Source files traced:** 16

## Enrichment Details

| Workflow | Steps Enriched | APIs Traced | Line Numbers Found |
|----------|----------------|-------------|-------------------|
| Alibaba_NLP_DeepResearch_ReAct_Web_Research | 7 | 9 | Yes |
| Alibaba_NLP_DeepResearch_Multimodal_VL_Search | 8 | 8 | Yes |
| Alibaba_NLP_DeepResearch_Benchmark_Evaluation | 8 | 9 | Yes |

## Implementation Types Found

| Type | Count | Examples |
|------|-------|----------|
| API Doc | 18 | `MultiTurnReactAgent._run`, `Search.call`, `Visit.call`, `VLSearchImage.search_image_by_image_url`, `call_llm_judge`, `single_round_statistics` |
| Wrapper Doc | 1 | `ThreadPoolExecutor` (for multi-rollout inference) |
| Pattern Doc | 6 | `os.getenv`, `json.load/loads`, string parsing for `<answer>` tags, template formatting |
| External Tool Doc | 0 | — |

## Source Files Traced

| File | Lines | APIs Extracted |
|------|-------|----------------|
| `inference/react_agent.py` | L47-55, L112-118, L120-226 | `__init__`, `count_tokens`, `_run` |
| `inference/run_multi_react.py` | L52-71, L174-225 | json parsing, ThreadPoolExecutor |
| `inference/tool_search.py` | L113-130 | `Search.call` |
| `inference/tool_visit.py` | L64-97 | `Visit.call` |
| `inference/tool_python.py` | L72-112 | `PythonInterpreter.call` |
| `inference/prompt.py` | L1-51 | SYSTEM_PROMPT, EXTRACTOR_PROMPT |
| `WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py` | L146-172, L182-401, L263-282 | `process_image`, `run_main`, answer extraction |
| `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/qwen_tool_call.py` | L21-49 | `Qwen_agent.__init__`, `_call_tool` |
| `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/generation.py` | L60-85 | `LLMGenerationManager.__init__` |
| `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_image.py` | L75-108 | `search_image_by_image_url` |
| `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/nlp_web_search.py` | L58-130 | `google_search`, `call` |
| `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/jialong_visit.py` | L68-90 | `Visit.call` |
| `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/sandbox_module.py` | L4-36 | `run_code_in_sandbox` |
| `evaluation/evaluate_deepsearch_official.py` | L76-144, L147-151, L209-325, L328-379, L382-415 | `call_llm_judge`, `process_single_round`, `single_round_statistics`, `calculate_enhanced_statistics`, `aggregate_results`, `calculate_pass_at_k` |
| `evaluation/evaluate_hle_official.py` | (referenced) | HLE-specific evaluation |
| `evaluation/prompt.py` | (referenced) | Judge prompts |

## Dependencies Summary

### External Libraries Used

| Library | Workflows | Purpose |
|---------|-----------|---------|
| `openai` | All 3 | LLM API client for chat completions |
| `requests` | All 3 | HTTP requests for tool APIs (Serper, Jina, SerpAPI) |
| `transformers` | All 3 | AutoTokenizer for token counting |
| `json5` | ReAct_Web_Research | Lenient JSON parsing for tool calls |
| `tiktoken` | ReAct, Benchmark | OpenAI tokenizer fallback |
| `qwen_agent` | ReAct, Multimodal | Agent base classes and tool registry |
| `serpapi` | Multimodal_VL_Search | Google reverse image search |
| `oss2` | Multimodal_VL_Search | Alibaba Cloud OSS for image upload |
| `litellm` | Benchmark_Evaluation | Multi-provider LLM client for judge |
| `PIL` | Multimodal_VL_Search | Image processing and resizing |
| `sandbox_fusion` | ReAct_Web_Research | Sandboxed Python code execution |
| `tqdm` | Benchmark | Progress bar for batch processing |
| `concurrent.futures` | All 3 | ThreadPoolExecutor for parallelization |

### Environment Variables Required

| Variable | Workflows | Purpose |
|----------|-----------|---------|
| `SERPER_KEY_ID` | ReAct, Multimodal | Google Serper API key for web search |
| `TEXT_SEARCH_KEY` | Multimodal | Alternative Serper key name |
| `IMG_SEARCH_KEY` | Multimodal | SerpAPI key for reverse image search |
| `JINA_API_KEYS` | ReAct | Jina Reader API key |
| `SANDBOX_FUSION_ENDPOINT` | ReAct, Multimodal | Python sandbox server URL |
| `API_KEY` | All 3 | LLM API key (various uses) |
| `API_BASE` | All 3 | LLM API base URL |
| `OPENAI_API_KEY` | Benchmark | OpenAI API key for judge |
| `OPENAI_API_BASE` | Benchmark | OpenAI API endpoint |
| `OSS_KEY_ID` | Multimodal | Alibaba Cloud OSS access key |
| `OSS_KEY_SECRET` | Multimodal | Alibaba Cloud OSS secret |
| `VLLM_MODEL` | Multimodal | Model name for vLLM inference |
| `IMAGE_DIR` | Multimodal | Directory for test images |

## Issues Found

1. **Missing import in sandbox_module.py** - The file references `requests` and `re` but imports are incomplete (line 18 references `requests.post` without visible import)
2. **Code duplication** - `Visit` tool implemented in multiple locations with slight variations:
   - `inference/tool_visit.py` (main)
   - `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/jialong_visit.py` (advanced)
   - `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/visit.py` (vLLM-specific)
3. **Hardcoded values** - Some defaults like `max_tokens = 110 * 1024` are hardcoded without environment variable override

## Ready for Repository Builder

- [x] All Step tables complete with 6 attributes each
- [x] All source locations verified with line numbers
- [x] Implementation Extraction Guides complete for each workflow
- [x] Global Implementation Extraction Guide added
- [x] No `<!-- ENRICHMENT NEEDED -->` comments remain

---

**Generated:** 2026-01-15
**Phase:** 1b - WorkflowIndex Enrichment
**Repository:** Alibaba_NLP_DeepResearch
