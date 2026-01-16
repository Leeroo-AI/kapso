# Phase 3: Enrichment Report

## Summary

- **Environment pages created:** 3
- **Heuristic pages created:** 4
- **Implementation pages updated:** 7
- **Index files updated:** 3 (_EnvironmentIndex, _HeuristicIndex, _ImplementationIndex)

## Environments Created

| Environment | Required By | Notes |
|-------------|-------------|-------|
| Alibaba_NLP_DeepResearch_API_Keys_Configuration | Search_call, Visit_call, PythonInterpreter_call, Call_Llm_Judge, WebSearch_call, Visit_call_multimodal | SERPER_KEY_ID, JINA_API_KEYS, API_KEY, DASHSCOPE_API_KEY |
| Alibaba_NLP_DeepResearch_Python_Dependencies | MultiTurnReactAgent__init__, MultiTurnReactAgent__run, count_tokens, OmniSearch_process_image, OmniSearch_run_main | transformers, openai, requests, json5, tiktoken, pillow, qwen-agent |
| Alibaba_NLP_DeepResearch_Sandbox_Execution_Environment | PythonInterpreter_call, run_code_in_sandbox | SandboxFusion endpoints for secure code execution |

## Heuristics Created

| Heuristic | Applies To | Notes |
|-----------|------------|-------|
| Alibaba_NLP_DeepResearch_Token_Limit_Management | MultiTurnReactAgent__run, count_tokens, Context_Management principle | 110K token limit triggers forced answer generation |
| Alibaba_NLP_DeepResearch_Image_Resizing_Constraints | OmniSearch_process_image, OmniSearch_run_main, Image_Processing principle | max_pixels=1024*28*28, min_pixels=256*28*28 for VLM |
| Alibaba_NLP_DeepResearch_Locale_Detection_Search | Search_call, WebSearch_call, Web_Search_Execution principle | Chinese char detection (U+4E00-U+9FFF) for locale |
| Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry | MultiTurnReactAgent__run, Visit_call, Call_Llm_Judge, ReAct_Loop_Execution principle | base * 2^attempt + jitter, max 30s |

## Links Added

### Environment Links (requires_env)
- Alibaba_NLP_DeepResearch_Search_call → API_Keys_Configuration
- Alibaba_NLP_DeepResearch_Visit_call → API_Keys_Configuration
- Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run → Python_Dependencies
- Alibaba_NLP_DeepResearch_PythonInterpreter_call → Sandbox_Execution_Environment, API_Keys_Configuration
- Alibaba_NLP_DeepResearch_count_tokens → Python_Dependencies
- Alibaba_NLP_DeepResearch_OmniSearch_process_image → Python_Dependencies

### Heuristic Links (uses_heuristic)
- Alibaba_NLP_DeepResearch_Search_call → Locale_Detection_Search
- Alibaba_NLP_DeepResearch_Visit_call → Exponential_Backoff_Retry
- Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run → Token_Limit_Management, Exponential_Backoff_Retry
- Alibaba_NLP_DeepResearch_count_tokens → Token_Limit_Management
- Alibaba_NLP_DeepResearch_OmniSearch_process_image → Image_Resizing_Constraints

## Source Code Evidence

### Environment Constraints Discovered

| Pattern | Source File | Lines |
|---------|-------------|-------|
| `SERPER_KEY=os.environ.get('SERPER_KEY_ID')` | tool_search.py | 15 |
| `JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")` | tool_visit.py | 20 |
| `SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')` | tool_python.py | 25 |
| `API_KEY= os.getenv("API_KEY","")` | evaluate_deepsearch_official.py | 22 |
| `DASHSCOPE_API_KEY` | .env.example | 66 |

### Heuristics Discovered

| Pattern | Source File | Lines |
|---------|-------------|-------|
| `max_tokens = 110 * 1024` | react_agent.py | 186 |
| `self.max_pixels = 1024 * 28 * 28; self.min_pixels = 256 * 28 * 28` | agent_eval.py | 138-139 |
| `contains_chinese_basic(query)` with Unicode check | tool_search.py | 39-47 |
| `sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)` | react_agent.py | 102 |
| `@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=4, max=60))` | rag_system.py | 38 |

## Files Created

### Environments (3 files)
```
environments/
├── Alibaba_NLP_DeepResearch_API_Keys_Configuration.md
├── Alibaba_NLP_DeepResearch_Python_Dependencies.md
└── Alibaba_NLP_DeepResearch_Sandbox_Execution_Environment.md
```

### Heuristics (4 files)
```
heuristics/
├── Alibaba_NLP_DeepResearch_Token_Limit_Management.md
├── Alibaba_NLP_DeepResearch_Image_Resizing_Constraints.md
├── Alibaba_NLP_DeepResearch_Locale_Detection_Search.md
└── Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry.md
```

## Index Files Updated

- `_EnvironmentIndex.md` - 3 entries added
- `_HeuristicIndex.md` - 4 entries added
- `_ImplementationIndex.md` - Updated with Env/Heuristic connections

## Notes for Audit Phase

### Verified Links
- All environment pages have backlinks to their dependent implementations
- All heuristic pages have backlinks to their using implementations/principles
- Implementation pages updated with forward links to environments and heuristics

### Potential Follow-ups
- WebSearch_call, WebSearch_call_multimodal could have explicit implementation page updates
- VLSearchImage_search_image_by_image_url may need API_Keys_Configuration link (uses SerpAPI)
- Additional heuristics could be documented:
  - Multi-service fallback pattern for Visit tool
  - 15-minute batch URL timeout
  - Token truncation (95K tokens) for webpage content

### External Dependencies Not Documented as Environment Pages
- vLLM server (runtime dependency, not package)
- SerpAPI service (external API, covered under API_Keys)
- Jina AI Reader (external API, covered under API_Keys)

---

**Generated:** 2026-01-15
**Phase:** 3 - Enrichment
**Repository:** Alibaba_NLP_DeepResearch
