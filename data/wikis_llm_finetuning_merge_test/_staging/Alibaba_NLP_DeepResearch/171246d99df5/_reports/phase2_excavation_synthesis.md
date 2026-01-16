# Phase 2: Excavation + Synthesis Report

## Summary

- **Implementation pages created:** 20
- **Principle pages created:** 20
- **1:1 mappings verified:** 20
- **Pattern Doc implementations:** 2

## Principle-Implementation Pairs

### ReAct_Web_Research Workflow (7 pairs)

| Principle | Implementation | Source | Type |
|-----------|----------------|--------|------|
| Agent_Initialization | MultiTurnReactAgent__init__ | react_agent.py:L47-55 | API Doc |
| ReAct_Loop_Execution | MultiTurnReactAgent__run | react_agent.py:L120-226 | API Doc |
| Web_Search_Execution | Search_call | tool_search.py:L113-130 | API Doc |
| Webpage_Visitation | Visit_call | tool_visit.py:L64-97 | API Doc |
| Sandboxed_Code_Execution | PythonInterpreter_call | tool_python.py:L72-112 | API Doc |
| Context_Management | count_tokens | react_agent.py:L112-118 | API Doc |
| Answer_Extraction | Answer_Extraction_Pattern | react_agent.py:L211-226 | Pattern Doc |

### Multimodal_VL_Search Workflow (7 pairs)

| Principle | Implementation | Source | Type |
|-----------|----------------|--------|------|
| Image_Processing | OmniSearch_process_image | agent_eval.py:L146-172 | API Doc |
| Multi_Turn_Agent_Loop | OmniSearch_run_main | agent_eval.py:L182-401 | API Doc |
| Reverse_Image_Search | VLSearchImage_search_image_by_image_url | vl_search_image.py:L75-108 | API Doc |
| Text_Web_Search | WebSearch_call | nlp_web_search.py:L116-130 | API Doc |
| Webpage_Visitation_Multimodal | Visit_call_multimodal | jialong_visit.py:L68-90 | API Doc |
| Sandbox_Code_Execution_Multimodal | run_code_in_sandbox | sandbox_module.py:L4-36 | API Doc |
| Prompt_Construction | Prompt_Construction_Pattern | agent_eval.py:L30-128 | Pattern Doc |

### Benchmark_Evaluation Workflow (6 pairs)

| Principle | Implementation | Source | Type |
|-----------|----------------|--------|------|
| LLM_Judge_Scoring | Call_Llm_Judge | evaluate_deepsearch_official.py:L76-144 | API Doc |
| Result_Collection | Process_Single_Round | evaluate_deepsearch_official.py:L147-151 | API Doc |
| Result_Aggregation | Aggregate_Results | evaluate_deepsearch_official.py:L382-402 | API Doc |
| Pass_At_K_Metrics | Calculate_Pass_At_K | evaluate_deepsearch_official.py:L405-415 | API Doc |
| Behavioral_Statistics | Single_Round_Statistics | evaluate_deepsearch_official.py:L209-325 | API Doc |
| Enhanced_Statistics | Calculate_Enhanced_Statistics | evaluate_deepsearch_official.py:L328-379 | API Doc |

## Implementation Types

| Type | Count | Examples |
|------|-------|----------|
| API Doc | 18 | MultiTurnReactAgent__run, Search_call, Visit_call, VLSearchImage_search_image_by_image_url, call_llm_judge |
| Pattern Doc | 2 | Answer_Extraction_Pattern, Prompt_Construction_Pattern |
| Wrapper Doc | 0 | — |
| External Tool Doc | 0 | — |

## Files Created

### Principles (20 files)

```
principles/
├── Alibaba_NLP_DeepResearch_Agent_Initialization.md
├── Alibaba_NLP_DeepResearch_Answer_Extraction.md
├── Alibaba_NLP_DeepResearch_Behavioral_Statistics.md
├── Alibaba_NLP_DeepResearch_Context_Management.md
├── Alibaba_NLP_DeepResearch_Enhanced_Statistics.md
├── Alibaba_NLP_DeepResearch_Image_Processing.md
├── Alibaba_NLP_DeepResearch_LLM_Judge_Scoring.md
├── Alibaba_NLP_DeepResearch_Multi_Turn_Agent_Loop.md
├── Alibaba_NLP_DeepResearch_Pass_At_K_Metrics.md
├── Alibaba_NLP_DeepResearch_Prompt_Construction.md
├── Alibaba_NLP_DeepResearch_ReAct_Loop_Execution.md
├── Alibaba_NLP_DeepResearch_Result_Aggregation.md
├── Alibaba_NLP_DeepResearch_Result_Collection.md
├── Alibaba_NLP_DeepResearch_Reverse_Image_Search.md
├── Alibaba_NLP_DeepResearch_Sandbox_Code_Execution_Multimodal.md
├── Alibaba_NLP_DeepResearch_Sandboxed_Code_Execution.md
├── Alibaba_NLP_DeepResearch_Text_Web_Search.md
├── Alibaba_NLP_DeepResearch_Web_Search_Execution.md
├── Alibaba_NLP_DeepResearch_Webpage_Visitation.md
└── Alibaba_NLP_DeepResearch_Webpage_Visitation_Multimodal.md
```

### Implementations (20 files)

```
implementations/
├── Alibaba_NLP_DeepResearch_Aggregate_Results.md
├── Alibaba_NLP_DeepResearch_Answer_Extraction_Pattern.md
├── Alibaba_NLP_DeepResearch_Calculate_Enhanced_Statistics.md
├── Alibaba_NLP_DeepResearch_Calculate_Pass_At_K.md
├── Alibaba_NLP_DeepResearch_Call_Llm_Judge.md
├── Alibaba_NLP_DeepResearch_MultiTurnReactAgent__init__.md
├── Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run.md
├── Alibaba_NLP_DeepResearch_OmniSearch_process_image.md
├── Alibaba_NLP_DeepResearch_OmniSearch_run_main.md
├── Alibaba_NLP_DeepResearch_Process_Single_Round.md
├── Alibaba_NLP_DeepResearch_Prompt_Construction_Pattern.md
├── Alibaba_NLP_DeepResearch_PythonInterpreter_call.md
├── Alibaba_NLP_DeepResearch_Search_call.md
├── Alibaba_NLP_DeepResearch_Single_Round_Statistics.md
├── Alibaba_NLP_DeepResearch_VLSearchImage_search_image_by_image_url.md
├── Alibaba_NLP_DeepResearch_Visit_call.md
├── Alibaba_NLP_DeepResearch_Visit_call_multimodal.md
├── Alibaba_NLP_DeepResearch_WebSearch_call.md
├── Alibaba_NLP_DeepResearch_count_tokens.md
└── Alibaba_NLP_DeepResearch_run_code_in_sandbox.md
```

## Index Files Updated

- `_ImplementationIndex.md` - 20 entries added
- `_PrincipleIndex.md` - 20 entries added

## Coverage Summary

| Metric | Value |
|--------|-------|
| WorkflowIndex step entries | 23 |
| Unique Principles created | 20 |
| Implementation pages created | 20 |
| 1:1 Mapping coverage | 100% |

## External Dependencies Documented

### External Libraries (Wrapper contexts)

| Library | Usage | Documented In |
|---------|-------|---------------|
| `openai` | LLM API client | MultiTurnReactAgent__run, Call_Llm_Judge |
| `transformers` | AutoTokenizer for token counting | count_tokens, Single_Round_Statistics |
| `serpapi` | Google reverse image search | VLSearchImage_search_image_by_image_url |
| `requests` | HTTP requests for external APIs | Search_call, Visit_call, WebSearch_call |
| `litellm` | Multi-provider LLM client for judge | Call_Llm_Judge |
| `sandbox_fusion` | Sandboxed Python execution | PythonInterpreter_call |
| `PIL` | Image processing | OmniSearch_process_image |
| `oss2` | Alibaba Cloud OSS | VLSearchImage (upload) |

### External Services

| Service | API Endpoint | Tools Using It |
|---------|--------------|----------------|
| Google Serper | `google.serper.dev/search` | Search_call, WebSearch_call |
| Jina AI Reader | `r.jina.ai/{url}` | Visit_call, Visit_call_multimodal |
| SerpAPI | `serpapi.GoogleSearch` | VLSearchImage_search_image_by_image_url |
| SandboxFusion | HTTP POST `/run_code` | PythonInterpreter_call, run_code_in_sandbox |

## Notes for Enrichment Phase

### Heuristics to Document

1. **Token Limit Management** - Max 110K tokens before forced answer generation
2. **Retry Strategies** - Exponential backoff with jitter for API calls
3. **Image Resizing** - max_pixels=1024*28*28, min_pixels=256*28*28 constraints
4. **Multi-Service Fallback** - Visit tool falls back through: Wikipedia dict → aidata cache → aidata online → Jina
5. **Search Locale Detection** - Auto-detects Chinese characters to switch locale

### Environment Pages to Create

1. **CUDA_Requirements** - GPU requirements for vLLM inference
2. **API_Keys** - Required environment variables (SERPER_KEY_ID, JINA_API_KEYS, etc.)
3. **Python_Dependencies** - transformers, openai, serpapi, sandbox_fusion, etc.
4. **External_Services** - Serper, Jina AI, SerpAPI, SandboxFusion

### Concept-Only Principles (No Implementation)

*None identified* - All principles have 1:1 implementation mappings.

---

**Generated:** 2026-01-15
**Phase:** 2 - Excavation + Synthesis
**Repository:** Alibaba_NLP_DeepResearch
