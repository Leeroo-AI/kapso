# Phase 1a: Anchoring Report

## Summary

- Workflows created: 3
- Total steps documented: 23

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| ReAct_Web_Research | `inference/react_agent.py`, `inference/run_multi_react.py`, `inference/tool_*.py` | 7 | `MultiTurnReactAgent._run`, `Search.call`, `Visit.call`, `PythonInterpreter.call` |
| Multimodal_VL_Search | `WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py`, `qwen_agent/tools/*.py` | 8 | `OmniSearch`, `VLSearchImage.search_image_by_image_url`, `Qwen_agent` |
| Benchmark_Evaluation | `evaluation/evaluate_deepsearch_official.py`, `evaluation/prompt.py` | 8 | `call_llm_judge`, `aggregate_results`, `calculate_pass_at_k` |

## Coverage Summary

- Source files covered: 68 (out of 130 total)
- Package files covered: 16 (100% of package files)
- WebWatcher files covered: 52 (multimodal workflow)
- Example files documented: All core inference and evaluation examples

## Source Files Identified Per Workflow

### Alibaba_NLP_DeepResearch_ReAct_Web_Research

| File | Purpose |
|------|---------|
| `inference/react_agent.py` | Core ReAct agent loop with think-act-observe cycle |
| `inference/run_multi_react.py` | Multi-rollout parallel execution runner |
| `inference/prompt.py` | System prompt with tool definitions in XML format |
| `inference/tool_search.py` | Google Serper API web search with locale detection |
| `inference/tool_visit.py` | Jina AI webpage fetching with LLM summarization |
| `inference/tool_python.py` | Sandboxed Python execution via SandboxFusion |
| `inference/tool_scholar.py` | Google Scholar academic search |
| `inference/tool_file.py` | Multi-format document parser wrapper |
| `inference/file_tools/file_parser.py` | PDF, DOCX, PPTX, CSV parsing implementation |
| `inference/file_tools/idp.py` | Alibaba DocMind API integration |
| `inference/file_tools/utils.py` | File handling utilities |
| `inference/file_tools/video_agent.py` | Video analysis orchestrator |
| `inference/file_tools/video_analysis.py` | Multimedia frame extraction |

### Alibaba_NLP_DeepResearch_Multimodal_VL_Search

| File | Purpose |
|------|---------|
| `WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py` | Main multimodal agent evaluation entry point |
| `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/qwen_tool_call.py` | Qwen tool calling adapter |
| `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/generation.py` | Multi-turn generation with image support |
| `qwen_agent/tools/vl_search_image.py` | Google reverse image search via SerpAPI |
| `qwen_agent/tools/vl_search_text.py` | Text-to-image search |
| `qwen_agent/tools/private/nlp_web_search.py` | Serper web search for text queries |
| `qwen_agent/tools/private/jialong_visit.py` | Advanced webpage visitor with extraction |
| `qwen_agent/tools/code_interpreter.py` | Jupyter kernel sandbox |
| `qwen_agent/tools/code_interpreter_http.py` | Cloud code execution |
| `qwen_agent/llm/base.py` | BaseChatModel abstract class |
| `qwen_agent/llm/qwenvl_dashscope.py` | Vision-language DashScope client |
| `qwen_agent/agent.py` | Abstract Agent base class |

### Alibaba_NLP_DeepResearch_Benchmark_Evaluation

| File | Purpose |
|------|---------|
| `evaluation/evaluate_deepsearch_official.py` | Main evaluation script with LLM-as-Judge |
| `evaluation/evaluate_hle_official.py` | Humanity's Last Exam benchmark evaluator |
| `evaluation/prompt.py` | Judge prompts for GAIA, BrowseComp, WebWalker, XBench |
| `inference/run_multi_react.py` | Multi-rollout inference runner (shared) |

## Notes for Phase 1b (Enrichment)

### Files that need line-by-line tracing

1. `inference/react_agent.py:120-226` - The `_run` method contains the core ReAct loop logic
2. `evaluation/evaluate_deepsearch_official.py:76-144` - The `call_llm_judge` function with retry logic
3. `qwen_agent/tools/vl_search_image.py:75-100` - The `search_image_by_image_url` method
4. `inference/tool_visit.py:179-254` - The `readpage_jina` method with LLM summarization

### External APIs to document

| API | Library | Usage |
|-----|---------|-------|
| Google Serper | `http.client` | Web search via `google.serper.dev/search` |
| Jina AI Reader | `requests` | Webpage content extraction via `r.jina.ai/{url}` |
| SerpAPI | `serpapi` | Reverse image search via GoogleSearch |
| LiteLLM | `litellm` | LLM-as-Judge via `litellm.completion()` |
| SandboxFusion | HTTP POST | Code execution sandbox |
| DashScope | `dashscope` | Qwen model inference |
| OpenAI | `openai` | GPT-4o judge and alternative inference |

### Any unclear mappings

1. **WebAgent variants**: The repository contains multiple agent implementations (WebDancer, WebSailor, WebResummer, WebWalker, NestBrowse, ParallelMuse) that share similar patterns but have different specializations. These could be documented as additional workflows in future phases if needed.

2. **Qwen-agent framework**: The `WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/` directory contains a full fork of the qwen-agent framework with modifications. The relationship between this fork and the upstream package should be clarified.

3. **Tool registration pattern**: The `@register_tool` decorator pattern appears throughout the codebase and could be documented as a separate pattern for Phase 1b.

## Architecture Insights

### ReAct Loop Pattern

All workflows follow the ReAct (Reasoning + Acting) paradigm:
1. Agent receives question/query
2. Agent generates `<think>` reasoning
3. Agent selects tool via `<tool_call>` with JSON arguments
4. Tool executes and returns `<tool_response>`
5. Loop continues until `<answer>` is generated

### Multi-Rollout Evaluation

The evaluation strategy uses N independent runs (typically 3) per question to compute:
- Pass@1: Single-run accuracy (best and average)
- Pass@3: At-least-one-correct across 3 runs
- This provides more robust metrics for non-deterministic agent behavior

### Tool Ecosystem

| Tool | Input | Output | External Service |
|------|-------|--------|-----------------|
| search | query string(s) | Search snippets with URLs | Google Serper |
| visit | URL, goal | Summarized content | Jina AI + LLM |
| PythonInterpreter | code string | stdout capture | SandboxFusion |
| google_scholar | query string(s) | Academic paper snippets | Google Scholar via Serper |
| parse_file | file path(s) | Extracted text content | DocMind / local parsers |
| VLSearchImage | image URL | Similar images + metadata | SerpAPI reverse search |

## Completion Status

- [x] Read Phase 0 report
- [x] Read Repository Map
- [x] Identify 3 candidate workflows
- [x] Create ReAct_Web_Research workflow page
- [x] Create Multimodal_VL_Search workflow page
- [x] Create Benchmark_Evaluation workflow page
- [x] Update Repository Map with coverage
- [x] Create WorkflowIndex with rough structure
- [x] Write Phase 1a execution report
