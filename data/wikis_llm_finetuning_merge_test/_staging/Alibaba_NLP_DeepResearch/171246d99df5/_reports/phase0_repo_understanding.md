# Phase 0: Repository Understanding Report

## Summary
- Files explored: 130/130
- Completion: 100%

## Repository Overview

**Alibaba_NLP_DeepResearch** is a comprehensive research framework for building and evaluating web-based AI agents that can perform deep research tasks. The repository contains multiple distinct agent systems and a shared evaluation infrastructure.

## Key Discoveries

### Main Entry Points

1. **Core Inference System** (`inference/`)
   - `react_agent.py` - Core ReAct (Reasoning + Acting) agent loop
   - `run_multi_react.py` - Production multi-rollout evaluation runner
   - `prompt.py` - System prompts and tool definitions

2. **Evaluation System** (`evaluation/`)
   - `evaluate_deepsearch_official.py` - Multi-benchmark evaluator (GAIA, BrowseComp, WebWalker, XBench)
   - `evaluate_hle_official.py` - Humanity's Last Exam benchmark evaluator
   - `prompt.py` - Extensive prompt library for agents and LLM judges

### Core Modules Identified

#### Agent Systems (WebAgent/)

| Agent | Purpose | Key Innovation |
|-------|---------|----------------|
| **NestBrowse** | Browser-based async agent | MCP protocol for browser control, click/fill interactions |
| **ParallelMuse** | Multi-trajectory sampling | Uncertainty-based branching via perplexity, reasoning aggregation |
| **WebDancer** | Interactive web search demo | Gradio UI, search + visit tools, citation generation |
| **WebResummer** | Context-aware ReAct agent | ReSum mechanism for conversation summarization |
| **WebSailor** | Lightweight vLLM agent | Local inference, Jina reader integration |
| **WebWalker** | Single-website navigation | Memory accumulation, critic stages for answer generation |
| **WebWatcher** | Multimodal vision-language | Image/text search, code execution, comprehensive tooling |

#### Shared Infrastructure

1. **qwen-agent Framework** (`WebWatcher/.../qwen_agent/`)
   - Full agent framework with LLM backends (OpenAI, DashScope, Azure, OpenVINO)
   - Tool registration system with 40+ built-in tools
   - Multiple function calling formats (Qwen, Nous/Hermes, code blocks)
   - Memory and RAG support

2. **Tools Ecosystem**
   - `search` - Web search via Google Serper API
   - `visit` - Webpage extraction via Jina AI + LLM summarization
   - `code_interpreter` - Sandboxed Python execution (Jupyter or HTTP)
   - `VLSearchImage` / `VLSearchText` - Vision-language search
   - `file_parser` - Multi-format document parsing (PDF, DOCX, PPTX, etc.)

### Architecture Patterns Observed

1. **ReAct Loop Pattern**: All agents implement think → tool_call → tool_response → answer cycles with XML-style tags
2. **Multi-Rollout Evaluation**: Pass@k metrics computed from N independent runs (typically 3)
3. **LLM-as-Judge**: Evaluation uses structured prompts with Qwen2.5-72B or GPT-4o as judges
4. **Parallel Tool Execution**: ThreadPoolExecutor for concurrent search/visit operations
5. **Caching Layer**: JSONL-based caching with file locking for expensive web operations
6. **Security Sandboxing**: AST-based code safety checking before execution

### File Type Distribution

| Category | Count | Description |
|----------|-------|-------------|
| Package Files | 16 | Core evaluation and inference modules |
| WebAgent Files | 114 | Agent implementations and tooling |
| - NestBrowse | 7 | Browser automation agent |
| - ParallelMuse | 2 | Multi-trajectory reasoning |
| - WebDancer | 17 | Interactive demo system |
| - WebResummer | 8 | Context summarization agent |
| - WebSailor | 6 | Lightweight search agent |
| - WebWalker | 6 | Website navigation agent |
| - WebWatcher | 68 | Multimodal agent framework |

## Recommendations for Next Phase

### Suggested Workflows to Document

1. **Web Research Workflow**: search → visit → extract → iterate → answer
2. **Multimodal Query Workflow**: image upload → VLSearch → context retrieval → answer
3. **Code Execution Workflow**: query → code generation → sandbox execution → result processing
4. **Evaluation Pipeline**: data loading → agent rollouts → LLM judging → metric aggregation

### Key APIs to Trace

1. `MultiTurnReactAgent._run()` - Core agent loop execution
2. `Search.call()` / `Visit.call()` - Tool implementations
3. `BaseChatModel.chat()` - LLM interaction with retries and caching
4. `call_llm_judge()` - Evaluation scoring

### Important Files for Anchoring Phase

| File | Significance |
|------|--------------|
| `inference/react_agent.py:247` | Core ReAct implementation |
| `inference/prompt.py:51` | Tool definitions and system prompts |
| `evaluation/prompt.py:458` | Complete prompt library |
| `WebWatcher/.../agent.py:316` | Abstract Agent base class |
| `WebWatcher/.../base.py:202` | Tool registration infrastructure |

### Benchmark Support

The repository supports evaluation on:
- GAIA (General AI Assistants)
- BrowseComp (Chinese/English web browsing comprehension)
- HLE (Humanity's Last Exam)
- WebWalkerQA (website navigation QA)
- XBench-DeepSearch
- SimpleQA, LiveVQA, MMSearch

## Technical Notes

- **Primary LLM**: Qwen2.5 series (72B-Instruct for judgment, various sizes for agents)
- **External Services**: Serper API, Jina AI Reader, Alibaba DashScope, Alibaba IDP
- **Deployment Options**: Cloud (DashScope), local vLLM, OpenVINO CPU inference
- **Security**: Multi-layer code safety checking, content safety inspection (CSI)
