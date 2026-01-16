# Repository Map: Alibaba_NLP_DeepResearch

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/Alibaba-NLP/DeepResearch |
| Branch | main |
| Generated | 2026-01-15 19:00 |
| Python Files | 130 |
| Total Lines | 22,640 |
| Explored | 130/130 |

## Structure

📦 **Packages:** evaluation, inference

📖 README: `README.md`

---

## 📦 Package Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `evaluation/evaluate_deepsearch_official.py` | 584 | Deep search benchmark eval | Workflow: Alibaba_NLP_DeepResearch_Benchmark_Evaluation | [→](./_files/evaluation_evaluate_deepsearch_official_py.md) |
| ✅ | `evaluation/evaluate_hle_official.py` | 243 | HLE benchmark evaluation | Workflow: Alibaba_NLP_DeepResearch_Benchmark_Evaluation | [→](./_files/evaluation_evaluate_hle_official_py.md) |
| ✅ | `evaluation/prompt.py` | 458 | Agent and judge prompts | Workflow: Alibaba_NLP_DeepResearch_Benchmark_Evaluation | [→](./_files/evaluation_prompt_py.md) |
| ✅ | `inference/file_tools/file_parser.py` | 578 | Multi-format doc parser | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_file_tools_file_parser_py.md) |
| ✅ | `inference/file_tools/idp.py` | 90 | Alibaba DocMind API | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_file_tools_idp_py.md) |
| ✅ | `inference/file_tools/utils.py` | 542 | File handling utilities | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_file_tools_utils_py.md) |
| ✅ | `inference/file_tools/video_agent.py` | 92 | Video analysis orchestrator | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_file_tools_video_agent_py.md) |
| ✅ | `inference/file_tools/video_analysis.py` | 619 | Multimedia frame extraction | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_file_tools_video_analysis_py.md) |
| ✅ | `inference/prompt.py` | 51 | System prompt templates | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_prompt_py.md) |
| ✅ | `inference/react_agent.py` | 247 | Core ReAct agent loop | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_react_agent_py.md) |
| ✅ | `inference/run_multi_react.py` | 229 | Multi-rollout evaluation | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research, Alibaba_NLP_DeepResearch_Benchmark_Evaluation | [→](./_files/inference_run_multi_react_py.md) |
| ✅ | `inference/tool_file.py` | 141 | File parsing tool wrapper | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_tool_file_py.md) |
| ✅ | `inference/tool_python.py` | 150 | Sandboxed code execution | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_tool_python_py.md) |
| ✅ | `inference/tool_scholar.py` | 110 | Google Scholar search | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_tool_scholar_py.md) |
| ✅ | `inference/tool_search.py` | 131 | Web search via Serper | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_tool_search_py.md) |
| ✅ | `inference/tool_visit.py` | 256 | Webpage content extractor | Workflow: Alibaba_NLP_DeepResearch_ReAct_Web_Research | [→](./_files/inference_tool_visit_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `WebAgent/NestBrowse/infer_async_nestbrowse.py` | 203 | Async browser agent | Implementation: Alibaba_NLP_DeepResearch_NestBrowse_Infer_Async | [→](./_files/WebAgent_NestBrowse_infer_async_nestbrowse_py.md) |
| ✅ | `WebAgent/NestBrowse/prompts.py` | 82 | Browser agent prompts | — | [→](./_files/WebAgent_NestBrowse_prompts_py.md) |
| ✅ | `WebAgent/NestBrowse/toolkit/browser.py` | 192 | Browser interaction tools | Implementation: Alibaba_NLP_DeepResearch_NestBrowse_Browser_Tools | [→](./_files/WebAgent_NestBrowse_toolkit_browser_py.md) |
| ✅ | `WebAgent/NestBrowse/toolkit/mcp_client.py` | 49 | MCP protocol client | Implementation: Alibaba_NLP_DeepResearch_MCP_Client | [→](./_files/WebAgent_NestBrowse_toolkit_mcp_client_py.md) |
| ✅ | `WebAgent/NestBrowse/toolkit/tool_explore.py` | 47 | Page content summarizer | Implementation: Alibaba_NLP_DeepResearch_Tool_Explore | [→](./_files/WebAgent_NestBrowse_toolkit_tool_explore_py.md) |
| ✅ | `WebAgent/NestBrowse/toolkit/tool_search.py` | 5 | Search tool placeholder | — | [→](./_files/WebAgent_NestBrowse_toolkit_tool_search_py.md) |
| ✅ | `WebAgent/NestBrowse/utils.py` | 81 | LLM call utilities | — | [→](./_files/WebAgent_NestBrowse_utils_py.md) |
| ✅ | `WebAgent/ParallelMuse/compressed_reasoning_aggregation.py` | 295 | Multi-trajectory aggregation | Implementation: Alibaba_NLP_DeepResearch_Compressed_Reasoning_Aggregation | [→](./_files/WebAgent_ParallelMuse_compressed_reasoning_aggregation_py.md) |
| ✅ | `WebAgent/ParallelMuse/functionality_specified_partial_rollout.py` | 526 | Uncertainty-based branching | Implementation: Alibaba_NLP_DeepResearch_Functionality_Specified_Partial_Rollout | [→](./_files/WebAgent_ParallelMuse_functionality_specified_partial_rollout_py.md) |
| ✅ | `WebAgent/WebDancer/demos/__init__.py` | 1 | Package initializer | — | [→](./_files/WebAgent_WebDancer_demos___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/agents/__init__.py` | 1 | Agents package init | — | [→](./_files/WebAgent_WebDancer_demos_agents___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/agents/search_agent.py` | 113 | Web search agent class | Implementation: Alibaba_NLP_DeepResearch_SearchAgent | [→](./_files/WebAgent_WebDancer_demos_agents_search_agent_py.md) |
| ✅ | `WebAgent/WebDancer/demos/assistant_qwq_chat.py` | 140 | WebDancer demo entry | Implementation: Alibaba_NLP_DeepResearch_WebDancer_Demo | [→](./_files/WebAgent_WebDancer_demos_assistant_qwq_chat_py.md) |
| ✅ | `WebAgent/WebDancer/demos/gui/__init__.py` | 1 | GUI package init | — | [→](./_files/WebAgent_WebDancer_demos_gui___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/gui/html_decorate.py` | 157 | Markdown to HTML render | — | [→](./_files/WebAgent_WebDancer_demos_gui_html_decorate_py.md) |
| ✅ | `WebAgent/WebDancer/demos/gui/web_ui.py` | 389 | Gradio chat interface | Implementation: Alibaba_NLP_DeepResearch_WebUI | [→](./_files/WebAgent_WebDancer_demos_gui_web_ui_py.md) |
| ✅ | `WebAgent/WebDancer/demos/llm/__init__.py` | 1 | LLM package init | — | [→](./_files/WebAgent_WebDancer_demos_llm___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/llm/oai.py` | 218 | OpenAI API backend | Implementation: Alibaba_NLP_DeepResearch_TextChatAtOAI | [→](./_files/WebAgent_WebDancer_demos_llm_oai_py.md) |
| ✅ | `WebAgent/WebDancer/demos/llm/qwen_dashscope.py` | 140 | DashScope API backend | Implementation: Alibaba_NLP_DeepResearch_QwenChatAtDS | [→](./_files/WebAgent_WebDancer_demos_llm_qwen_dashscope_py.md) |
| ✅ | `WebAgent/WebDancer/demos/tools/__init__.py` | 7 | Tools module exports | — | [→](./_files/WebAgent_WebDancer_demos_tools___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/tools/private/__init__.py` | 8 | Private tools init | — | [→](./_files/WebAgent_WebDancer_demos_tools_private___init___py.md) |
| ✅ | `WebAgent/WebDancer/demos/tools/private/cache_utils.py` | 57 | JSONL file caching | — | [→](./_files/WebAgent_WebDancer_demos_tools_private_cache_utils_py.md) |
| ✅ | `WebAgent/WebDancer/demos/tools/private/search.py` | 99 | Serper search tool | Implementation: Alibaba_NLP_DeepResearch_WebDancer_Search_Tool | [→](./_files/WebAgent_WebDancer_demos_tools_private_search_py.md) |
| ✅ | `WebAgent/WebDancer/demos/tools/private/visit.py` | 173 | Jina webpage visitor | Implementation: Alibaba_NLP_DeepResearch_WebDancer_Visit_Tool | [→](./_files/WebAgent_WebDancer_demos_tools_private_visit_py.md) |
| ✅ | `WebAgent/WebDancer/demos/utils/date.py` | 71 | Date utility functions | — | [→](./_files/WebAgent_WebDancer_demos_utils_date_py.md) |
| ✅ | `WebAgent/WebDancer/demos/utils/logs.py` | 51 | Logging configuration | — | [→](./_files/WebAgent_WebDancer_demos_utils_logs_py.md) |
| ✅ | `WebAgent/WebResummer/src/evaluate.py` | 309 | LLM-judge evaluation | Implementation: Alibaba_NLP_DeepResearch_WebResummer_Evaluate | [→](./_files/WebAgent_WebResummer_src_evaluate_py.md) |
| ✅ | `WebAgent/WebResummer/src/judge_prompt.py` | 150 | Judge prompt templates | — | [→](./_files/WebAgent_WebResummer_src_judge_prompt_py.md) |
| ✅ | `WebAgent/WebResummer/src/main.py` | 164 | Multi-rollout entry point | Implementation: Alibaba_NLP_DeepResearch_WebResummer_Main | [→](./_files/WebAgent_WebResummer_src_main_py.md) |
| ✅ | `WebAgent/WebResummer/src/prompt.py` | 169 | System prompts and tools | — | [→](./_files/WebAgent_WebResummer_src_prompt_py.md) |
| ✅ | `WebAgent/WebResummer/src/react_agent.py` | 202 | ReAct agent with ReSum | Implementation: Alibaba_NLP_DeepResearch_WebResummer_ReActAgent | [→](./_files/WebAgent_WebResummer_src_react_agent_py.md) |
| ✅ | `WebAgent/WebResummer/src/summary_utils.py` | 66 | Conversation summarizer | Implementation: Alibaba_NLP_DeepResearch_Summary_Utils | [→](./_files/WebAgent_WebResummer_src_summary_utils_py.md) |
| ✅ | `WebAgent/WebResummer/src/tool_search.py` | 112 | Serper search tool | Implementation: Alibaba_NLP_DeepResearch_WebResummer_Search_Tool | [→](./_files/WebAgent_WebResummer_src_tool_search_py.md) |
| ✅ | `WebAgent/WebResummer/src/tool_visit.py` | 240 | LLM webpage extractor | Implementation: Alibaba_NLP_DeepResearch_WebResummer_Visit_Tool | [→](./_files/WebAgent_WebResummer_src_tool_visit_py.md) |
| ✅ | `WebAgent/WebSailor/src/evaluate.py` | 329 | Pass@k evaluation | Implementation: Alibaba_NLP_DeepResearch_WebSailor_Evaluate | [→](./_files/WebAgent_WebSailor_src_evaluate_py.md) |
| ✅ | `WebAgent/WebSailor/src/prompt.py` | 206 | Agent prompt library | — | [→](./_files/WebAgent_WebSailor_src_prompt_py.md) |
| ✅ | `WebAgent/WebSailor/src/react_agent.py` | 162 | vLLM ReAct agent | Implementation: Alibaba_NLP_DeepResearch_WebSailor_ReActAgent | [→](./_files/WebAgent_WebSailor_src_react_agent_py.md) |
| ✅ | `WebAgent/WebSailor/src/run_multi_react.py` | 188 | Parallel evaluation runner | Implementation: Alibaba_NLP_DeepResearch_WebSailor_Main | [→](./_files/WebAgent_WebSailor_src_run_multi_react_py.md) |
| ✅ | `WebAgent/WebSailor/src/tool_search.py` | 103 | Google Serper search | Implementation: Alibaba_NLP_DeepResearch_WebSailor_Search_Tool | [→](./_files/WebAgent_WebSailor_src_tool_search_py.md) |
| ✅ | `WebAgent/WebSailor/src/tool_visit.py` | 220 | Jina + LLM extractor | Implementation: Alibaba_NLP_DeepResearch_WebSailor_Visit_Tool | [→](./_files/WebAgent_WebSailor_src_tool_visit_py.md) |
| ✅ | `WebAgent/WebWalker/src/agent.py` | 208 | Website navigator agent | Implementation: Alibaba_NLP_DeepResearch_WebWalker_Agent | [→](./_files/WebAgent_WebWalker_src_agent_py.md) |
| ✅ | `WebAgent/WebWalker/src/app.py` | 271 | Streamlit demo UI | Implementation: Alibaba_NLP_DeepResearch_WebWalker_App | [→](./_files/WebAgent_WebWalker_src_app_py.md) |
| ✅ | `WebAgent/WebWalker/src/evaluate.py` | 156 | WebWalkerQA evaluation | Implementation: Alibaba_NLP_DeepResearch_WebWalker_Evaluate | [→](./_files/WebAgent_WebWalker_src_evaluate_py.md) |
| ✅ | `WebAgent/WebWalker/src/prompts.py` | 65 | Navigation prompts | — | [→](./_files/WebAgent_WebWalker_src_prompts_py.md) |
| ✅ | `WebAgent/WebWalker/src/rag_system.py` | 335 | Baseline RAG comparison | Implementation: Alibaba_NLP_DeepResearch_RAG_System | [→](./_files/WebAgent_WebWalker_src_rag_system_py.md) |
| ✅ | `WebAgent/WebWalker/src/utils.py` | 77 | URL and markdown utils | — | [→](./_files/WebAgent_WebWalker_src_utils_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/evaluation/evaluate_hle_official.py` | 245 | HLE LLM-judge eval | Workflow: Alibaba_NLP_DeepResearch_Benchmark_Evaluation | [→](./_files/WebAgent_WebWatcher_infer_evaluation_evaluate_hle_official_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/evaluation/prompt.py` | 458 | WebWatcher prompts | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_evaluation_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py` | 481 | Multimodal agent eval | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_agent_eval_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/download_image.py` | 46 | Image dataset downloader | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_download_image_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/__init__.py` | 0 | Code package init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_code___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/code_register.py` | 17 | Code tool registration | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_code_code_register_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/sandbox_module.py` | 53 | Remote code sandbox | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_code_sandbox_module_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/__init__.py` | 0 | LLM agent package init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_llm_agent___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/generation.py` | 730 | Multi-turn generation | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_llm_agent_generation_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/qwen_tool_call.py` | 60 | Qwen tool adapter | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_llm_agent_qwen_tool_call_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/tensor_helper.py` | 75 | Tensor padding utils | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_scripts_eval_mmrag_r1_llm_agent_tensor_helper_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/__init__.py` | 8 | Qwen-agent entry point | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/agent.py` | 316 | Abstract Agent base | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_agent_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/__init__.py` | 94 | LLM factory module | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/azure.py` | 41 | Azure OpenAI adapter | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_azure_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/base.py` | 580 | BaseChatModel ABC | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_base_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/__init__.py` | 0 | Fncall prompts init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/base_fncall_prompt.py` | 72 | Abstract prompt base | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts_base_fncall_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/code_fncall_prompt.py` | 191 | Code block fn format | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts_code_fncall_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py` | 208 | Nous/Hermes XML format | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts_nous_fncall_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/nous_fncall_prompt_think.py` | 283 | XML + think reasoning | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts_nous_fncall_prompt_think_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py` | 389 | Native Qwen fn format | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_fncall_prompts_qwen_fncall_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/function_calling.py` | 182 | Tool calling base | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_function_calling_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/oai.py` | 168 | OpenAI API backend | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_oai_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/openvino.py` | 148 | Local CPU inference | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_openvino_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwen_dashscope.py` | 144 | Qwen DashScope client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_qwen_dashscope_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenaudio_dashscope.py` | 12 | Audio model client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_qwenaudio_dashscope_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenomni_dashscope.py` | 17 | Omni-modal client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_qwenomni_dashscope_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenvl_dashscope.py` | 144 | Vision-language client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_qwenvl_dashscope_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenvl_oai.py` | 59 | VL OpenAI-style API | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_qwenvl_oai_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/schema.py` | 142 | Message data models | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_llm_schema_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/log.py` | 23 | Centralized logging | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_log_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/memory/__init__.py` | 5 | Memory package init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_memory___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/memory/memory.py` | 137 | RAG memory agent | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_memory_memory_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/settings.py` | 27 | Configuration constants | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_settings_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/__init__.py` | 42 | Tools registry exports | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/base.py` | 202 | BaseTool abstract class | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_base_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/code_interpreter.py` | 413 | Jupyter kernel sandbox | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_code_interpreter_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/code_interpreter_http.py` | 169 | Cloud code execution | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_code_interpreter_http_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/__init__.py` | 0 | GPT4o tools init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/base.py` | 31 | API client base class | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_base_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/constant.py` | 8 | API arg whitelist | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_constant_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/eleven_tts.py` | 30 | ElevenLabs TTS client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_eleven_tts_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/logger.py` | 178 | Color-coded logger | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_logger_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/openai_style_api_client.py` | 174 | Multi-provider LLM client | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_openai_style_api_client_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/parallel_run.py` | 133 | Batch parallel inference | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_parallel_run_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/utils.py` | 249 | Shared utility functions | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_gpt4o_utils_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/__init__.py` | 0 | Private tools init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/cache_utils.py` | 57 | Thread-safe JSONL cache | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_cache_utils_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/jialong_visit.py` | 318 | Advanced webpage visitor | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_jialong_visit_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/nlp_web_search.py` | 137 | Serper web search | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_nlp_web_search_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/prompt.py` | 206 | Private prompt templates | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_prompt_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/readpage.py` | 195 | Alternative page reader | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_readpage_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/sfilter.py` | 159 | Sentence relevance filter | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_sfilter_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/visit.py` | 239 | vLLM webpage visitor | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_private_visit_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/resource/code_interpreter_init_kernel.py` | 50 | Kernel initialization | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_resource_code_interpreter_init_kernel_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/simple_doc_parser.py` | 550 | Multi-format doc parser | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_simple_doc_parser_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/storage.py` | 102 | File-based key-value | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_storage_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_image.py` | 324 | Reverse image search | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_vl_search_image_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_text.py` | 328 | Text-to-image search | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_tools_vl_search_text_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/__init__.py` | 0 | Utils package init | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils___init___py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/code_safety_checker.py` | 345 | Code security validator | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_code_safety_checker_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/csi.py` | 90 | Content safety inspection | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_csi_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/parallel_executor.py` | 85 | Concurrent tool runner | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_parallel_executor_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/str_processing.py` | 30 | Text cleaning utilities | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_str_processing_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/tokenization_qwen.py` | 217 | Qwen tokenizer impl | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_tokenization_qwen_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/utils.py` | 551 | Core utility library | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_qwen_agent_utils_utils_py.md) |
| ✅ | `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/setup.py` | 113 | Package distribution | Workflow: Alibaba_NLP_DeepResearch_Multimodal_VL_Search | [→](./_files/WebAgent_WebWatcher_infer_vl_search_r1_qwen-agent-o1_search_setup_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description |
|-------|-------------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |
