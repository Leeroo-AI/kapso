# Repository Map: langchain-ai_langchain

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/langchain-ai/langchain |
| Branch | main |
| Generated | 2025-12-18 12:30 |
| Python Files | 200 |
| Total Lines | 46,629 |
| Explored | 200/200 |

## Structure


📖 README: `README.md`

---

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.github/scripts/check_diff.py` | 340 | CI test optimization | — | [→](./_files/_github_scripts_check_diff_py.md) |
| ✅ | `.github/scripts/check_prerelease_dependencies.py` | 36 | Release dependency validation | — | [→](./_files/_github_scripts_check_prerelease_dependencies_py.md) |
| ✅ | `.github/scripts/get_min_versions.py` | 199 | Minimum version testing | — | [→](./_files/_github_scripts_get_min_versions_py.md) |
| ✅ | `libs/langchain/langchain_classic/__init__.py` | 424 | Deprecation management entry | — | [→](./_files/libs_langchain_langchain_classic___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/base_language.py` | 7 | BaseLanguageModel re-export | — | [→](./_files/libs_langchain_langchain_classic_base_language_py.md) |
| ✅ | `libs/langchain/langchain_classic/base_memory.py` | 116 | Memory abstraction (deprecated) | — | [→](./_files/libs_langchain_langchain_classic_base_memory_py.md) |
| ✅ | `libs/langchain/langchain_classic/cache.py` | 72 | Lazy cache import proxy | — | [→](./_files/libs_langchain_langchain_classic_cache_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/__init__.py` | 96 | Chains public API entry | — | [→](./_files/libs_langchain_langchain_classic_chains___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/base.py` | 806 | Abstract Chain base class | — | [→](./_files/libs_langchain_langchain_classic_chains_base_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/example_generator.py` | 22 | Example generation utility | — | [→](./_files/libs_langchain_langchain_classic_chains_example_generator_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/history_aware_retriever.py` | 68 | Conversational retriever factory | — | [→](./_files/libs_langchain_langchain_classic_chains_history_aware_retriever_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/llm.py` | 432 | Deprecated LLMChain impl | — | [→](./_files/libs_langchain_langchain_classic_chains_llm_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/llm_requests.py` | 23 | LLMRequestsChain redirect | — | [→](./_files/libs_langchain_langchain_classic_chains_llm_requests_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/loading.py` | 742 | Chain serialization/deserialization | — | [→](./_files/libs_langchain_langchain_classic_chains_loading_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/mapreduce.py` | 117 | Map-reduce document processing | — | [→](./_files/libs_langchain_langchain_classic_chains_mapreduce_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/moderation.py` | 129 | OpenAI content moderation | — | [→](./_files/libs_langchain_langchain_classic_chains_moderation_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/prompt_selector.py` | 65 | Dynamic prompt selection | — | [→](./_files/libs_langchain_langchain_classic_chains_prompt_selector_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/retrieval.py` | 68 | RAG chain factory | — | [→](./_files/libs_langchain_langchain_classic_chains_retrieval_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/sequential.py` | 208 | Sequential chain composition | — | [→](./_files/libs_langchain_langchain_classic_chains_sequential_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/transform.py` | 79 | Custom function wrapper | — | [→](./_files/libs_langchain_langchain_classic_chains_transform_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/__init__.py` | 6 | Chat loaders documentation | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/base.py` | 3 | BaseChatLoader re-export | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_base_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py` | 32 | Facebook Messenger redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_facebook_messenger_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/gmail.py` | 23 | Gmail loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_gmail_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/imessage.py` | 23 | iMessage loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_imessage_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/langsmith.py` | 30 | LangSmith loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_langsmith_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/slack.py` | 23 | Slack loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_slack_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/telegram.py` | 23 | Telegram loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_telegram_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/utils.py` | 36 | Chat loader utilities redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_utils_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/whatsapp.py` | 23 | WhatsApp loader redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_whatsapp_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/baidu_qianfan_endpoint.py` | 27 | Baidu Qianfan redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_models_baidu_qianfan_endpoint_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/everlyai.py` | 23 | EverlyAI redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_models_everlyai_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/tongyi.py` | 23 | Alibaba Tongyi redirect | — | [→](./_files/libs_langchain_langchain_classic_chat_models_tongyi_py.md) |
| ✅ | `libs/langchain/langchain_classic/env.py` | 17 | Runtime environment info | — | [→](./_files/libs_langchain_langchain_classic_env_py.md) |
| ✅ | `libs/langchain/langchain_classic/example_generator.py` | 5 | Example generation shim | — | [→](./_files/libs_langchain_langchain_classic_example_generator_py.md) |
| ✅ | `libs/langchain/langchain_classic/formatting.py` | 5 | String formatting re-export | — | [→](./_files/libs_langchain_langchain_classic_formatting_py.md) |
| ✅ | `libs/langchain/langchain_classic/globals.py` | 19 | Global configuration API | — | [→](./_files/libs_langchain_langchain_classic_globals_py.md) |
| ✅ | `libs/langchain/langchain_classic/hub.py` | 153 | LangChain Hub integration | — | [→](./_files/libs_langchain_langchain_classic_hub_py.md) |
| ✅ | `libs/langchain/langchain_classic/input.py` | 15 | Text formatting re-export | — | [→](./_files/libs_langchain_langchain_classic_input_py.md) |
| ✅ | `libs/langchain/langchain_classic/model_laboratory.py` | 98 | Model comparison utility | — | [→](./_files/libs_langchain_langchain_classic_model_laboratory_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/__init__.py` | 82 | Output parsers API entry | — | [→](./_files/libs_langchain_langchain_classic_output_parsers___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/boolean.py` | 54 | Boolean value extraction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_boolean_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/combining.py` | 58 | Multi-parser orchestration | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_combining_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/datetime.py` | 58 | Temporal data extraction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_datetime_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/enum.py` | 45 | Constrained choice validation | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_enum_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/ernie_functions.py` | 45 | ERNIE parser compat shim | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_ernie_functions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/fix.py` | 156 | LLM-powered error correction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_fix_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/format_instructions.py` | 79 | Format instruction templates | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_format_instructions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/json.py` | 15 | JSON parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_json_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/list.py` | 13 | List parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_list_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/loading.py` | 22 | Config-based parser loading | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_loading_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/openai_functions.py` | 13 | OpenAI functions re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_openai_functions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/openai_tools.py` | 7 | OpenAI tools re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_openai_tools_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py` | 171 | DataFrame query parsing | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_pandas_dataframe_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/prompts.py` | 21 | Fix parser prompts | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_prompts_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/pydantic.py` | 3 | Pydantic parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_pydantic_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/rail_parser.py` | 25 | Guardrails AI compat | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_rail_parser_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/regex.py` | 40 | Pattern-based extraction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_regex_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/regex_dict.py` | 42 | Labeled key-value extraction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_regex_dict_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/retry.py` | 315 | Context-aware retry parsing | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_retry_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/structured.py` | 116 | Schema-based dict extraction | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_structured_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/xml.py` | 3 | XML parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_xml_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/yaml.py` | 69 | YAML format parsing | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_yaml_py.md) |
| ✅ | `libs/langchain/langchain_classic/python.py` | 19 | PythonREPL deprecation shim | — | [→](./_files/libs_langchain_langchain_classic_python_py.md) |
| ✅ | `libs/langchain/langchain_classic/requests.py` | 35 | HTTP utilities shim | — | [→](./_files/libs_langchain_langchain_classic_requests_py.md) |
| ✅ | `libs/langchain/langchain_classic/serpapi.py` | 25 | SerpAPI wrapper redirect | — | [→](./_files/libs_langchain_langchain_classic_serpapi_py.md) |
| ✅ | `libs/langchain/langchain_classic/sql_database.py` | 25 | SQLDatabase redirect | — | [→](./_files/libs_langchain_langchain_classic_sql_database_py.md) |
| ✅ | `libs/langchain/langchain_classic/text_splitter.py` | 50 | Text splitter re-export | Workflow: Text_Splitting_Workflow | [→](./_files/libs_langchain_langchain_classic_text_splitter_py.md) |
| ✅ | `libs/langchain/scripts/check_imports.py` | 33 | Fast import validation | — | [→](./_files/libs_langchain_scripts_check_imports_py.md) |
| ✅ | `libs/langchain/tests/__init__.py` | 1 | Test package marker | — | [→](./_files/libs_langchain_tests___init___py.md) |
| ✅ | `libs/langchain/tests/data.py` | 12 | Test data path constants | — | [→](./_files/libs_langchain_tests_data_py.md) |
| ✅ | `libs/langchain_v1/langchain/__init__.py` | 3 | Package version entry | — | [→](./_files/libs_langchain_v1_langchain___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/__init__.py` | 9 | Public agents API | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/factory.py` | 1682 | Core agent factory | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_factory_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/__init__.py` | 78 | Middleware public API | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_execution.py` | 389 | Shell execution policies | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware__execution_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_redaction.py` | 364 | PII detection utilities | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware__redaction_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_retry.py` | 123 | Shared retry utilities | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware__retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/context_editing.py` | 278 | Context window management | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_context_editing_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/file_search.py` | 387 | Filesystem search tools | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_file_search_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py` | 357 | Human approval workflow | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_human_in_the_loop_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_call_limit.py` | 256 | Model call tracking | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_call_limit_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_fallback.py` | 135 | Model failover on errors | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_fallback_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_retry.py` | 300 | Model retry with backoff | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/pii.py` | 369 | PII detection middleware | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_pii_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/shell_tool.py` | 760 | Persistent shell session | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_shell_tool_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/summarization.py` | 535 | Conversation summarization | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_summarization_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/todo.py` | 224 | Task list management | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_todo_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py` | 488 | Tool call enforcement | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_call_limit_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_emulator.py` | 209 | LLM-based tool emulation | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_emulator_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_retry.py` | 396 | Tool retry with backoff | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_selection.py` | 320 | LLM-based tool filtering | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_selection_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/types.py` | 1848 | Middleware type system | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_middleware_types_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/structured_output.py` | 443 | Structured output strategies | Workflow: Agent_Creation_Workflow, Structured_Output_Workflow | [→](./_files/libs_langchain_v1_langchain_agents_structured_output_py.md) |
| ✅ | `libs/langchain_v1/langchain/chat_models/__init__.py` | 7 | Chat models entry point | Workflow: Chat_Model_Initialization_Workflow | [→](./_files/libs_langchain_v1_langchain_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/chat_models/base.py` | 944 | Universal chat model factory | Workflow: Chat_Model_Initialization_Workflow, Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_chat_models_base_py.md) |
| ✅ | `libs/langchain_v1/langchain/embeddings/__init__.py` | 17 | Embeddings entry point | — | [→](./_files/libs_langchain_v1_langchain_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/embeddings/base.py` | 245 | Embeddings factory function | — | [→](./_files/libs_langchain_v1_langchain_embeddings_base_py.md) |
| ✅ | `libs/langchain_v1/langchain/messages/__init__.py` | 73 | Message types entry point | — | [→](./_files/libs_langchain_v1_langchain_messages___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/rate_limiters/__init__.py` | 13 | Rate limiting abstractions | — | [→](./_files/libs_langchain_v1_langchain_rate_limiters___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/tools/__init__.py` | 22 | Tools entry point | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_tools___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/tools/tool_node.py` | 20 | LangGraph integration layer | Workflow: Agent_Creation_Workflow | [→](./_files/libs_langchain_v1_langchain_tools_tool_node_py.md) |
| ✅ | `libs/langchain_v1/scripts/check_imports.py` | 33 | Fast import validation | — | [→](./_files/libs_langchain_v1_scripts_check_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/__init__.py` | 1 | Test package marker | — | [→](./_files/libs_langchain_v1_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/__init__.py` | 1 | Integration tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/__init__.py` | 1 | Agent tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/middleware/__init__.py` | 1 | Middleware tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/middleware/test_shell_tool_integration.py` | 147 | Shell tool integration tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents_middleware_test_shell_tool_integration_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/cache/__init__.py` | 1 | Cache tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_cache___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/cache/fake_embeddings.py` | 91 | Fake embeddings for testing | — | [→](./_files/libs_langchain_v1_tests_integration_tests_cache_fake_embeddings_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/chat_models/__init__.py` | 0 | Chat model tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/chat_models/test_base.py` | 57 | Chat model factory tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_chat_models_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/conftest.py` | 34 | Integration test config | — | [→](./_files/libs_langchain_v1_tests_integration_tests_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/embeddings/__init__.py` | 0 | Embeddings tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/embeddings/test_base.py` | 44 | Embeddings factory tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_embeddings_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/test_compile.py` | 6 | Compilation verification | — | [→](./_files/libs_langchain_v1_tests_integration_tests_test_compile_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/__init__.py` | 0 | Unit tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/__init__.py` | 0 | Agent unit tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/any_str.py` | 19 | Flexible string matcher | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_any_str_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest.py` | 194 | Agent test fixtures | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest_checkpointer.py` | 64 | Checkpointer test factories | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_checkpointer_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest_store.py` | 58 | Store test factories | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_store_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/memory_assert.py` | 56 | Immutability checkpoint saver | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_memory_assert_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/messages.py` | 28 | Message factory utilities | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_messages_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/__init__.py` | 0 | Middleware tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/__init__.py` | 0 | Core tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_composition.py` | 275 | Middleware composition tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_composition_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_decorators.py` | 757 | Decorator API tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_decorators_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_diagram.py` | 192 | Graph visualization tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_diagram_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_framework.py` | 1048 | Framework integration tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_framework_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_overrides.py` | 378 | Request immutability tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_overrides_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_sync_async_wrappers.py` | 426 | Sync/async interop tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_sync_async_wrappers_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_tools.py` | 338 | Tool management tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_tools_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_model_call.py` | 1271 | Model call interception tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_wrap_model_call_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_tool_call.py` | 808 | Tool call interception tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_wrap_tool_call_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/__init__.py` | 0 | Implementations marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_context_editing.py` | 451 | Context editing tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_context_editing_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_file_search.py` | 364 | File search security tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_file_search_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_human_in_the_loop.py` | 751 | Human approval tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_human_in_the_loop_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_call_limit.py` | 226 | Model limit tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_call_limit_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_fallback.py` | 357 | Model failover tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_fallback_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_retry.py` | 690 | Model retry tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_pii.py` | 638 | PII detection tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_pii_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_execution_policies.py` | 403 | Shell execution tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_shell_execution_policies_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_tool.py` | 556 | Shell tool tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_shell_tool_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_structured_output_retry.py` | 369 | Structured output retry tests | Workflow: Structured_Output_Workflow | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_structured_output_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_summarization.py` | 889 | Summarization tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_summarization_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_todo.py` | 520 | Todo middleware tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_todo_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py` | 797 | Tool limit tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_call_limit_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_emulator.py` | 627 | Tool emulation tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_emulator_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_retry.py` | 1007 | Tool retry tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_selection.py` | 596 | Tool selection tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_selection_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/model.py` | 109 | Fake chat model for tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_model_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_agent_name.py` | 99 | Agent name attribution tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_agent_name_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_create_agent_tool_validation.py` | 379 | Tool validation error tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_create_agent_tool_validation_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_injected_runtime_create_agent.py` | 831 | Runtime injection tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_injected_runtime_create_agent_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_react_agent.py` | 987 | React agent tests (disabled) | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_react_agent_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_response_format.py` | 875 | Structured output format tests | Workflow: Structured_Output_Workflow | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_response_format_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_response_format_integration.py` | 193 | OpenAI VCR integration tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_response_format_integration_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_responses.py` | 140 | Response component tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_responses_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_responses_spec.py` | 148 | Spec-driven response tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_responses_spec_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_graph.py` | 73 | Return direct graph tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_return_direct_graph_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_spec.py` | 107 | Return direct spec tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_return_direct_spec_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_state_schema.py` | 189 | Custom state schema tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_state_schema_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_system_message.py` | 1010 | System message tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_system_message_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/utils.py` | 21 | Test spec loading utilities | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_utils_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/chat_models/__init__.py` | 0 | Chat model tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/chat_models/test_chat_models.py` | 287 | Chat model factory tests | Workflow: Chat_Model_Initialization_Workflow | [→](./_files/libs_langchain_v1_tests_unit_tests_chat_models_test_chat_models_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/conftest.py` | 127 | Pytest config and fixtures | — | [→](./_files/libs_langchain_v1_tests_unit_tests_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/__init__.py` | 0 | Embeddings tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/test_base.py` | 111 | Embeddings factory tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/test_imports.py` | 10 | Embeddings API contract tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings_test_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_dependencies.py` | 39 | Dependency validation tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_dependencies_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_imports.py` | 56 | Public API import tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_pytest_config.py` | 9 | Socket blocking validation | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_pytest_config_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/tools/__init__.py` | 0 | Tools tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_tools___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/tools/test_imports.py` | 16 | Tools API contract tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_tools_test_imports_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/__init__.py` | 68 | Text splitters entry point | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters___init___py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/base.py` | 370 | Core splitter abstractions | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_base_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/character.py` | 793 | Character-based splitters | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_character_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/html.py` | 1006 | HTML structure splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_html_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/json.py` | 157 | JSON structure splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_json_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/jsx.py` | 102 | JSX/React code splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_jsx_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/konlpy.py` | 42 | Korean language splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_konlpy_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/latex.py` | 17 | LaTeX document splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_latex_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/markdown.py` | 468 | Markdown structure splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_markdown_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/nltk.py` | 59 | NLTK sentence tokenization | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_nltk_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/python.py` | 17 | Python code splitting | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_python_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/sentence_transformers.py` | 112 | SentenceTransformers token align | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_sentence_transformers_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/spacy.py` | 71 | spaCy sentence segmentation | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_langchain_text_splitters_spacy_py.md) |
| ✅ | `libs/text-splitters/scripts/check_imports.py` | 19 | Fast import validation | — | [→](./_files/libs_text-splitters_scripts_check_imports_py.md) |
| ✅ | `libs/text-splitters/tests/__init__.py` | 0 | Test package marker | — | [→](./_files/libs_text-splitters_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/__init__.py` | 0 | Integration tests marker | — | [→](./_files/libs_text-splitters_tests_integration_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_compile.py` | 6 | Compilation verification | — | [→](./_files/libs_text-splitters_tests_integration_tests_test_compile_py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_nlp_text_splitters.py` | 123 | NLP splitter tests | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_tests_integration_tests_test_nlp_text_splitters_py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_text_splitter.py` | 114 | Tokenizer splitter tests | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_tests_integration_tests_test_text_splitter_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/__init__.py` | 0 | Unit tests marker | — | [→](./_files/libs_text-splitters_tests_unit_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/conftest.py` | 86 | Pytest dependency markers | — | [→](./_files/libs_text-splitters_tests_unit_tests_conftest_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/test_html_security.py` | 130 | XXE attack prevention tests | — | [→](./_files/libs_text-splitters_tests_unit_tests_test_html_security_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/test_text_splitters.py` | 3881 | Comprehensive splitter tests | Workflow: Text_Splitting_Workflow | [→](./_files/libs_text-splitters_tests_unit_tests_test_text_splitters_py.md) |

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
