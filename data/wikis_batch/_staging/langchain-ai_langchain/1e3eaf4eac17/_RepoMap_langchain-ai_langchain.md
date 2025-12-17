# Repository Map: langchain-ai_langchain

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/langchain-ai/langchain |
| Branch | main |
| Generated | 2025-12-17 18:59 |
| Python Files | 200 |
| Total Lines | 46,629 |
| Explored | 200/200 |

## Structure


📖 README: `README.md`

---

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.github/scripts/check_diff.py` | 340 | CI test matrix generator | Impl:check_diff | [→](./_files/_github_scripts_check_diff_py.md) |
| ✅ | `.github/scripts/check_prerelease_dependencies.py` | 36 | Prerelease dependency validator | — | [→](./_files/_github_scripts_check_prerelease_dependencies_py.md) |
| ✅ | `.github/scripts/get_min_versions.py` | 199 | Minimum version resolver | — | [→](./_files/_github_scripts_get_min_versions_py.md) |
| ✅ | `libs/langchain/langchain_classic/__init__.py` | 424 | Backwards compatibility import manager | Impl:langchain_classic_init | [→](./_files/libs_langchain_langchain_classic___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/base_language.py` | 7 | BaseLanguageModel compatibility re-export | — | [→](./_files/libs_langchain_langchain_classic_base_language_py.md) |
| ✅ | `libs/langchain/langchain_classic/base_memory.py` | 116 | Deprecated memory abstraction base | Impl:BaseMemory | [→](./_files/libs_langchain_langchain_classic_base_memory_py.md) |
| ✅ | `libs/langchain/langchain_classic/cache.py` | 72 | Cache implementations compatibility layer | — | [→](./_files/libs_langchain_langchain_classic_cache_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/__init__.py` | 96 | Lazy import chain orchestrator | — | [→](./_files/libs_langchain_langchain_classic_chains___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/base.py` | 806 | Abstract chain base class | Impl:Chain | [→](./_files/libs_langchain_langchain_classic_chains_base_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/example_generator.py` | 22 | LLM-based example generator | Impl:generate_example | [→](./_files/libs_langchain_langchain_classic_chains_example_generator_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/history_aware_retriever.py` | 68 | Conversational query reformulation chain | Impl:create_history_aware_retriever | [→](./_files/libs_langchain_langchain_classic_chains_history_aware_retriever_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/llm.py` | 432 | Deprecated prompt-LLM chain class | Impl:LLMChain | [→](./_files/libs_langchain_langchain_classic_chains_llm_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/llm_requests.py` | 23 | Community migration compatibility shim | — | [→](./_files/libs_langchain_langchain_classic_chains_llm_requests_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/loading.py` | 742 | Deprecated chain serialization system | Impl:load_chain | [→](./_files/libs_langchain_langchain_classic_chains_loading_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/mapreduce.py` | 117 | Deprecated document splitting chain | Impl:MapReduceChain | [→](./_files/libs_langchain_langchain_classic_chains_mapreduce_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/moderation.py` | 129 | OpenAI content moderation chain | Impl:OpenAIModerationChain | [→](./_files/libs_langchain_langchain_classic_chains_moderation_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/prompt_selector.py` | 65 | Model-aware prompt selection utility | Impl:ConditionalPromptSelector | [→](./_files/libs_langchain_langchain_classic_chains_prompt_selector_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/retrieval.py` | 68 | Modern RAG chain builder | Impl:create_retrieval_chain | [→](./_files/libs_langchain_langchain_classic_chains_retrieval_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/sequential.py` | 208 | Multi-stage chain pipeline orchestrator | Impl:SequentialChain | [→](./_files/libs_langchain_langchain_classic_chains_sequential_py.md) |
| ✅ | `libs/langchain/langchain_classic/chains/transform.py` | 79 | Custom function chain wrapper | Impl:TransformChain | [→](./_files/libs_langchain_langchain_classic_chains_transform_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/__init__.py` | 6 | Module documentation for chat loaders | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/base.py` | 3 | Re-exports BaseChatLoader abstract interface | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_base_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py` | 32 | Facebook Messenger deprecated import shims | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_facebook_messenger_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/gmail.py` | 23 | Gmail loader deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_gmail_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/imessage.py` | 23 | iMessage loader deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_imessage_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/langsmith.py` | 30 | LangSmith loaders deprecated import shims | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_langsmith_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/slack.py` | 23 | Slack loader deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_slack_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/telegram.py` | 23 | Telegram loader deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_telegram_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/utils.py` | 36 | Chat utilities deprecated import shims | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_utils_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_loaders/whatsapp.py` | 23 | WhatsApp loader deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_loaders_whatsapp_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/baidu_qianfan_endpoint.py` | 27 | Baidu Qianfan deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_models_baidu_qianfan_endpoint_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/everlyai.py` | 23 | EverlyAI model deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_models_everlyai_py.md) |
| ✅ | `libs/langchain/langchain_classic/chat_models/tongyi.py` | 23 | Tongyi model deprecated import shim | — | [→](./_files/libs_langchain_langchain_classic_chat_models_tongyi_py.md) |
| ✅ | `libs/langchain/langchain_classic/env.py` | 17 | Runtime environment information collector | — | [→](./_files/libs_langchain_langchain_classic_env_py.md) |
| ✅ | `libs/langchain/langchain_classic/example_generator.py` | 5 | Backwards compatibility for example generation | — | [→](./_files/libs_langchain_langchain_classic_example_generator_py.md) |
| ✅ | `libs/langchain/langchain_classic/formatting.py` | 5 | Deprecated text formatting re-exports | — | [→](./_files/libs_langchain_langchain_classic_formatting_py.md) |
| ✅ | `libs/langchain/langchain_classic/globals.py` | 19 | Global configuration re-exports | — | [→](./_files/libs_langchain_langchain_classic_globals_py.md) |
| ✅ | `libs/langchain/langchain_classic/hub.py` | 153 | LangChain Hub push/pull interface | Impl:Hub | [→](./_files/libs_langchain_langchain_classic_hub_py.md) |
| ✅ | `libs/langchain/langchain_classic/input.py` | 15 | Terminal text formatting utilities re-export | — | [→](./_files/libs_langchain_langchain_classic_input_py.md) |
| ✅ | `libs/langchain/langchain_classic/model_laboratory.py` | 98 | LLM model comparison utility tool | Impl:ModelLaboratory | [→](./_files/libs_langchain_langchain_classic_model_laboratory_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/__init__.py` | 82 | Output parser module exports | — | [→](./_files/libs_langchain_langchain_classic_output_parsers___init___py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/boolean.py` | 54 | Parse text to boolean | Impl:BooleanOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_boolean_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/combining.py` | 58 | Combine multiple output parsers | Impl:CombiningOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_combining_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/datetime.py` | 58 | Parse text to datetime | Impl:DatetimeOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_datetime_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/enum.py` | 45 | Parse text to enum | Impl:EnumOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_enum_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/ernie_functions.py` | 45 | Ernie functions deprecation redirects | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_ernie_functions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/fix.py` | 156 | Auto-fix parsing errors | Impl:OutputFixingParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_fix_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/format_instructions.py` | 79 | Parser format instruction templates | Impl:format_instructions | [→](./_files/libs_langchain_langchain_classic_output_parsers_format_instructions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/json.py` | 15 | JSON parsing utilities re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_json_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/list.py` | 13 | List parsers re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_list_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/loading.py` | 22 | Dynamic parser configuration loading | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_loading_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/openai_functions.py` | 13 | OpenAI functions parsers re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_openai_functions_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/openai_tools.py` | 7 | OpenAI tools parsers re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_openai_tools_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py` | 171 | Parse DataFrame query strings | Impl:PandasDataFrameOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_pandas_dataframe_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/prompts.py` | 21 | Retry parser prompt templates | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_prompts_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/pydantic.py` | 3 | Pydantic parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_pydantic_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/rail_parser.py` | 25 | Guardrails parser deprecation redirect | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_rail_parser_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/regex.py` | 40 | Regex pattern output parser | Impl:RegexParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_regex_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/regex_dict.py` | 42 | Regex dictionary output parser | Impl:RegexDictParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_regex_dict_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/retry.py` | 315 | Retry failed parsing attempts | Impl:RetryOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_retry_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/structured.py` | 116 | Schema-based structured output parser | Impl:StructuredOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_structured_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/xml.py` | 3 | XML parser re-export | — | [→](./_files/libs_langchain_langchain_classic_output_parsers_xml_py.md) |
| ✅ | `libs/langchain/langchain_classic/output_parsers/yaml.py` | 69 | YAML to Pydantic parser | Impl:YamlOutputParser | [→](./_files/libs_langchain_langchain_classic_output_parsers_yaml_py.md) |
| ✅ | `libs/langchain/langchain_classic/python.py` | 19 | Deprecated PythonREPL proxy shim | — | [→](./_files/libs_langchain_langchain_classic_python_py.md) |
| ✅ | `libs/langchain/langchain_classic/requests.py` | 35 | HTTP request wrappers re-export | — | [→](./_files/libs_langchain_langchain_classic_requests_py.md) |
| ✅ | `libs/langchain/langchain_classic/serpapi.py` | 25 | SerpAPI integration compatibility proxy | — | [→](./_files/libs_langchain_langchain_classic_serpapi_py.md) |
| ✅ | `libs/langchain/langchain_classic/sql_database.py` | 25 | SQL database utility re-export | — | [→](./_files/libs_langchain_langchain_classic_sql_database_py.md) |
| ✅ | `libs/langchain/langchain_classic/text_splitter.py` | 50 | Text splitter compatibility re-exports | — | [→](./_files/libs_langchain_langchain_classic_text_splitter_py.md) |
| ✅ | `libs/langchain/scripts/check_imports.py` | 33 | Fast Python import validation script | — | [→](./_files/libs_langchain_scripts_check_imports_py.md) |
| ✅ | `libs/langchain/tests/__init__.py` | 1 | Test package initialization marker | — | [→](./_files/libs_langchain_tests___init___py.md) |
| ✅ | `libs/langchain/tests/data.py` | 12 | Test PDF file path definitions | — | [→](./_files/libs_langchain_tests_data_py.md) |
| ✅ | `libs/langchain_v1/langchain/__init__.py` | 3 | Package version entrypoint | — | [→](./_files/libs_langchain_v1_langchain___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/__init__.py` | 9 | Agents module public API | — | [→](./_files/libs_langchain_v1_langchain_agents___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/factory.py` | 1682 | Agent factory implementation | Workflow: Agent_Creation_and_Execution, Middleware_Composition | [→](./_files/libs_langchain_v1_langchain_agents_factory_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/__init__.py` | 78 | Middleware module public API | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_execution.py` | 389 | Shell command execution policies | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware__execution_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_redaction.py` | 364 | PII detection and redaction | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware__redaction_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/_retry.py` | 123 | Shared retry logic utilities | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware__retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/context_editing.py` | 278 | Context window management middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_context_editing_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/file_search.py` | 387 | Filesystem search tools middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_file_search_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py` | 357 | Human approval workflow middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_human_in_the_loop_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_call_limit.py` | 256 | Model call quota enforcement | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_call_limit_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_fallback.py` | 135 | Automatic model failover middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_fallback_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/model_retry.py` | 300 | Model call retry middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_model_retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/pii.py` | 369 | PII detection and sanitization | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_pii_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/shell_tool.py` | 760 | Persistent shell session middleware | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_shell_tool_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/summarization.py` | 535 | Conversation history auto-summarization | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_summarization_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/todo.py` | 224 | Task tracking and planning | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_todo_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py` | 488 | Tool execution limit enforcement | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_call_limit_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_emulator.py` | 209 | LLM-based tool testing emulator | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_emulator_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_retry.py` | 396 | Automatic tool retry logic | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_retry_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/tool_selection.py` | 320 | LLM-powered tool filtering | — | [→](./_files/libs_langchain_v1_langchain_agents_middleware_tool_selection_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/middleware/types.py` | 1848 | Middleware framework type system | Workflow: Middleware_Composition | [→](./_files/libs_langchain_v1_langchain_agents_middleware_types_py.md) |
| ✅ | `libs/langchain_v1/langchain/agents/structured_output.py` | 443 | Structured output strategies | Workflow: Agent_Creation_and_Execution | [→](./_files/libs_langchain_v1_langchain_agents_structured_output_py.md) |
| ✅ | `libs/langchain_v1/langchain/chat_models/__init__.py` | 7 | Chat models public API | — | [→](./_files/libs_langchain_v1_langchain_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/chat_models/base.py` | 944 | Chat model factory | Workflow: Chat_Model_Initialization, Agent_Creation_and_Execution | [→](./_files/libs_langchain_v1_langchain_chat_models_base_py.md) |
| ✅ | `libs/langchain_v1/langchain/embeddings/__init__.py` | 17 | Embeddings module public API | — | [→](./_files/libs_langchain_v1_langchain_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/embeddings/base.py` | 245 | Embeddings factory implementation | — | [→](./_files/libs_langchain_v1_langchain_embeddings_base_py.md) |
| ✅ | `libs/langchain_v1/langchain/messages/__init__.py` | 73 | Message types public API | — | [→](./_files/libs_langchain_v1_langchain_messages___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/rate_limiters/__init__.py` | 13 | Rate limiter public API | — | [→](./_files/libs_langchain_v1_langchain_rate_limiters___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/tools/__init__.py` | 22 | Tools module public API | — | [→](./_files/libs_langchain_v1_langchain_tools___init___py.md) |
| ✅ | `libs/langchain_v1/langchain/tools/tool_node.py` | 20 | Tool node compatibility layer | — | [→](./_files/libs_langchain_v1_langchain_tools_tool_node_py.md) |
| ✅ | `libs/langchain_v1/scripts/check_imports.py` | 33 | Import validation script | — | [→](./_files/libs_langchain_v1_scripts_check_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/__init__.py` | 1 | Test package marker file | — | [→](./_files/libs_langchain_v1_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/__init__.py` | 1 | Integration tests package marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/__init__.py` | 1 | Agent integration tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/middleware/__init__.py` | 1 | Middleware integration tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/agents/middleware/test_shell_tool_integration.py` | 147 | Shell tool middleware tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_agents_middleware_test_shell_tool_integration_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/cache/__init__.py` | 1 | Cache integration tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_cache___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/cache/fake_embeddings.py` | 91 | Fake embeddings test utility | — | [→](./_files/libs_langchain_v1_tests_integration_tests_cache_fake_embeddings_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/chat_models/__init__.py` | 0 | Chat models tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/chat_models/test_base.py` | 57 | Chat model initialization tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_chat_models_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/conftest.py` | 34 | Integration tests configuration setup | — | [→](./_files/libs_langchain_v1_tests_integration_tests_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/embeddings/__init__.py` | 0 | Embeddings integration tests marker | — | [→](./_files/libs_langchain_v1_tests_integration_tests_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/embeddings/test_base.py` | 44 | Embeddings initialization tests | — | [→](./_files/libs_langchain_v1_tests_integration_tests_embeddings_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/integration_tests/test_compile.py` | 6 | Test compilation placeholder | — | [→](./_files/libs_langchain_v1_tests_integration_tests_test_compile_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/__init__.py` | 0 | Unit tests package marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/__init__.py` | 0 | Agent unit tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/any_str.py` | 19 | Flexible string matching utility | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_any_str_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest.py` | 194 | Agent tests fixture configuration | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest_checkpointer.py` | 64 | Checkpointer factory functions | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_checkpointer_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/conftest_store.py` | 58 | Store factory functions | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_conftest_store_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/memory_assert.py` | 56 | Checkpoint immutability validator | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_memory_assert_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/messages.py` | 28 | Test message helper utilities | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_messages_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/__init__.py` | 0 | Middleware unit tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/__init__.py` | 0 | Core middleware tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_composition.py` | 275 | Handler composition chaining tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_composition_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_decorators.py` | 757 | Decorator API comprehensive validation | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_decorators_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_diagram.py` | 192 | Graph diagram snapshot testing | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_diagram_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_framework.py` | 1048 | Integration testing framework validation | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_framework_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_overrides.py` | 378 | Request immutability override tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_overrides_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_sync_async_wrappers.py` | 426 | Tool call sync/async compatibility | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_sync_async_wrappers_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_tools.py` | 338 | Tool modification and filtering | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_tools_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_model_call.py` | 1271 | Model call wrapping patterns | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_wrap_model_call_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_tool_call.py` | 808 | Tool call interception decorator | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_core_test_wrap_tool_call_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/__init__.py` | 0 | Middleware implementations tests marker | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_context_editing.py` | 451 | Token management middleware tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_context_editing_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_file_search.py` | 364 | File search security tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_file_search_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_human_in_the_loop.py` | 751 | Human approval workflow tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_human_in_the_loop_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_call_limit.py` | 226 | Model call limit enforcement | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_call_limit_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_fallback.py` | 357 | Model fallback reliability tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_fallback_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_retry.py` | 690 | Retry with exponential backoff | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_model_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_pii.py` | 638 | PII detection and protection | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_pii_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_execution_policies.py` | 403 | Shell execution policy testing | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_shell_execution_policies_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_tool.py` | 556 | Persistent shell session testing | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_shell_tool_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_structured_output_retry.py` | 369 | Structured output retry pattern | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_structured_output_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_summarization.py` | 889 | Conversation context summarization | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_summarization_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_todo.py` | 520 | Task planning middleware | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_todo_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py` | 797 | Tool call budget enforcement | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_call_limit_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_emulator.py` | 627 | LLM-based tool mocking | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_emulator_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_retry.py` | 1007 | Automatic tool retry logic | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_retry_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_selection.py` | 596 | Dynamic tool filtering | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_middleware_implementations_test_tool_selection_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/model.py` | 109 | Mock tool-calling model fixture | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_model_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_agent_name.py` | 99 | Agent name propagation tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_agent_name_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_create_agent_tool_validation.py` | 379 | Tool validation error filtering | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_create_agent_tool_validation_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_injected_runtime_create_agent.py` | 831 | ToolRuntime dependency injection tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_injected_runtime_create_agent_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_react_agent.py` | 987 | Comprehensive agent tests (commented) | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_react_agent_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_response_format.py` | 875 | Structured output format testing | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_response_format_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_response_format_integration.py` | 193 | OpenAI structured output integration | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_response_format_integration_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_responses.py` | 140 | ToolStrategy and OutputToolBinding tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_responses_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_responses_spec.py` | 148 | Spec-driven response integration tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_responses_spec_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_graph.py` | 73 | Graph structure validation tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_return_direct_graph_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_spec.py` | 107 | Return-direct polling integration tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_return_direct_spec_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_state_schema.py` | 189 | Custom state schema tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_state_schema_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/test_system_message.py` | 1010 | System message middleware tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_test_system_message_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/agents/utils.py` | 21 | JSON test specification loader | — | [→](./_files/libs_langchain_v1_tests_unit_tests_agents_utils_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/chat_models/__init__.py` | 0 | Python package marker file | — | [→](./_files/libs_langchain_v1_tests_unit_tests_chat_models___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/chat_models/test_chat_models.py` | 287 | Chat model initialization tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_chat_models_test_chat_models_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/conftest.py` | 127 | Pytest configuration and fixtures | — | [→](./_files/libs_langchain_v1_tests_unit_tests_conftest_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/__init__.py` | 0 | Python package marker file | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/test_base.py` | 111 | Embeddings parsing and validation | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings_test_base_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/embeddings/test_imports.py` | 10 | Embeddings API contract test | — | [→](./_files/libs_langchain_v1_tests_unit_tests_embeddings_test_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_dependencies.py` | 39 | Required dependencies guard test | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_dependencies_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_imports.py` | 56 | Comprehensive import smoke tests | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_imports_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/test_pytest_config.py` | 9 | Network blocking verification test | — | [→](./_files/libs_langchain_v1_tests_unit_tests_test_pytest_config_py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/tools/__init__.py` | 0 | Python package marker file | — | [→](./_files/libs_langchain_v1_tests_unit_tests_tools___init___py.md) |
| ✅ | `libs/langchain_v1/tests/unit_tests/tools/test_imports.py` | 16 | Tools API contract test | — | [→](./_files/libs_langchain_v1_tests_unit_tests_tools_test_imports_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/__init__.py` | 68 | Text splitters public API | — | [→](./_files/libs_text-splitters_langchain_text_splitters___init___py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/base.py` | 370 | Core text splitting abstractions | Workflow: Text_Splitting_for_RAG | [→](./_files/libs_text-splitters_langchain_text_splitters_base_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/character.py` | 793 | Recursive character-based splitting | Workflow: Text_Splitting_for_RAG | [→](./_files/libs_text-splitters_langchain_text_splitters_character_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/html.py` | 1006 | HTML structure-preserving splitters | — | [→](./_files/libs_text-splitters_langchain_text_splitters_html_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/json.py` | 157 | Hierarchical JSON chunk splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_json_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/jsx.py` | 102 | React component boundary splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_jsx_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/konlpy.py` | 42 | Korean language sentence splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_konlpy_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/latex.py` | 17 | LaTeX document section splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_latex_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/markdown.py` | 468 | Markdown header-aware splitting | — | [→](./_files/libs_text-splitters_langchain_text_splitters_markdown_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/nltk.py` | 59 | NLTK linguistic sentence splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_nltk_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/python.py` | 17 | Python syntax-aware code splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_python_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/sentence_transformers.py` | 112 | Embedding model token splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_sentence_transformers_py.md) |
| ✅ | `libs/text-splitters/langchain_text_splitters/spacy.py` | 71 | spaCy multilingual sentence splitter | — | [→](./_files/libs_text-splitters_langchain_text_splitters_spacy_py.md) |
| ✅ | `libs/text-splitters/scripts/check_imports.py` | 19 | Dynamic import validation script | — | [→](./_files/libs_text-splitters_scripts_check_imports_py.md) |
| ✅ | `libs/text-splitters/tests/__init__.py` | 0 | Test package marker file | — | [→](./_files/libs_text-splitters_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/__init__.py` | 0 | Integration test package marker | — | [→](./_files/libs_text-splitters_tests_integration_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_compile.py` | 6 | Compilation validation placeholder test | — | [→](./_files/libs_text-splitters_tests_integration_tests_test_compile_py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_nlp_text_splitters.py` | 123 | NLTK and Spacy splitter tests | — | [→](./_files/libs_text-splitters_tests_integration_tests_test_nlp_text_splitters_py.md) |
| ✅ | `libs/text-splitters/tests/integration_tests/test_text_splitter.py` | 114 | Tokenizer integration tests suite | — | [→](./_files/libs_text-splitters_tests_integration_tests_test_text_splitter_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/__init__.py` | 0 | Unit test package marker | — | [→](./_files/libs_text-splitters_tests_unit_tests___init___py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/conftest.py` | 86 | Pytest configuration and markers | — | [→](./_files/libs_text-splitters_tests_unit_tests_conftest_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/test_html_security.py` | 130 | XXE vulnerability prevention tests | — | [→](./_files/libs_text-splitters_tests_unit_tests_test_html_security_py.md) |
| ✅ | `libs/text-splitters/tests/unit_tests/test_text_splitters.py` | 3881 | Comprehensive text splitter tests | — | [→](./_files/libs_text-splitters_tests_unit_tests_test_text_splitters_py.md) |

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
