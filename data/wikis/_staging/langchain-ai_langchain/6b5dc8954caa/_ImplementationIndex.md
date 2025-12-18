# Implementation Index: langchain-ai_langchain

> Index of all Implementation pages in this wiki.

---

## Workflow-Connected Implementations

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| langchain-ai_langchain_init_chat_model | [→](./implementations/langchain-ai_langchain_init_chat_model.md) | ✅Principle:langchain-ai_langchain_Chat_Model_Initialization, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 1 |
| langchain-ai_langchain_BaseTool_and_StructuredTool | [→](./implementations/langchain-ai_langchain_BaseTool_and_StructuredTool.md) | ✅Principle:langchain-ai_langchain_Tool_Definition, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 2 |
| langchain-ai_langchain_AgentMiddleware_class | [→](./implementations/langchain-ai_langchain_AgentMiddleware_class.md) | ✅Principle:langchain-ai_langchain_Middleware_Composition, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 3 |
| langchain-ai_langchain_ResponseFormat_strategies | [→](./implementations/langchain-ai_langchain_ResponseFormat_strategies.md) | ✅Principle:langchain-ai_langchain_Structured_Output_Strategy, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 4 |
| langchain-ai_langchain_create_agent_graph_building | [→](./implementations/langchain-ai_langchain_create_agent_graph_building.md) | ✅Principle:langchain-ai_langchain_StateGraph_Assembly, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 5 |
| langchain-ai_langchain_CompiledStateGraph_invocation | [→](./implementations/langchain-ai_langchain_CompiledStateGraph_invocation.md) | ✅Principle:langchain-ai_langchain_Agent_Loop_Execution, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Agent_Creation Step 6 |
| langchain-ai_langchain_RecursiveCharacterTextSplitter | [→](./implementations/langchain-ai_langchain_RecursiveCharacterTextSplitter.md) | ✅Principle:langchain-ai_langchain_Splitter_Strategy_Selection, ✅Env:langchain-ai_langchain_Text_Splitters_Environment | Text_Splitting Step 1 |
| langchain-ai_langchain_TextSplitter_length_functions | [→](./implementations/langchain-ai_langchain_TextSplitter_length_functions.md) | ✅Principle:langchain-ai_langchain_Length_Function_Setup, ✅Env:langchain-ai_langchain_Text_Splitters_Environment | Text_Splitting Step 2 |
| langchain-ai_langchain_TextSplitter_init | [→](./implementations/langchain-ai_langchain_TextSplitter_init.md) | ✅Principle:langchain-ai_langchain_Chunk_Size_Configuration, ✅Env:langchain-ai_langchain_Text_Splitters_Environment | Text_Splitting Step 3 |
| langchain-ai_langchain_TextSplitter_split_methods | [→](./implementations/langchain-ai_langchain_TextSplitter_split_methods.md) | ✅Principle:langchain-ai_langchain_Document_Transformation, ✅Env:langchain-ai_langchain_Text_Splitters_Environment | Text_Splitting Step 4 |
| langchain-ai_langchain_TextSplitter_create_documents | [→](./implementations/langchain-ai_langchain_TextSplitter_create_documents.md) | ✅Principle:langchain-ai_langchain_Metadata_Handling, ✅Env:langchain-ai_langchain_Text_Splitters_Environment | Text_Splitting Step 5 |
| langchain-ai_langchain_model_parsing_functions | [→](./implementations/langchain-ai_langchain_model_parsing_functions.md) | ✅Principle:langchain-ai_langchain_Model_Identifier_Parsing, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat_Model_Init Step 1 |
| langchain-ai_langchain_check_pkg | [→](./implementations/langchain-ai_langchain_check_pkg.md) | ✅Principle:langchain-ai_langchain_Provider_Package_Validation, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat_Model_Init Step 2 |
| langchain-ai_langchain_init_chat_model_helper | [→](./implementations/langchain-ai_langchain_init_chat_model_helper.md) | ✅Principle:langchain-ai_langchain_Model_Instantiation, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat_Model_Init Step 3 |
| langchain-ai_langchain_ConfigurableModel_class | [→](./implementations/langchain-ai_langchain_ConfigurableModel_class.md) | ✅Principle:langchain-ai_langchain_Configurable_Model_Setup, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat_Model_Init Step 4 |
| langchain-ai_langchain_ConfigurableModel_declarative_methods | [→](./implementations/langchain-ai_langchain_ConfigurableModel_declarative_methods.md) | ✅Principle:langchain-ai_langchain_Declarative_Operation_Binding, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat_Model_Init Step 5 |
| langchain-ai_langchain_SchemaSpec_class | [→](./implementations/langchain-ai_langchain_SchemaSpec_class.md) | ✅Principle:langchain-ai_langchain_Schema_Definition, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 1 |
| langchain-ai_langchain_ResponseFormat_type_union | [→](./implementations/langchain-ai_langchain_ResponseFormat_type_union.md) | ✅Principle:langchain-ai_langchain_Strategy_Selection, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 2 |
| langchain-ai_langchain_OutputToolBinding_class | [→](./implementations/langchain-ai_langchain_OutputToolBinding_class.md) | ✅Principle:langchain-ai_langchain_Output_Tool_Binding, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 3 |
| langchain-ai_langchain_agent_model_binding | [→](./implementations/langchain-ai_langchain_agent_model_binding.md) | ✅Principle:langchain-ai_langchain_Model_Invocation_With_Schema, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 4 |
| langchain-ai_langchain_parse_with_schema | [→](./implementations/langchain-ai_langchain_parse_with_schema.md) | ✅Principle:langchain-ai_langchain_Response_Parsing, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 5 |
| langchain-ai_langchain_structured_output_error_classes | [→](./implementations/langchain-ai_langchain_structured_output_error_classes.md) | ✅Principle:langchain-ai_langchain_Structured_Output_Error_Handling, ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Structured_Output Step 6 |

---

## Orphan Implementations (No Workflow Connection)

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| langchain-ai_langchain_LLMChain | [→](./implementations/langchain-ai_langchain_LLMChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Deprecated since 0.1.17 |
| langchain-ai_langchain_SequentialChain | [→](./implementations/langchain-ai_langchain_SequentialChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Deprecated, use LCEL |
| langchain-ai_langchain_MapReduceChain | [→](./implementations/langchain-ai_langchain_MapReduceChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Deprecated since 0.2.13 |
| langchain-ai_langchain_TransformChain | [→](./implementations/langchain-ai_langchain_TransformChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Custom function wrapper |
| langchain-ai_langchain_Chain_base | [→](./implementations/langchain-ai_langchain_Chain_base.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Abstract Chain class |
| langchain-ai_langchain_chain_loading | [→](./implementations/langchain-ai_langchain_chain_loading.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Chain serialization |
| langchain-ai_langchain_BaseMemory | [→](./implementations/langchain-ai_langchain_BaseMemory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Deprecated memory base |
| langchain-ai_langchain_ConversationBufferMemory | [→](./implementations/langchain-ai_langchain_ConversationBufferMemory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Deprecated, use LangGraph |
| langchain-ai_langchain_ConversationEntityMemory | [→](./implementations/langchain-ai_langchain_ConversationEntityMemory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Entity extraction memory |
| langchain-ai_langchain_ConversationSummaryMemory | [→](./implementations/langchain-ai_langchain_ConversationSummaryMemory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | LLM-summarized memory |
| langchain-ai_langchain_ReadOnlySharedMemory | [→](./implementations/langchain-ai_langchain_ReadOnlySharedMemory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Read-only memory wrapper |
| langchain-ai_langchain_BooleanOutputParser | [→](./implementations/langchain-ai_langchain_BooleanOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Yes/No extraction |
| langchain-ai_langchain_CombiningOutputParser | [→](./implementations/langchain-ai_langchain_CombiningOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Multi-parser orchestration |
| langchain-ai_langchain_DatetimeOutputParser | [→](./implementations/langchain-ai_langchain_DatetimeOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Temporal extraction |
| langchain-ai_langchain_EnumOutputParser | [→](./implementations/langchain-ai_langchain_EnumOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Constrained choice |
| langchain-ai_langchain_OutputFixingParser | [→](./implementations/langchain-ai_langchain_OutputFixingParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | LLM-powered fixing |
| langchain-ai_langchain_PandasDataFrameOutputParser | [→](./implementations/langchain-ai_langchain_PandasDataFrameOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | DataFrame query |
| langchain-ai_langchain_RegexParser | [→](./implementations/langchain-ai_langchain_RegexParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Pattern extraction |
| langchain-ai_langchain_RegexDictParser | [→](./implementations/langchain-ai_langchain_RegexDictParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Key-value extraction |
| langchain-ai_langchain_RetryOutputParser | [→](./implementations/langchain-ai_langchain_RetryOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Context-aware retry |
| langchain-ai_langchain_StructuredOutputParser | [→](./implementations/langchain-ai_langchain_StructuredOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Schema-based dict |
| langchain-ai_langchain_YamlOutputParser | [→](./implementations/langchain-ai_langchain_YamlOutputParser.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | YAML format |
| langchain-ai_langchain_format_instructions | [→](./implementations/langchain-ai_langchain_format_instructions.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Parser templates |
| langchain-ai_langchain_hub | [→](./implementations/langchain-ai_langchain_hub.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | LangChain Hub |
| langchain-ai_langchain_ModelLaboratory | [→](./implementations/langchain-ai_langchain_ModelLaboratory.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Model testing utility |
| langchain-ai_langchain_OpenAIModerationChain | [→](./implementations/langchain-ai_langchain_OpenAIModerationChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Content filtering |
| langchain-ai_langchain_ConditionalPromptSelector | [→](./implementations/langchain-ai_langchain_ConditionalPromptSelector.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Dynamic prompts |
| langchain-ai_langchain_create_retrieval_chain | [→](./implementations/langchain-ai_langchain_create_retrieval_chain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | RAG pipeline factory |
| langchain-ai_langchain_create_history_aware_retriever | [→](./implementations/langchain-ai_langchain_create_history_aware_retriever.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chat history RAG |
| langchain-ai_langchain_create_stuff_documents_chain | [→](./implementations/langchain-ai_langchain_create_stuff_documents_chain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Doc concatenation |
| langchain-ai_langchain_load_summarize_chain | [→](./implementations/langchain-ai_langchain_load_summarize_chain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Chain factory |
| langchain-ai_langchain_generate_example | [→](./implementations/langchain-ai_langchain_generate_example.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Example generation |
| langchain-ai_langchain_init_embeddings | [→](./implementations/langchain-ai_langchain_init_embeddings.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Embeddings factory |
| langchain-ai_langchain_langchain_classic_init | [→](./implementations/langchain-ai_langchain_langchain_classic_init.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Package entry |
| langchain-ai_langchain_schema_init | [→](./implementations/langchain-ai_langchain_schema_init.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | Schema re-exports |
| langchain-ai_langchain_BaseCombineDocumentsChain | [→](./implementations/langchain-ai_langchain_BaseCombineDocumentsChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Doc chain base |
| langchain-ai_langchain_ReduceDocumentsChain | [→](./implementations/langchain-ai_langchain_ReduceDocumentsChain.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment, ✅Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic | Recursive reduction |
| langchain-ai_langchain_check_diff | [→](./implementations/langchain-ai_langchain_check_diff.md) | ✅Env:langchain-ai_langchain_LangChain_Runtime_Environment | CI test optimization |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

## Summary

- **Total Implementations:** 60
- **Workflow-Connected:** 22
- **Orphan Implementations:** 38
