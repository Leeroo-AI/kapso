# Principle Index: langchain-ai_langchain

> Index of all Principle pages in this wiki.

---

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| langchain-ai_langchain_Agent_Loop_Execution | [→](./principles/langchain-ai_langchain_Agent_Loop_Execution.md) | ✅Impl:langchain-ai_langchain_CompiledStateGraph_invocation, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Agent execution loop |
| langchain-ai_langchain_Chat_Model_Initialization | [→](./principles/langchain-ai_langchain_Chat_Model_Initialization.md) | ✅Impl:langchain-ai_langchain_init_chat_model, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Chat model factory |
| langchain-ai_langchain_Chunk_Size_Configuration | [→](./principles/langchain-ai_langchain_Chunk_Size_Configuration.md) | ✅Impl:langchain-ai_langchain_TextSplitter_init, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | Chunk size setup |
| langchain-ai_langchain_Configurable_Model_Setup | [→](./principles/langchain-ai_langchain_Configurable_Model_Setup.md) | ✅Impl:langchain-ai_langchain_ConfigurableModel_class, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow | Runtime configurable models |
| langchain-ai_langchain_Declarative_Operation_Binding | [→](./principles/langchain-ai_langchain_Declarative_Operation_Binding.md) | ✅Impl:langchain-ai_langchain_ConfigurableModel_declarative_methods, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow | bind_tools, with_structured_output |
| langchain-ai_langchain_Document_Transformation | [→](./principles/langchain-ai_langchain_Document_Transformation.md) | ✅Impl:langchain-ai_langchain_TextSplitter_split_methods, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | Split text/documents |
| langchain-ai_langchain_Length_Function_Setup | [→](./principles/langchain-ai_langchain_Length_Function_Setup.md) | ✅Impl:langchain-ai_langchain_TextSplitter_length_functions, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | Token counting |
| langchain-ai_langchain_Metadata_Handling | [→](./principles/langchain-ai_langchain_Metadata_Handling.md) | ✅Impl:langchain-ai_langchain_TextSplitter_create_documents, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | Metadata preservation |
| langchain-ai_langchain_Middleware_Composition | [→](./principles/langchain-ai_langchain_Middleware_Composition.md) | ✅Impl:langchain-ai_langchain_AgentMiddleware_class, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Agent middleware hooks |
| langchain-ai_langchain_Model_Identifier_Parsing | [→](./principles/langchain-ai_langchain_Model_Identifier_Parsing.md) | ✅Impl:langchain-ai_langchain_model_parsing_functions, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow | Parse model strings |
| langchain-ai_langchain_Model_Instantiation | [→](./principles/langchain-ai_langchain_Model_Instantiation.md) | ✅Impl:langchain-ai_langchain_init_chat_model_helper, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow | Create model instances |
| langchain-ai_langchain_Model_Invocation_With_Schema | [→](./principles/langchain-ai_langchain_Model_Invocation_With_Schema.md) | ✅Impl:langchain-ai_langchain_agent_model_binding, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Model binding for structured output |
| langchain-ai_langchain_Output_Tool_Binding | [→](./principles/langchain-ai_langchain_Output_Tool_Binding.md) | ✅Impl:langchain-ai_langchain_OutputToolBinding_class, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Structured output as tools |
| langchain-ai_langchain_Provider_Package_Validation | [→](./principles/langchain-ai_langchain_Provider_Package_Validation.md) | ✅Impl:langchain-ai_langchain_check_pkg, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow | Check provider packages |
| langchain-ai_langchain_Response_Parsing | [→](./principles/langchain-ai_langchain_Response_Parsing.md) | ✅Impl:langchain-ai_langchain_parse_with_schema, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Parse structured responses |
| langchain-ai_langchain_Schema_Definition | [→](./principles/langchain-ai_langchain_Schema_Definition.md) | ✅Impl:langchain-ai_langchain_SchemaSpec_class, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Schema specification |
| langchain-ai_langchain_Splitter_Strategy_Selection | [→](./principles/langchain-ai_langchain_Splitter_Strategy_Selection.md) | ✅Impl:langchain-ai_langchain_RecursiveCharacterTextSplitter, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | Choose text splitter |
| langchain-ai_langchain_StateGraph_Assembly | [→](./principles/langchain-ai_langchain_StateGraph_Assembly.md) | ✅Impl:langchain-ai_langchain_create_agent_graph_building, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Build agent graph |
| langchain-ai_langchain_Strategy_Selection | [→](./principles/langchain-ai_langchain_Strategy_Selection.md) | ✅Impl:langchain-ai_langchain_ResponseFormat_type_union, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Choose output strategy |
| langchain-ai_langchain_Structured_Output_Error_Handling | [→](./principles/langchain-ai_langchain_Structured_Output_Error_Handling.md) | ✅Impl:langchain-ai_langchain_structured_output_error_classes, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow | Error handling |
| langchain-ai_langchain_Structured_Output_Strategy | [→](./principles/langchain-ai_langchain_Structured_Output_Strategy.md) | ✅Impl:langchain-ai_langchain_ResponseFormat_strategies, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Output strategies |
| langchain-ai_langchain_Tool_Definition | [→](./principles/langchain-ai_langchain_Tool_Definition.md) | ✅Impl:langchain-ai_langchain_BaseTool_and_StructuredTool, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow | Tool definitions |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

## Summary

- **Total Principles:** 22
- **By Workflow:**
  - Agent_Creation_Workflow: 6
  - Text_Splitting_Workflow: 5
  - Chat_Model_Initialization_Workflow: 5
  - Structured_Output_Workflow: 6
