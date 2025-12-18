# Heuristic Index: langchain-ai_langchain

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| langchain-ai_langchain_Chunk_Size_Selection | [→](./heuristics/langchain-ai_langchain_Chunk_Size_Selection.md) | ✅Impl:langchain-ai_langchain_TextSplitter_init, ✅Impl:langchain-ai_langchain_RecursiveCharacterTextSplitter, ✅Impl:langchain-ai_langchain_TextSplitter_split_methods, ✅Principle:langchain-ai_langchain_Chunk_Size_Configuration, ✅Workflow:langchain-ai_langchain_Text_Splitting_Workflow | chunk_size=4000, chunk_overlap=200 defaults |
| langchain-ai_langchain_Token_Counting_Strategy | [→](./heuristics/langchain-ai_langchain_Token_Counting_Strategy.md) | ✅Impl:langchain-ai_langchain_TextSplitter_length_functions, ✅Workflow:langchain-ai_langchain_Agent_Creation_Workflow, ✅Principle:langchain-ai_langchain_Length_Function_Setup | ~3.3 chars/token for Claude, ~4 default |
| langchain-ai_langchain_Model_Provider_Inference | [→](./heuristics/langchain-ai_langchain_Model_Provider_Inference.md) | ✅Impl:langchain-ai_langchain_model_parsing_functions, ✅Impl:langchain-ai_langchain_init_chat_model, ✅Impl:langchain-ai_langchain_init_chat_model_helper, ✅Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow, ✅Principle:langchain-ai_langchain_Model_Identifier_Parsing | gpt-*→openai, claude*→anthropic, etc. |
| langchain-ai_langchain_Structured_Output_Strategy_Selection | [→](./heuristics/langchain-ai_langchain_Structured_Output_Strategy_Selection.md) | ✅Impl:langchain-ai_langchain_ResponseFormat_strategies, ✅Impl:langchain-ai_langchain_ResponseFormat_type_union, ✅Workflow:langchain-ai_langchain_Structured_Output_Workflow, ✅Principle:langchain-ai_langchain_Strategy_Selection, ✅Principle:langchain-ai_langchain_Structured_Output_Strategy | ToolStrategy vs ProviderStrategy decision |
| langchain-ai_langchain_Warning_Deprecated_langchain_classic | [→](./heuristics/langchain-ai_langchain_Warning_Deprecated_langchain_classic.md) | ✅Impl:langchain-ai_langchain_LLMChain, ✅Impl:langchain-ai_langchain_SequentialChain, ✅Impl:langchain-ai_langchain_MapReduceChain, ✅Impl:langchain-ai_langchain_BaseMemory, ✅Impl:langchain-ai_langchain_ConversationBufferMemory, ✅Impl:langchain-ai_langchain_chain_loading | Deprecation warning for langchain_classic |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

## Summary

- **Total Heuristics:** 5
- **By Domain:**
  - RAG/NLP: 1 (Chunk_Size_Selection)
  - LLMs/Optimization: 1 (Token_Counting_Strategy)
  - LLMs/Configuration: 1 (Model_Provider_Inference)
  - LLMs/Agents: 1 (Structured_Output_Strategy_Selection)
  - Deprecation/Migration: 1 (Warning_Deprecated_langchain_classic)
