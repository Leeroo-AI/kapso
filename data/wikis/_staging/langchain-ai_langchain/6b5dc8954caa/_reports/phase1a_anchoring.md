# Phase 1a: Anchoring Report

## Summary
- Workflows created: 4
- Total steps documented: 22

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| Agent_Creation_Workflow | factory.py, types.py, structured_output.py, base.py | 6 | `create_agent`, `init_chat_model`, `AgentMiddleware`, `ToolStrategy`, `StateGraph` |
| Text_Splitting_Workflow | base.py, character.py, html.py, markdown.py, json.py | 5 | `TextSplitter`, `RecursiveCharacterTextSplitter`, `from_tiktoken_encoder`, `split_documents` |
| Chat_Model_Initialization_Workflow | base.py | 5 | `init_chat_model`, `_attempt_infer_model_provider`, `_ConfigurableModel` |
| Structured_Output_Workflow | structured_output.py, factory.py, types.py | 6 | `ToolStrategy`, `ProviderStrategy`, `AutoStrategy`, `OutputToolBinding`, `_parse_with_schema` |

## Coverage Summary
- Source files covered: 38
- Example files documented: 0 (repo has no dedicated examples directory - code is production code)

## Source Files Identified Per Workflow

### langchain-ai_langchain_Agent_Creation_Workflow
- `libs/langchain_v1/langchain/agents/factory.py` - Core agent factory with `create_agent` function (1682 lines)
- `libs/langchain_v1/langchain/agents/middleware/types.py` - Middleware type system and `AgentMiddleware` class (1848 lines)
- `libs/langchain_v1/langchain/agents/structured_output.py` - Structured output strategies (443 lines)
- `libs/langchain_v1/langchain/chat_models/base.py` - Chat model factory (944 lines)
- `libs/langchain_v1/langchain/tools/__init__.py` - Tools entry point
- `libs/langchain_v1/langchain/agents/middleware/__init__.py` - Middleware public API
- All middleware implementations in `agents/middleware/` directory

### langchain-ai_langchain_Text_Splitting_Workflow
- `libs/text-splitters/langchain_text_splitters/base.py` - Core splitter abstractions (370 lines)
- `libs/text-splitters/langchain_text_splitters/character.py` - Character-based splitters (793 lines)
- `libs/text-splitters/langchain_text_splitters/html.py` - HTML structure splitting (1006 lines)
- `libs/text-splitters/langchain_text_splitters/markdown.py` - Markdown structure splitting (468 lines)
- `libs/text-splitters/langchain_text_splitters/json.py` - JSON structure splitting (157 lines)
- `libs/text-splitters/langchain_text_splitters/jsx.py` - JSX/React code splitting
- `libs/text-splitters/langchain_text_splitters/konlpy.py` - Korean language splitting
- `libs/text-splitters/langchain_text_splitters/latex.py` - LaTeX document splitting
- `libs/text-splitters/langchain_text_splitters/nltk.py` - NLTK sentence tokenization
- `libs/text-splitters/langchain_text_splitters/python.py` - Python code splitting
- `libs/text-splitters/langchain_text_splitters/sentence_transformers.py` - SentenceTransformers token alignment
- `libs/text-splitters/langchain_text_splitters/spacy.py` - spaCy sentence segmentation

### langchain-ai_langchain_Chat_Model_Initialization_Workflow
- `libs/langchain_v1/langchain/chat_models/base.py` - Universal chat model factory (944 lines)
- Provider packages (external: langchain-openai, langchain-anthropic, langchain-google-vertexai, etc.)

### langchain-ai_langchain_Structured_Output_Workflow
- `libs/langchain_v1/langchain/agents/structured_output.py` - Structured output strategies and parsing (443 lines)
- `libs/langchain_v1/langchain/agents/factory.py` - Agent factory with structured output integration
- `libs/langchain_v1/langchain/agents/middleware/types.py` - ResponseFormat type definitions

## Workflow Details

### 1. Agent_Creation_Workflow
**Purpose:** Create and execute AI agents with tool calling, middleware composition, and structured output.

**Key Concepts:**
- LangGraph StateGraph-based agent loop
- Middleware hooks: before_agent, before_model, after_model, wrap_model_call, wrap_tool_call
- Multiple structured output strategies (ToolStrategy, ProviderStrategy, AutoStrategy)
- 20+ provider support via unified `init_chat_model`

**Steps:**
1. Model Initialization - `init_chat_model` factory
2. Tool Definition - `BaseTool`, `StructuredTool` conversion
3. Middleware Composition - `AgentMiddleware` subclassing and decorators
4. Response Format Configuration - Strategy selection for structured output
5. Graph Construction - StateGraph assembly with conditional routing
6. Agent Execution - `invoke`/`stream` execution modes

### 2. Text_Splitting_Workflow
**Purpose:** Split documents into semantic chunks for RAG and LLM context windows.

**Key Concepts:**
- Multiple splitter strategies (recursive, language-aware, format-specific)
- Configurable length functions (character, tiktoken, HuggingFace)
- Chunk overlap for context continuity
- Metadata preservation through splits

**Steps:**
1. Splitter Selection - Choose appropriate splitter for content type
2. Length Function Configuration - Token counting vs character counting
3. Chunk Parameters - chunk_size, chunk_overlap configuration
4. Document Splitting - Recursive splitting with separator fallback
5. Metadata Preservation - Copy and augment metadata on chunks

### 3. Chat_Model_Initialization_Workflow
**Purpose:** Initialize chat models from any provider using a unified factory interface.

**Key Concepts:**
- Provider inference from model name prefixes
- Configurable models for runtime switching
- Declarative operation queuing (bind_tools, with_structured_output)
- Package validation for provider integrations

**Steps:**
1. Model Identifier Parsing - Parse `provider:model` syntax
2. Provider Package Validation - Check integration package installation
3. Model Instantiation - Create provider-specific model instance
4. Configurable Model Setup - Optional runtime configuration
5. Declarative Operation Binding - Queue operations for configurable models

### 4. Structured_Output_Workflow
**Purpose:** Extract validated, structured responses from language models.

**Key Concepts:**
- Multiple schema types (Pydantic, dataclass, TypedDict, JSON schema)
- Strategy selection (Auto, Tool, Provider)
- Validation with retry on parse errors
- Union types for multiple valid schemas

**Steps:**
1. Schema Definition - Define response structure
2. Strategy Selection - Choose extraction strategy
3. Tool Binding - Create synthetic tools for ToolStrategy
4. Model Invocation - Execute with response format
5. Response Parsing and Validation - Parse to typed Python objects
6. Error Handling and Retry - Configurable retry on validation errors

## Notes for Phase 1b (Enrichment)

### Files needing line-by-line tracing:
- `libs/langchain_v1/langchain/agents/factory.py` - Complex graph construction logic
- `libs/langchain_v1/langchain/chat_models/base.py` - Provider mapping and configurable model setup
- `libs/langchain_v1/langchain/agents/middleware/types.py` - Middleware protocol implementations
- `libs/text-splitters/langchain_text_splitters/character.py` - Recursive splitting algorithm

### External APIs to document:
- LangGraph: `StateGraph`, `add_node`, `add_conditional_edges`, `compile`
- langchain-core: `BaseTool`, `StructuredTool`, `BaseChatModel`
- Pydantic: `TypeAdapter`, `validate_python`
- tiktoken: `encoding_for_model`, `get_encoding`

### Unclear mappings to resolve:
- The `create_agent` function has many internal helper functions that may need separate Implementation pages
- Middleware decorator APIs (`@before_model`, `@wrap_tool_call`) may warrant their own Principle pages
- Text splitter `from_language` factory covers 30+ programming languages - may need language-specific pages
