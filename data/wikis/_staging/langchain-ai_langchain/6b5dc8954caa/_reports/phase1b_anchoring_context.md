# Phase 1b: WorkflowIndex Enrichment Report

## Summary
- Workflows enriched: 4
- Steps with detailed tables: 22
- Source files traced: 5

## Enrichment Details

| Workflow | Steps Enriched | APIs Traced | Line Numbers Found |
|----------|----------------|-------------|-------------------|
| Agent_Creation_Workflow | 6 | 8 | Yes |
| Text_Splitting_Workflow | 5 | 7 | Yes |
| Chat_Model_Initialization_Workflow | 5 | 6 | Yes |
| Structured_Output_Workflow | 6 | 9 | Yes |

## Implementation Types Found

| Type | Count | Examples |
|------|-------|----------|
| API Doc | 20 | `init_chat_model`, `RecursiveCharacterTextSplitter`, `ToolStrategy`, `_SchemaSpec`, `TextSplitter.__init__` |
| Wrapper Doc | 2 | `BaseTool`, `StructuredTool` (from langchain_core) |
| Pattern Doc | 0 | N/A |
| External Tool Doc | 0 | N/A |

## Source Files Traced

| File | Lines | APIs Extracted |
|------|-------|----------------|
| `libs/langchain_v1/langchain/chat_models/base.py` | L59-945 | `init_chat_model`, `_parse_model`, `_attempt_infer_model_provider`, `_check_pkg`, `_init_chat_model_helper`, `_ConfigurableModel`, `bind_tools`, `with_structured_output` |
| `libs/langchain_v1/langchain/agents/structured_output.py` | L34-443 | `ToolStrategy`, `ProviderStrategy`, `AutoStrategy`, `_SchemaSpec`, `OutputToolBinding`, `ProviderStrategyBinding`, `_parse_with_schema`, error classes |
| `libs/langchain_v1/langchain/agents/factory.py` | L401-1483 | `create_agent` (graph building), `_get_bound_model`, `model_node`, `amodel_node`, `_handle_structured_output_error` |
| `libs/text-splitters/langchain_text_splitters/base.py` | L44-371 | `TextSplitter.__init__`, `split_text`, `split_documents`, `create_documents`, `from_tiktoken_encoder`, `from_huggingface_tokenizer`, `_merge_splits` |
| `libs/text-splitters/langchain_text_splitters/character.py` | L11-169 | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `from_language`, `get_separators_for_language`, `_split_text`, `_split_text_with_regex` |

## Detailed API Signatures Extracted

### Agent_Creation_Workflow

1. **init_chat_model**
   - Signature: `init_chat_model(model: str | None = None, *, model_provider: str | None = None, configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None, config_prefix: str | None = None, **kwargs: Any) -> BaseChatModel | _ConfigurableModel`
   - Location: `chat_models/base.py:L59-329`

2. **ToolStrategy**
   - Signature: `ToolStrategy(schema: type[SchemaT], *, tool_message_content: str | None = None, handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True)`
   - Location: `structured_output.py:L181-243`

3. **ProviderStrategy**
   - Signature: `ProviderStrategy(schema: type[SchemaT], *, strict: bool | None = None)`
   - Location: `structured_output.py:L246-286`

### Text_Splitting_Workflow

1. **TextSplitter.__init__**
   - Signature: `TextSplitter.__init__(chunk_size: int = 4000, chunk_overlap: int = 200, length_function: Callable[[str], int] = len, keep_separator: bool | Literal["start", "end"] = False, add_start_index: bool = False, strip_whitespace: bool = True)`
   - Location: `base.py:L47-85`

2. **RecursiveCharacterTextSplitter.from_language**
   - Signature: `from_language(cls, language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter`
   - Location: `character.py:L153-169`

3. **from_tiktoken_encoder**
   - Signature: `from_tiktoken_encoder(cls, encoding_name: str = "gpt2", model_name: str | None = None, allowed_special: Literal["all"] | AbstractSet[str] = set(), disallowed_special: Literal["all"] | Collection[str] = "all", **kwargs: Any) -> Self`
   - Location: `base.py:L190-231`

### Chat_Model_Initialization_Workflow

1. **_parse_model**
   - Signature: `_parse_model(model: str, model_provider: str | None) -> tuple[str, str]`
   - Location: `base.py:L515-530`

2. **_ConfigurableModel**
   - Signature: `_ConfigurableModel(*, default_config: dict | None = None, configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any", config_prefix: str = "", queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = ())`
   - Location: `base.py:L547-648`

### Structured_Output_Workflow

1. **_SchemaSpec**
   - Signature: `_SchemaSpec(schema: type[SchemaT], *, name: str | None = None, description: str | None = None, strict: bool | None = None)`
   - Location: `structured_output.py:L104-177`

2. **OutputToolBinding.from_schema_spec**
   - Signature: `from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self`
   - Location: `structured_output.py:L307-325`

3. **_parse_with_schema**
   - Signature: `_parse_with_schema(schema: type[SchemaT] | dict, schema_kind: SchemaKind, data: dict[str, Any]) -> Any`
   - Location: `structured_output.py:L76-101`

## External Dependencies Identified

| Category | Dependencies |
|----------|--------------|
| LangChain Core | `langchain_core.language_models.BaseChatModel`, `langchain_core.documents.Document`, `langchain_core.tools.BaseTool`, `langchain_core.tools.StructuredTool`, `langchain_core.runnables.Runnable` |
| LangGraph | `langgraph.graph.state.StateGraph`, `langgraph.prebuilt.tool_node.ToolNode`, `langgraph.types.Command`, `langgraph.types.Send` |
| Pydantic | `pydantic.BaseModel`, `pydantic.TypeAdapter` |
| Provider Packages | `langchain_openai`, `langchain_anthropic`, `langchain_google_vertexai`, `langchain_aws`, etc. |
| Optional | `tiktoken`, `transformers.PreTrainedTokenizerBase` |

## Issues Found

- **No issues**: All APIs were successfully traced to source locations
- **External dependencies**: Some tool-related classes (`BaseTool`, `StructuredTool`) are defined in `langchain_core`, which is outside this repo but part of the LangChain ecosystem
- **LangGraph dependency**: Agent factory relies heavily on `langgraph` package for graph construction and compilation

## Ready for Phase 2

- [x] All Step tables complete (22 steps across 4 workflows)
- [x] All source locations verified with line numbers
- [x] Implementation Extraction Guides complete for all 4 workflows
- [x] All 9 attributes filled in for each step table
- [x] No `<!-- ENRICHMENT NEEDED -->` comments remain
