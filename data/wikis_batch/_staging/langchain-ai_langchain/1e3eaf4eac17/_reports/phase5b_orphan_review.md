# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 60
- Approved: 34
- Rejected: 26

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.github/scripts/check_prerelease_dependencies.py` | REJECTED | CI script, no public API |
| `.github/scripts/get_min_versions.py` | REJECTED | CI utility script, internal |
| `libs/langchain/langchain_classic/base_memory.py` | APPROVED | Public ABC class, user-facing |
| `libs/langchain/langchain_classic/cache.py` | REJECTED | Re-export shim, no logic |
| `libs/langchain/langchain_classic/chains/example_generator.py` | APPROVED | Public function, user-facing |
| `libs/langchain/langchain_classic/chains/history_aware_retriever.py` | APPROVED | Public function, user-facing RAG |
| `libs/langchain/langchain_classic/chains/llm_requests.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chains/mapreduce.py` | APPROVED | Public Chain class, user-facing |
| `libs/langchain/langchain_classic/chains/moderation.py` | APPROVED | Public Chain class, user-facing |
| `libs/langchain/langchain_classic/chains/prompt_selector.py` | APPROVED | Public classes, user-facing API |
| `libs/langchain/langchain_classic/chains/retrieval.py` | APPROVED | Public function, user-facing RAG |
| `libs/langchain/langchain_classic/chains/sequential.py` | APPROVED | Public Chain classes, user-facing |
| `libs/langchain/langchain_classic/chains/transform.py` | APPROVED | Public Chain class, user-facing |
| `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/gmail.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/imessage.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/langsmith.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/slack.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/telegram.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/utils.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_loaders/whatsapp.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_models/baidu_qianfan_endpoint.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_models/everlyai.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/chat_models/tongyi.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/hub.py` | APPROVED | Public API, user-facing hub ops |
| `libs/langchain/langchain_classic/model_laboratory.py` | APPROVED | Public class, user-facing utility |
| `libs/langchain/langchain_classic/output_parsers/boolean.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/combining.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/datetime.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/enum.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/ernie_functions.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/output_parsers/fix.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/format_instructions.py` | APPROVED | Public constants, reusable prompts |
| `libs/langchain/langchain_classic/output_parsers/loading.py` | REJECTED | Internal loading helper |
| `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/prompts.py` | REJECTED | Small internal constants |
| `libs/langchain/langchain_classic/output_parsers/rail_parser.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/output_parsers/regex.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/regex_dict.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/structured.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/output_parsers/yaml.py` | APPROVED | Public parser class, user-facing |
| `libs/langchain/langchain_classic/requests.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/serpapi.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/sql_database.py` | REJECTED | Deprecated import shim only |
| `libs/langchain/langchain_classic/text_splitter.py` | REJECTED | Re-export module, no logic |
| `libs/langchain/scripts/check_imports.py` | REJECTED | Internal CI script |
| `libs/langchain_v1/langchain/agents/middleware/_retry.py` | REJECTED | Private module (_prefix) |
| `libs/langchain_v1/langchain/agents/middleware/context_editing.py` | APPROVED | Public middleware class |
| `libs/langchain_v1/langchain/agents/middleware/model_call_limit.py` | APPROVED | Public middleware class |
| `libs/langchain_v1/langchain/agents/middleware/model_fallback.py` | APPROVED | Public middleware class |
| `libs/langchain_v1/langchain/agents/middleware/todo.py` | APPROVED | Public middleware class |
| `libs/langchain_v1/langchain/agents/middleware/tool_emulator.py` | APPROVED | Public middleware class |
| `libs/langchain_v1/langchain/embeddings/base.py` | APPROVED | Public init_embeddings factory |
| `libs/langchain_v1/scripts/check_imports.py` | REJECTED | Internal CI script |
| `libs/text-splitters/langchain_text_splitters/json.py` | APPROVED | Public splitter class |
| `libs/text-splitters/langchain_text_splitters/jsx.py` | APPROVED | Public splitter class |
| `libs/text-splitters/langchain_text_splitters/konlpy.py` | APPROVED | Public splitter class |
| `libs/text-splitters/langchain_text_splitters/nltk.py` | APPROVED | Public splitter class |
| `libs/text-splitters/langchain_text_splitters/sentence_transformers.py` | APPROVED | Public splitter class |
| `libs/text-splitters/langchain_text_splitters/spacy.py` | APPROVED | Public splitter class |

## Notes

### Patterns Observed

1. **Deprecated Import Shims (14 files rejected)**: The `langchain_classic` package contains many backward-compatibility modules that simply re-export classes from `langchain_community`. These have no actual implementation logic and just use `create_importer()` with `DEPRECATED_LOOKUP` dictionaries. Examples include chat loaders (Facebook, Gmail, Slack, etc.) and chat models (Baidu, EverlyAI, Tongyi).

2. **CI/Build Scripts (4 files rejected)**: Scripts in `.github/scripts/` and `scripts/` directories are internal tooling for CI validation (checking imports, prerelease dependencies, minimum versions). These are not user-facing and have no public API.

3. **Output Parsers (11 files approved)**: The output parsers in `langchain_classic/output_parsers/` contain substantial user-facing implementations like `BooleanOutputParser`, `RegexParser`, `YamlOutputParser`, `StructuredOutputParser`, etc. These implement distinct parsing algorithms.

4. **Agent Middleware (5 files approved)**: New middleware classes in `langchain_v1/langchain/agents/middleware/` provide user-facing functionality for agent orchestration: context editing, model call limits, model fallback, todo tracking, and tool emulation.

5. **Text Splitters (6 files approved)**: All text splitter implementations are user-facing with public classes that implement distinct splitting algorithms for JSON, JSX, Korean (Konlpy), NLTK, Sentence Transformers, and spaCy.

6. **Classic Chain Classes (7 files approved)**: Chain implementations like `SequentialChain`, `MapReduceChain`, `TransformChain`, and RAG utilities (`create_retrieval_chain`, `create_history_aware_retriever`) are user-facing public APIs.

### Borderline Files

- `format_instructions.py`: Approved because it contains public constants that users may want to reference or customize for their own parsers.
- `_retry.py`: Rejected due to the underscore prefix indicating private/internal status, despite containing useful retry logic utilities.
- `base_memory.py`: Approved despite deprecation warnings because it defines a public ABC that users may still reference.
