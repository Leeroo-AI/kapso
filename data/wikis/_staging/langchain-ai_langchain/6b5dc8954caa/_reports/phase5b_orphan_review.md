# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 47
- Approved: 24
- Rejected: 23

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.github/scripts/check_prerelease_dependencies.py` | REJECTED | CI script, no public API |
| `.github/scripts/get_min_versions.py` | REJECTED | CI script, no public API |
| `libs/langchain/langchain_classic/base_memory.py` | APPROVED | Public BaseMemory class, user-facing |
| `libs/langchain/langchain_classic/cache.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chains/example_generator.py` | APPROVED | Public generate_example function |
| `libs/langchain/langchain_classic/chains/history_aware_retriever.py` | APPROVED | Public factory, user-facing RAG |
| `libs/langchain/langchain_classic/chains/llm_requests.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chains/mapreduce.py` | APPROVED | Public MapReduceChain class |
| `libs/langchain/langchain_classic/chains/moderation.py` | APPROVED | Public OpenAIModerationChain |
| `libs/langchain/langchain_classic/chains/prompt_selector.py` | APPROVED | Public classes and helpers |
| `libs/langchain/langchain_classic/chains/retrieval.py` | APPROVED | Public create_retrieval_chain |
| `libs/langchain/langchain_classic/chains/sequential.py` | APPROVED | Public SequentialChain classes |
| `libs/langchain/langchain_classic/chains/transform.py` | APPROVED | Public TransformChain class |
| `libs/langchain/langchain_classic/chat_loaders/facebook_messenger.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/gmail.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/imessage.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/langsmith.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/slack.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/telegram.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/utils.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_loaders/whatsapp.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_models/baidu_qianfan_endpoint.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_models/everlyai.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/chat_models/tongyi.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/hub.py` | APPROVED | Public push/pull functions |
| `libs/langchain/langchain_classic/model_laboratory.py` | APPROVED | Public ModelLaboratory class |
| `libs/langchain/langchain_classic/output_parsers/boolean.py` | APPROVED | Public BooleanOutputParser |
| `libs/langchain/langchain_classic/output_parsers/combining.py` | APPROVED | Public CombiningOutputParser |
| `libs/langchain/langchain_classic/output_parsers/datetime.py` | APPROVED | Public DatetimeOutputParser |
| `libs/langchain/langchain_classic/output_parsers/enum.py` | APPROVED | Public EnumOutputParser |
| `libs/langchain/langchain_classic/output_parsers/ernie_functions.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/output_parsers/fix.py` | APPROVED | Public OutputFixingParser |
| `libs/langchain/langchain_classic/output_parsers/format_instructions.py` | APPROVED | Public format instruction strings |
| `libs/langchain/langchain_classic/output_parsers/loading.py` | REJECTED | Internal helper, trivial logic |
| `libs/langchain/langchain_classic/output_parsers/pandas_dataframe.py` | APPROVED | Public PandasDataFrameOutputParser |
| `libs/langchain/langchain_classic/output_parsers/prompts.py` | REJECTED | Internal constants, minimal logic |
| `libs/langchain/langchain_classic/output_parsers/rail_parser.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/output_parsers/regex.py` | APPROVED | Public RegexParser class |
| `libs/langchain/langchain_classic/output_parsers/regex_dict.py` | APPROVED | Public RegexDictParser class |
| `libs/langchain/langchain_classic/output_parsers/structured.py` | APPROVED | Public StructuredOutputParser |
| `libs/langchain/langchain_classic/output_parsers/yaml.py` | APPROVED | Public YamlOutputParser class |
| `libs/langchain/langchain_classic/requests.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/serpapi.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/langchain_classic/sql_database.py` | REJECTED | Import proxy shim, no logic |
| `libs/langchain/scripts/check_imports.py` | REJECTED | CI script, no public API |
| `libs/langchain_v1/langchain/embeddings/base.py` | APPROVED | Public init_embeddings factory |
| `libs/langchain_v1/scripts/check_imports.py` | REJECTED | CI script, no public API |

## Notes

### Patterns Observed

1. **Import Proxy Shims (17 files rejected)**: A large portion of the MANUAL_REVIEW files were backward-compatibility import proxies that use `create_importer()` to redirect imports to `langchain_community`. These contain no actual logic, just dynamic `__getattr__` handlers. Examples include all chat_loaders (facebook_messenger, gmail, slack, etc.) and chat_models providers.

2. **User-Facing Chains (9 files approved)**: Chain implementations with public classes that users would directly instantiate or call, like `SequentialChain`, `MapReduceChain`, `TransformChain`, and factory functions like `create_retrieval_chain`.

3. **Output Parsers (12 files approved)**: Concrete parser implementations with public APIs and distinct parsing algorithms. Each parser provides a different parsing strategy (boolean, datetime, enum, regex, structured, yaml, pandas, etc.).

4. **CI/Build Scripts (4 files rejected)**: Scripts in `.github/scripts/` and `scripts/` directories are internal tooling for the CI pipeline, not user-facing APIs.

### Borderline Decisions

- **`format_instructions.py`**: Approved despite being constants-only because these are publicly documented format strings that users reference when implementing custom parsers.

- **`prompts.py`**: Rejected because it only contains internal prompt templates used by `OutputFixingParser` - not independently useful.

- **`loading.py`**: Rejected because it's a minimal config-loading helper with only one supported parser type, not a general-purpose public utility.
