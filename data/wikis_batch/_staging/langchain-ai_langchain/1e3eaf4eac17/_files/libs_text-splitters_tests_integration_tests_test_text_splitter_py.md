# File: `libs/text-splitters/tests/integration_tests/test_text_splitter.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 114 |
| Functions | `test_huggingface_type_check`, `test_huggingface_tokenizer`, `test_token_text_splitter`, `test_token_text_splitter_overlap`, `test_token_text_splitter_from_tiktoken`, `test_sentence_transformers_count_tokens`, `test_sentence_transformers_split_text`, `test_sentence_transformers_multiple_tokens` |
| Imports | langchain_text_splitters, pytest, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests text splitters that depend on external tokenization libraries like HuggingFace transformers, tiktoken, and sentence-transformers.

**Mechanism:** Tests TokenTextSplitter with and without overlap using basic splitting logic. Validates integration with HuggingFace's GPT2TokenizerFast, tiktoken encoders (for OpenAI models like gpt-3.5-turbo), and sentence-transformers tokenizers. Verifies type checking, token counting, and proper chunk generation with edge cases for different token multipliers.

**Significance:** Integration tests for token-aware text splitting, which is critical when working with LLMs that have specific token limits (e.g., GPT models). Ensures accurate token counting and splitting that respects model constraints while maintaining semantic coherence.
