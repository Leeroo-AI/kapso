# File: `src/transformers/convert_slow_tokenizer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 2083 |
| Classes | `SentencePieceExtractor`, `GemmaSentencePieceExtractor`, `Converter`, `BertConverter`, `SplinterConverter`, `FunnelConverter`, `MPNetConverter`, `OpenAIGPTConverter`, `GPT2Converter`, `HerbertConverter`, `Qwen2Converter`, `RobertaConverter`, `RoFormerConverter`, `DebertaConverter`, `SpmConverter`, `AlbertConverter`, `BarthezConverter`, `CamembertConverter`, `DebertaV2Converter`, `MBartConverter`, `MBart50Converter`, `NllbConverter`, `SeamlessM4TConverter`, `XLMRobertaConverter`, `XLNetConverter`, `ReformerConverter`, `RemBertConverter`, `BertGenerationConverter`, `PegasusConverter`, `T5Converter`, `UdopConverter`, `WhisperConverter`, `BigBirdConverter`, `CLIPConverter`, `LayoutLMv2Converter`, `BlenderbotConverter`, `XGLMConverter`, `GemmaConverter`, `LlamaConverter`, `MarkupLMConverter`, `MoshiConverter`, `HeliumConverter`, `ParakeetConverter`, `TikTokenConverter`, `MistralConverter` |
| Functions | `import_protobuf`, `generate_merges`, `check_number_comma`, `bytes_to_unicode`, `convert_slow_tokenizer` |
| Imports | collections, functools, packaging, tokenizers, tqdm, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Converts Python-based "slow" tokenizers to Rust-backed "fast" tokenizers by extracting vocabularies, merges, and special tokens from various tokenizer types and constructing equivalent tokenizers library objects.

**Mechanism:** Converter base class with model-specific subclasses (BertConverter, GPT2Converter, LlamaConverter, etc.) that override converted() method. Each converter: extracts vocabulary and merges from original tokenizer (BPE ranks, SentencePiece models, WordPiece vocabs), constructs tokenizers.Tokenizer with appropriate model (BPE, Unigram, WordPiece), adds normalizers (BertNormalizer, NFKC), pre-tokenizers (ByteLevel, BertPreTokenizer), decoders (ByteLevel, WordPiece), and post-processors (TemplateProcessing for special tokens). SentencePieceExtractor parses .model files using protobuf. Handles edge cases like MBART language codes, Gemma byte-fallback, and TikToken encoding.

**Significance:** Critical for tokenizer migration and performance. Fast tokenizers are 10-100x faster than Python implementations, enable batch encoding, and provide offset mapping for token positions. This file enables seamless transition from legacy tokenizers to the modern tokenizers library while maintaining exact compatibility. Used during model conversion and by users upgrading to fast tokenizers.
