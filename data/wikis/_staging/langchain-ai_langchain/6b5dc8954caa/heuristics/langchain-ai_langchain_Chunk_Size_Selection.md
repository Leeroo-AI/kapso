# Chunk Size Selection Heuristic

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Text Splitters|https://github.com/langchain-ai/langchain/tree/main/libs/text-splitters]]
* [[source::Discussion|Text Splitting Best Practices|https://docs.langchain.com/oss/python/langchain/how_to/text_splitters]]
|-
! Domains
| [[domain::RAG]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Guidance for selecting optimal `chunk_size` and `chunk_overlap` parameters in text splitters to balance retrieval precision with context preservation.

=== Description ===
Text splitting parameters significantly impact RAG (Retrieval-Augmented Generation) performance. The `chunk_size` determines how much text each chunk contains, while `chunk_overlap` controls continuity between adjacent chunks. The defaults of `chunk_size=4000` and `chunk_overlap=200` characters work well for general use cases, but specific applications may require tuning.

=== Usage ===
Apply this heuristic when:
- Setting up a new RAG pipeline
- Documents are being split but retrieval quality is poor
- Chunks are too small (losing context) or too large (diluting relevance)
- Using token-based splitting with tiktoken or HuggingFace tokenizers

== The Insight (Rule of Thumb) ==

* **Action:** Set `chunk_size` based on your use case and model context window
* **Values:**
  - Default: `chunk_size=4000` characters, `chunk_overlap=200` characters
  - For semantic precision: `chunk_size=500-1000` characters
  - For context-heavy tasks: `chunk_size=2000-4000` characters
  - For code: `chunk_size=1000-2000` characters with language-aware separators
* **Trade-off:** Smaller chunks = better retrieval precision but less context; Larger chunks = more context but potentially diluted relevance
* **Constraint:** `chunk_overlap` must be < `chunk_size` and >= 0

=== Overlap Guidelines ===
* **Rule of thumb:** Set `chunk_overlap` to 5-15% of `chunk_size`
* For `chunk_size=1000`: use `chunk_overlap=50-150`
* For `chunk_size=4000`: use `chunk_overlap=200-500`
* Higher overlap preserves more cross-chunk continuity but increases storage/processing

=== Token-Based vs Character-Based ===
* **Character-based (default):** Fast but imprecise for token-heavy models
* **Token-based (tiktoken):** More accurate for OpenAI models; use `from_tiktoken_encoder()`
* **HuggingFace tokenizer:** Use `from_huggingface_tokenizer()` for specific models

== Reasoning ==

The default `chunk_size=4000` characters (approximately 1000 tokens for GPT models) was chosen to:
1. Fit comfortably within typical embedding model limits (512-8192 tokens)
2. Provide enough context for semantic understanding
3. Avoid the warning about chunks exceeding specified size

The chunk overlap of 200 characters (5% of default chunk_size) ensures:
1. Sentences split at boundaries appear in adjacent chunks
2. Cross-chunk context is preserved for entities spanning chunk boundaries
3. Minimal duplication overhead

Code Evidence from `base.py:68-79` showing validation:
<syntaxhighlight lang="python">
if chunk_size <= 0:
    msg = f"chunk_size must be > 0, got {chunk_size}"
    raise ValueError(msg)
if chunk_overlap < 0:
    msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
    raise ValueError(msg)
if chunk_overlap > chunk_size:
    msg = (
        f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
        f"({chunk_size}), should be smaller."
    )
    raise ValueError(msg)
</syntaxhighlight>

Warning issued when chunks exceed size from `base.py:139-145`:
<syntaxhighlight lang="python">
if total > self._chunk_size:
    logger.warning(
        "Created a chunk of size %d, which is longer than the "
        "specified %d",
        total,
        self._chunk_size,
    )
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:langchain-ai_langchain_TextSplitter_init]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_RecursiveCharacterTextSplitter]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_TextSplitter_split_methods]]
* [[uses_heuristic::Principle:langchain-ai_langchain_Chunk_Size_Configuration]]
* [[uses_heuristic::Workflow:langchain-ai_langchain_Text_Splitting_Workflow]]
