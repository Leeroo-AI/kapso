{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|tiktoken|https://github.com/openai/tiktoken]]
* [[source::Paper|BPE|https://arxiv.org/abs/1508.07909]]
* [[source::Doc|HuggingFace Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::LLM]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Configuration pattern for defining how text length is measured when splitting documents, enabling alignment with LLM tokenization.

=== Description ===

Length Function Setup determines how chunk sizes are measured during text splitting. By default, splitters use character count (`len()`), but this doesn't align with how LLMs process text—they use tokens.

The mismatch matters because:
* Token count varies by text type (code is denser than prose)
* Different models use different tokenizers
* Context windows are defined in tokens, not characters
* Multilingual text has varying character-to-token ratios

Configuring token-based length functions ensures chunks fit model context windows accurately.

=== Usage ===

Use token-based length measurement when:
* Building production RAG systems
* Targeting specific model context limits
* Working with code (higher token density)
* Processing multilingual documents
* Optimizing embedding batch sizes

Character-based length is fine for:
* Quick prototyping
* Model-agnostic processing
* Simple text preprocessing

== Theoretical Basis ==

Length Function Setup configures the **measurement function** for chunk sizing.

'''1. The Measurement Problem'''

<syntaxhighlight lang="python">
# Why characters ≠ tokens
text = "Hello, world!"

# Character count
char_count = len(text)  # 13 characters

# Token count varies by tokenizer
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
token_count = len(enc.encode(text))  # Might be 4 tokens

# Ratio varies dramatically:
code = "def function_name_here():"
prose = "The quick brown fox jumps"
# Code has lower chars/token ratio
</syntaxhighlight>

'''2. Length Function Interface'''

<syntaxhighlight lang="python">
# The length_function signature
from typing import Callable

LengthFunction = Callable[[str], int]

# Default implementation
def character_length(text: str) -> int:
    return len(text)

# Token-based implementation
def make_tiktoken_length(model_name: str) -> LengthFunction:
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)

    def tiktoken_length(text: str) -> int:
        return len(enc.encode(text))

    return tiktoken_length

# HuggingFace implementation
def make_hf_length(tokenizer) -> LengthFunction:
    def hf_length(text: str) -> int:
        return len(tokenizer.tokenize(text))
    return hf_length
</syntaxhighlight>

'''3. Chunk Size Interpretation'''

<syntaxhighlight lang="python">
# With character-based length:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 1000 characters
    length_function=len,  # Default
)

# With token-based length:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 1000 TOKENS
    length_function=tiktoken_length,
)

# The chunk_size parameter means different things!
</syntaxhighlight>

'''4. Model-Tokenizer Alignment'''

<syntaxhighlight lang="python">
# Best practice: Match tokenizer to target model
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "llama-2": "meta-llama/Llama-2-7b-hf",  # HuggingFace
    "mistral": "mistralai/Mistral-7B-v0.1",
}

def create_aligned_splitter(model_name: str, chunk_size: int):
    """Create splitter aligned with model's tokenizer."""
    if model_name.startswith("gpt"):
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
        )
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ENCODINGS[model_name])
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )
</syntaxhighlight>

'''5. Context Window Budgeting'''

<syntaxhighlight lang="python">
# Pseudo-code for context window planning
MODEL_CONTEXT_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "claude-3-sonnet": 200000,
}

def plan_chunk_size(model: str, num_chunks_to_retrieve: int, safety_margin: float = 0.8):
    """Calculate optimal chunk size for RAG."""
    context_limit = MODEL_CONTEXT_LIMITS[model]
    available = context_limit * safety_margin  # Leave room for system prompt, etc.

    # Reserve space for retrieved chunks
    chunk_size = int(available / num_chunks_to_retrieve)

    return chunk_size

# Example: Retrieve 5 chunks for GPT-4 (8192 context)
# chunk_size = 8192 * 0.8 / 5 ≈ 1310 tokens
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_TextSplitter_length_functions]]

=== Used By Workflows ===
* Text_Splitting_Workflow (Step 2)
