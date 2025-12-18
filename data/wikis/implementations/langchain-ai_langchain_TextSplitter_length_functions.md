{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|tiktoken|https://github.com/openai/tiktoken]]
* [[source::Doc|HuggingFace Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::RAG]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for configuring token-based length measurement in text splitters using tiktoken or HuggingFace tokenizers, provided by LangChain text-splitters.

=== Description ===

`TextSplitter.from_tiktoken_encoder` and `from_huggingface_tokenizer` are factory methods that create splitters with token-based length functions instead of character-based. This is critical for:
* Respecting LLM context window limits (measured in tokens, not characters)
* Consistent sizing across different text types (code vs prose have different token densities)
* Alignment with specific model tokenizers

These methods replace the default `len()` function with a tokenizer-based counter.

=== Usage ===

Use token-based length functions when:
* Building RAG systems targeting specific models
* Optimizing chunks for context window utilization
* Ensuring chunks don't exceed token limits
* Working with multilingual text (characters-to-tokens ratio varies)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/text-splitters/langchain_text_splitters/base.py
* '''Lines:''' L169-231

=== Signature ===
<syntaxhighlight lang="python">
class TextSplitter(ABC):
    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> Self:
        """Create splitter using tiktoken encoder for length measurement.

        Args:
            encoding_name: tiktoken encoding name (e.g., "cl100k_base", "gpt2")
            model_name: Model name to infer encoding (overrides encoding_name)
            allowed_special: Special tokens to allow in encoding
            disallowed_special: Special tokens to disallow
            **kwargs: TextSplitter args (chunk_size, chunk_overlap, etc.)

        Returns:
            TextSplitter with token-based length function.

        Raises:
            ImportError: If tiktoken is not installed.
        """

    @classmethod
    def from_huggingface_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: Any,
    ) -> TextSplitter:
        """Create splitter using HuggingFace tokenizer for length measurement.

        Args:
            tokenizer: HuggingFace tokenizer instance
            **kwargs: TextSplitter args

        Returns:
            TextSplitter with HuggingFace token-based length function.

        Raises:
            ValueError: If transformers is not installed or tokenizer invalid.
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For tiktoken-based:
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)

# For HuggingFace-based:
from transformers import AutoTokenizer
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (from_tiktoken_encoder) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| encoding_name || str || No || tiktoken encoding name (default: "gpt2")
|-
| model_name || str | None || No || Model name to infer encoding (e.g., "gpt-4")
|-
| allowed_special || "all" | Set[str] || No || Special tokens to allow
|-
| disallowed_special || "all" | Collection[str] || No || Special tokens to disallow
|-
| **kwargs || Any || No || TextSplitter constructor args
|}

=== Inputs (from_huggingface_tokenizer) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizerBase || Yes || HuggingFace tokenizer instance
|-
| **kwargs || Any || No || TextSplitter constructor args
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || TextSplitter (subclass) || Splitter with token-based length function
|}

== Usage Examples ==

=== Using tiktoken for OpenAI Models ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create splitter for GPT-4 (uses cl100k_base encoding)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=500,  # 500 tokens, not characters
    chunk_overlap=50,
)

# Or specify encoding directly
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4, GPT-3.5-turbo encoding
    chunk_size=1000,
    chunk_overlap=100,
)

text = "Your long document text here..."
chunks = splitter.split_text(text)

# Each chunk will be approximately 500 tokens
</syntaxhighlight>

=== Using HuggingFace Tokenizer ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create splitter using this tokenizer
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512,  # 512 tokens
    chunk_overlap=64,
)

chunks = splitter.split_text(text)
</syntaxhighlight>

=== Comparing Character vs Token Counts ===
<syntaxhighlight lang="python">
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Sample text
text = "Hello, world! This is a sample text." * 100

# Character-based (default)
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 200 characters
)
char_chunks = char_splitter.split_text(text)
print(f"Character-based: {len(char_chunks)} chunks")

# Token-based
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=50,  # 50 tokens
)
token_chunks = token_splitter.split_text(text)
print(f"Token-based: {len(token_chunks)} chunks")

# Verify token count
enc = tiktoken.encoding_for_model("gpt-4")
for chunk in token_chunks[:3]:
    tokens = enc.encode(chunk)
    print(f"Chunk: {len(chunk)} chars, {len(tokens)} tokens")
</syntaxhighlight>

=== Common Encoding Names ===
<syntaxhighlight lang="python">
# tiktoken encoding mappings
ENCODINGS = {
    "cl100k_base": ["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"],
    "p50k_base": ["text-davinci-003", "code-davinci-002"],
    "p50k_edit": ["text-davinci-edit-001", "code-davinci-edit-001"],
    "r50k_base": ["davinci", "curie", "babbage", "ada"],
    "gpt2": ["gpt2"],  # Default fallback
    "o200k_base": ["gpt-4o", "o1", "o3"],  # Newer models
}
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Length_Function_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_Text_Splitters_Environment]]
