# Implementation: RawTextDataLoader

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Utility class for loading raw text files and converting them into chunked datasets suitable for causal language model training.

=== Description ===
`RawTextDataLoader` handles loading raw text from various file formats (.txt, .md, .json, .jsonl, .csv) and splitting it into overlapping chunks with proper tokenization. It supports both tokenized output (for direct training) and text output (for further processing). The chunking algorithm respects token boundaries and adds EOS tokens appropriately.

=== Usage ===
Use this class when you want to train a language model on raw text data (e.g., books, articles, code) rather than structured instruction-response pairs. It handles the chunking and tokenization automatically.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/dataprep/raw_text.py unsloth/dataprep/raw_text.py]
* '''Lines:''' 37-242

=== Signature ===
<syntaxhighlight lang="python">
class RawTextDataLoader:
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 2048,
        stride: int = 512,
        return_tokenized: bool = True
    ):
        """
        Initialize the raw text data loader.

        Args:
            tokenizer: HuggingFace tokenizer for the target model
            chunk_size: Maximum tokens per chunk (default: 2048)
            stride: Overlap between consecutive chunks (default: 512)
            return_tokenized: Return tokenized dicts vs text strings (default: True)
        """

    def load_from_file(self, file_path: str, return_tokenized: bool = None) -> Dataset:
        """Load raw text from a single file and return a HuggingFace Dataset."""

    def load_from_files(self, file_paths: List[str], return_tokenized: bool = None) -> Dataset:
        """Load raw text from multiple files."""

    def smart_chunk_text(
        self,
        text: str,
        chunk_size: int,
        stride: int,
        return_tokenized: bool = True
    ) -> List[Union[str, Dict]]:
        """Split text into overlapping chunks with proper token boundaries."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import RawTextDataLoader
# or
from unsloth.dataprep.raw_text import RawTextDataLoader
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for chunking and tokenization
|-
| chunk_size || int || No (default: 2048) || Maximum tokens per chunk
|-
| stride || int || No (default: 512) || Overlap between chunks (must be < chunk_size)
|-
| return_tokenized || bool || No (default: True) || Return tokenized dicts or text strings
|-
| file_path || str || Yes (for load_from_file) || Path to text file
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Dataset || datasets.Dataset || HuggingFace Dataset with chunks
|-
| (tokenized mode) || Dict || {"input_ids": List[int], "attention_mask": List[int], "labels": List[int]}
|-
| (text mode) || Dict || {"text": str}
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, RawTextDataLoader

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Create data loader
loader = RawTextDataLoader(
    tokenizer,
    chunk_size=2048,
    stride=512,
)

# Load dataset from raw text file
dataset = loader.load_from_file("./my_corpus.txt")
print(f"Created {len(dataset)} training chunks")
</syntaxhighlight>

=== Multiple Files ===
<syntaxhighlight lang="python">
# Load from multiple files
files = ["book1.txt", "book2.txt", "article.md"]
dataset = loader.load_from_files(files)
</syntaxhighlight>

=== JSON Lines Format ===
<syntaxhighlight lang="python">
# Automatically handles JSON lines format
# Extracts text from common fields: "text", "content", "message", "body"
dataset = loader.load_from_file("./data.jsonl")
</syntaxhighlight>

=== Text Mode (Non-Tokenized) ===
<syntaxhighlight lang="python">
# Get text chunks instead of tokenized output
loader = RawTextDataLoader(
    tokenizer,
    chunk_size=2048,
    stride=512,
    return_tokenized=False,  # Return text strings
)
dataset = loader.load_from_file("./corpus.txt")
# dataset[0] = {"text": "chunk content here..."}
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
