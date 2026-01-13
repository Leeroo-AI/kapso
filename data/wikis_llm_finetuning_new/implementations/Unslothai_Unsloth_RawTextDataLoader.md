# Implementation: RawTextDataLoader

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
RawTextDataLoader and TextPreprocessor provide utilities for loading raw text files and preprocessing them into chunked datasets suitable for causal language model training.

=== Description ===
The module contains two main classes:

'''RawTextDataLoader''': A flexible data loader that automatically detects file formats (.txt, .md, .json, .jsonl, .csv), reads and parses them, intelligently chunks the text based on token counts with configurable overlap/stride, and creates HuggingFace Dataset objects ready for causal language modeling. The chunking algorithm respects token boundaries and adds EOS tokens appropriately.

'''TextPreprocessor''': A utility class for cleaning and validating text data. It can normalize whitespace, remove non-ASCII characters, extract specific sections using regex patterns, add structural tokens for markdown headings and code blocks, and validate datasets for quality issues like empty samples, encoding problems, or repeated content.

=== Usage ===
Import RawTextDataLoader when you need to load raw text files (books, documents, articles) for continued pretraining or domain adaptation. Import TextPreprocessor when you need to clean text data or validate dataset quality before training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/dataprep/raw_text.py
* '''Lines:''' 1-349

=== Signature ===
<syntaxhighlight lang="python">
class RawTextDataLoader:
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 2048,
        stride: int = 512,
        return_tokenized: bool = True
    ) -> None: ...

class TextPreprocessor:
    def clean_text(self, text: str) -> str: ...
    def extract_sections(self, text: str, patterns: List[str]) -> List[str]: ...
    def add_structure_tokens(self, text: str) -> str: ...
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.dataprep.raw_text import RawTextDataLoader, TextPreprocessor
</syntaxhighlight>

== I/O Contract ==

=== Inputs (RawTextDataLoader.__init__) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizer || Yes || HuggingFace tokenizer for encoding/decoding text
|-
| chunk_size || int || No || Maximum number of tokens per chunk (default: 2048)
|-
| stride || int || No || Overlap between consecutive chunks in tokens (default: 512)
|-
| return_tokenized || bool || No || If True, returns tokenized chunks with input_ids/attention_mask; if False, returns text strings (default: True)
|}

=== Inputs (RawTextDataLoader.load_from_file) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| file_path || str || Yes || Path to the text file to load
|-
| return_tokenized || bool || No || Override instance setting for tokenized output
|}

=== Inputs (TextPreprocessor.validate_dataset) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| dataset || Dataset || Yes || HuggingFace Dataset with a "text" column to validate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| load_from_file return || Dataset || HuggingFace Dataset with input_ids, attention_mask, labels (if tokenized) or text column
|-
| clean_text return || str || Cleaned text with normalized whitespace and removed special characters
|-
| validate_dataset return || Dict[str, Any] || Statistics dict with total_samples, empty_samples, min/max/avg_length, repeated_content, encoding_issues, and warnings list
|}

== Usage Examples ==
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
from unsloth.dataprep.raw_text import RawTextDataLoader, TextPreprocessor

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B")

# Create data loader with 2048 token chunks and 256 token overlap
loader = RawTextDataLoader(
    tokenizer=tokenizer,
    chunk_size=2048,
    stride=256,
    return_tokenized=True
)

# Load a single text file
dataset = loader.load_from_file("./my_book.txt")
print(f"Created {len(dataset)} training samples")

# Load multiple files
dataset = loader.load_from_files(["./chapter1.txt", "./chapter2.md", "./data.jsonl"])

# Use TextPreprocessor to clean and validate
preprocessor = TextPreprocessor()

# Clean raw text
raw_text = "  Some   messy   text\n\n\n\nwith extra   spaces  "
clean = preprocessor.clean_text(raw_text)
print(clean)  # "Some messy text\n\nwith extra spaces"

# Add structural tokens to markdown
markdown_text = "# Chapter 1\n## Section A\n```python\ncode here\n```"
structured = preprocessor.add_structure_tokens(markdown_text)
# "<|chapter|>Chapter 1<|/chapter|>\n<|section|>Section A<|/section|>\n<|code|python|>code here<|/code|>"

# Validate dataset quality
stats = preprocessor.validate_dataset(dataset)
print(f"Total samples: {stats['total_samples']}")
print(f"Average length: {stats['avg_length']:.0f} chars")
if stats['warnings']:
    print("Warnings:", stats['warnings'])
</syntaxhighlight>

== Related Pages ==
* [[related::Implementation:Unslothai_Unsloth_get_chat_template]]
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
