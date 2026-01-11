# Principle: CLI_Data_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::CLI]]
|-
! Last Updated
| [[last_updated::2026-01-09 17:00 GMT]]
|}

== Overview ==
Smart dataset loading mechanism that automatically detects and processes various data formats including raw text files, HuggingFace datasets, and ModelScope datasets.

=== Description ===
CLI Data Loading is a flexible data ingestion principle that abstracts away format-specific loading logic. It enables users to pass any dataset path to the CLI without needing to know the underlying format. The system:

1. **Format Detection**: Analyzes file extension (.txt, .md, .json, .jsonl) or path type (HuggingFace identifier)
2. **Automatic Loader Selection**: Routes to RawTextDataLoader for local files or datasets.load_dataset for remote datasets
3. **ModelScope Support**: Detects UNSLOTH_USE_MODELSCOPE environment variable for Chinese model ecosystem integration
4. **Alpaca Formatting**: Applies standard instruction/input/output formatting for structured datasets

=== Usage ===
Use this principle when implementing CLI tools that need to accept heterogeneous data inputs without requiring users to specify the format explicitly. The pattern is particularly useful for:
* Command-line training scripts
* Batch processing pipelines
* User-facing tools where simplicity is paramount

== Theoretical Basis ==
The smart loading pattern follows a chain-of-responsibility design:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
def load_dataset_smart(args):
    if args.raw_text_file:
        # Explicit raw text file
        return RawTextDataLoader(tokenizer).load_from_file(args.raw_text_file)
    elif args.dataset.endswith((".txt", ".md", ".json", ".jsonl")):
        # Auto-detect local file
        return RawTextDataLoader(tokenizer).load_from_file(args.dataset)
    elif use_modelscope:
        # ModelScope dataset
        return MsDataset.load(args.dataset, split="train")
    else:
        # Default: HuggingFace dataset
        return load_dataset(args.dataset, split="train").map(formatting_func)
</syntaxhighlight>

== Related Pages ==
* [[implemented_by::Implementation:Unslothai_Unsloth_CLI]]
* [[implemented_by::Implementation:Unslothai_Unsloth_RawTextDataLoader]]
