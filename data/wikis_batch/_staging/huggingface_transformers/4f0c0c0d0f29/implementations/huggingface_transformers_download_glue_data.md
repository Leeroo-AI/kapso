# Implementation: huggingface_transformers_download_glue_data

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Paper|GLUE Benchmark|https://arxiv.org/abs/1804.07461]]
|-
! Domains
| [[domain::Data_Utilities]], [[domain::Benchmarking]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

CLI utility script for downloading GLUE benchmark datasets, handling task-specific data formats and MRPC special processing.

=== Description ===

The `download_glue_data` script downloads and extracts all GLUE (General Language Understanding Evaluation) benchmark datasets. It supports 11 tasks: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, SNLI, QNLI, RTE, WNLI, and diagnostic. MRPC requires special handling due to licensing - it either downloads from SentEval or processes local MSR Paraphrase Corpus files.

=== Usage ===

Run as a CLI script to download GLUE data before running benchmark evaluations. Essential for reproducing NLP benchmark results.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/download_glue_data.py utils/download_glue_data.py]
* '''Lines:''' 1-161

=== Signature ===
<syntaxhighlight lang="python">
# Supported GLUE tasks
TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]

def download_and_extract(task: str, data_dir: str) -> None:
    """Download and extract a GLUE task dataset."""

def format_mrpc(data_dir: str, path_to_data: str) -> None:
    """Process MRPC data with train/dev/test splits."""

def download_diagnostic(data_dir: str) -> None:
    """Download GLUE diagnostic dataset."""

def get_tasks(task_names: str) -> list[str]:
    """Parse comma-separated task names."""

def main(arguments: list[str]) -> int:
    """
    CLI entry point.

    Args:
        --data_dir: Directory to save data (default: glue_data)
        --tasks: Comma-separated tasks or 'all' (default: all)
        --path_to_mrpc: Path to local MRPC data (optional)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python utils/download_glue_data.py --data_dir ./glue_data --tasks all
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --data_dir || str || No || Output directory (default: glue_data)
|-
| --tasks || str || No || Tasks to download (default: all)
|-
| --path_to_mrpc || str || No || Local MRPC data path
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| {data_dir}/{TASK}/ || Directory || Task-specific data folders with TSV files
|-
| train.tsv || File || Training split
|-
| dev.tsv || File || Development/validation split
|-
| test.tsv || File || Test split (labels may be hidden)
|}

== Usage Examples ==

=== Download All GLUE Tasks ===
<syntaxhighlight lang="bash">
# Download all GLUE benchmark data
python utils/download_glue_data.py --data_dir ./glue_data --tasks all

# Result structure:
# glue_data/
#   CoLA/
#     train.tsv, dev.tsv, test.tsv
#   SST-2/
#     train.tsv, dev.tsv, test.tsv
#   MRPC/
#     train.tsv, dev.tsv, test.tsv
#   ...
</syntaxhighlight>

=== Download Specific Tasks ===
<syntaxhighlight lang="bash">
# Download only sentiment and paraphrase tasks
python utils/download_glue_data.py --tasks SST,MRPC,QQP

# Download with local MRPC data (if you have MSRParaphraseCorpus)
python utils/download_glue_data.py --tasks MRPC --path_to_mrpc ./local_mrpc/
</syntaxhighlight>

=== Use with Training ===
<syntaxhighlight lang="python">
# After downloading, load with datasets library
from datasets import load_dataset

# Load MRPC from downloaded files
mrpc = load_dataset(
    "csv",
    data_files={
        "train": "glue_data/MRPC/train.tsv",
        "validation": "glue_data/MRPC/dev.tsv",
    },
    delimiter="\t",
)

# Or use HuggingFace datasets directly (recommended)
mrpc = load_dataset("glue", "mrpc")
</syntaxhighlight>

== Related Pages ==
