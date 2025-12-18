{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Documentation]], [[domain::Maintenance]], [[domain::Utilities]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Automated utility script that adds or updates release dates and Transformers commit dates to model documentation cards in the Hugging Face Transformers repository.

=== Description ===
The add_dates.py script is a 427-line maintenance utility designed to manage date information in model documentation. It performs several key operations:

* Extracts paper release dates from Hugging Face Papers or arXiv links
* Determines when models were first added to Transformers using git history
* Inserts or updates date information in model documentation markdown files
* Validates existing date information for accuracy
* Converts arXiv paper links to Hugging Face Papers links when available
* Ensures copyright disclaimers are present in documentation files

The script uses the huggingface_hub library to fetch paper metadata, git commands to determine commit history, and regular expressions to parse and modify markdown documentation files. It can operate on individual models, all models, or only modified models (via git diff).

The date information is formatted as: "*This model was released on {release_date} and added to Hugging Face Transformers on {commit_date}.*" and is inserted after the copyright disclaimer in each model card.

=== Usage ===
Use this script when:
* Adding a new model to the Transformers library (automatically adds dates)
* Updating model documentation that's missing date information
* Validating that existing date information is correct
* Converting arXiv links to Hugging Face Papers links
* Running CI checks to ensure documentation completeness
* Maintaining consistency across model documentation

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' utils/add_dates.py

=== Signature ===
<syntaxhighlight lang="python">
# Main entry point
def main(all=False, models=None, check_only=False)

# Core functionality
def get_paper_link(model_card: str | None, path: str | None) -> str
def get_first_commit_date(model_name: str | None) -> str
def get_release_date(link: str) -> str
def insert_dates(model_card_list: list[str])
def replace_paper_links(file_path: str) -> bool

# Validation utilities
def check_missing_dates(model_card_list: list[str]) -> list[str]
def check_incorrect_dates(model_card_list: list[str]) -> list[str]

# Helper functions
def get_modified_cards() -> list[str]
def get_all_model_cards() -> list[str]
def check_file_exists_on_github(file_path: str) -> bool
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone script, typically run from command line
# But functions can be imported if needed:
from utils.add_dates import (
    get_paper_link,
    get_first_commit_date,
    insert_dates,
    check_missing_dates,
)
</syntaxhighlight>

== I/O Contract ==

=== Command Line Arguments ===
{| class="wikitable"
! Argument !! Type !! Required !! Description
|-
| --models || list[str] || No || Specific model names to process (without .md)
|-
| --all || flag || No || Process all model cards in docs directory
|-
| --check-only || flag || No || Only validate dates without modifying files
|}

=== Input Files ===
{| class="wikitable"
! Path !! Format !! Description
|-
| docs/source/en/model_doc/*.md || Markdown || Model documentation files
|-
| src/transformers/models/*/\__init__.py || Python || Model initialization files (for date lookup)
|}

=== Output Format ===
{| class="wikitable"
! Field !! Type !! Description
|-
| Modified files || Markdown || Updated .md files with date information
|-
| Console output || Text || Status messages and warnings
|-
| Exit code || int || 0 for success, non-zero for validation errors
|}

=== Date Format ===
{| class="wikitable"
! Pattern !! Example !! Description
|-
| YYYY-MM-DD || 2023-05-15 || ISO 8601 date format
|-
| {release_date} || {release_date} || Placeholder for missing paper date
|}

== Usage Examples ==

=== Check All Model Cards ===
<syntaxhighlight lang="python">
# Run from repository root
python utils/add_dates.py --check-only
# Output: Lists models with missing or incorrect dates
# Exits with error if issues found
</syntaxhighlight>

=== Process Specific Models ===
<syntaxhighlight lang="python">
# Add/update dates for specific models
python utils/add_dates.py --models bert gpt2 t5

# This will:
# 1. Find paper links in docs/source/en/model_doc/{bert,gpt2,t5}.md
# 2. Get paper release dates from HF Papers or arXiv
# 3. Get first commit dates from git history
# 4. Update the markdown files with date information
</syntaxhighlight>

=== Process Modified Models (Default) ===
<syntaxhighlight lang="python">
# Process only models that have been modified
python utils/add_dates.py

# Uses git diff to find modified files in docs/source/en/model_doc/
# Useful in CI or after making documentation changes
</syntaxhighlight>

=== Process All Models ===
<syntaxhighlight lang="python">
# Process every model in the repository
python utils/add_dates.py --all

# Iterates through all .md files in docs/source/en/model_doc/
# Useful for bulk updates or initial setup
</syntaxhighlight>

=== Using Functions Programmatically ===
<syntaxhighlight lang="python">
from utils.add_dates import get_paper_link, get_release_date, get_first_commit_date

# Get paper link from model card
paper_link = get_paper_link("bert", None)
# Returns: "https://huggingface.co/papers/1810.04805"

# Get release date from paper link
release_date = get_release_date(paper_link)
# Returns: "2018-10-11"

# Get first commit date for model
commit_date = get_first_commit_date("bert")
# Returns: "2019-08-07"
</syntaxhighlight>

=== Validate Specific Models ===
<syntaxhighlight lang="python">
from utils.add_dates import check_missing_dates, check_incorrect_dates

model_list = ["bert", "gpt2", "t5"]

# Check which have missing dates
missing = check_missing_dates(model_list)
print(f"Missing dates: {missing}")

# Check which have incorrect dates
incorrect = check_incorrect_dates(model_list)
print(f"Incorrect dates: {incorrect}")
</syntaxhighlight>

=== Integration in CI Pipeline ===
<syntaxhighlight lang="python">
# In GitHub Actions workflow
- name: Check documentation dates
  run: |
    python utils/add_dates.py --check-only
  # This will fail the CI if dates are missing or incorrect

# To fix issues found by CI:
- name: Update dates
  run: |
    python utils/add_dates.py --models $(cat failed_models.txt)
    git add docs/source/en/model_doc/*.md
    git commit -m "Update model documentation dates"
</syntaxhighlight>

=== Expected Output Format in Markdown ===
<syntaxhighlight lang="markdown">
<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); ...

-->
*This model was released on 2018-10-11 and added to Hugging Face Transformers on 2019-08-07.*

# BERT

## Overview

BERT is a language model...
</syntaxhighlight>

=== Handling New Models (Not Yet on GitHub) ===
<syntaxhighlight lang="python">
# For models not yet on main branch
python utils/add_dates.py --models new_model

# The script will:
# 1. Check if file exists on GitHub main branch
# 2. If not found (404), use today's date as commit date
# 3. Still fetch paper release date if available
# Output: "Model new_model not found on GitHub main branch, using today's date"
</syntaxhighlight>

=== Replace ArXiv Links with HF Papers ===
<syntaxhighlight lang="python">
from utils.add_dates import replace_paper_links

# This is called automatically by insert_dates()
file_path = "docs/source/en/model_doc/bert.md"
links_replaced = replace_paper_links(file_path)

if links_replaced:
    print("Updated paper links from arXiv to Hugging Face Papers")
# Also replaces hf.co with huggingface.co
</syntaxhighlight>

== Related Pages ==
* (Empty)
