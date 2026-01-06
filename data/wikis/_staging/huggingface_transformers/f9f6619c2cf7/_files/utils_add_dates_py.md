# File: `utils/add_dates.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 427 |
| Functions | `check_file_exists_on_github`, `get_modified_cards`, `get_paper_link`, `get_first_commit_date`, `get_release_date`, `replace_paper_links`, `check_missing_dates`, `check_incorrect_dates`, `... +3 more` |
| Imports | argparse, datetime, huggingface_hub, os, re, subprocess, urllib |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages release and commit dates in model documentation by automatically adding or updating date information in model card Markdown files.

**Mechanism:** The script operates on model documentation files in `docs/source/en/model_doc/`. It extracts paper links from model cards, fetches release dates from HuggingFace Papers API or ArXiv, determines when models were first committed to the repository using git history, and inserts/updates a standardized date line in the documentation. The script can check for missing or incorrect dates, replace ArXiv links with HuggingFace paper links when available, and ensure copyright disclaimers are present. It handles both individual model cards and batch processing of all/modified cards.

**Significance:** This utility maintains documentation quality and consistency by automating the tedious task of tracking when models were released and added to the library. The date information helps users understand model timelines and is used during CI checks to ensure documentation completeness. It's part of the repository's automated documentation maintenance infrastructure.
