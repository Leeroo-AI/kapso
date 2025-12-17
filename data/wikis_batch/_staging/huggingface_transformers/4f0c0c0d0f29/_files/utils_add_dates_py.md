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

**Purpose:** Manages release and commit dates in model documentation cards by automatically adding or updating date metadata in markdown files.

**Mechanism:** Scans model documentation files (docs/source/en/model_doc/*.md), extracts paper links (from HuggingFace Papers or arXiv), retrieves paper publication dates via the huggingface_hub API, determines when models were added to Transformers via git history, and injects/updates date information into docstrings. Includes a check mode to verify dates are present and correct, plus auto-replacement of arXiv links with HuggingFace Papers links when available.

**Significance:** Documentation maintenance utility that ensures model cards contain accurate metadata about when models were released and added to the library, improving documentation quality and helping users understand model timelines.
