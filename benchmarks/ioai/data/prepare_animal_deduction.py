#!/usr/bin/env python3
"""Prepare the IOAI 2026 Home Task 3 ("Animal Deduction") run root.

Downloads the official dataset folder from Google Drive and the task notebook
from GitHub, then lays out a leakage-sanitized run root:

    <root>/
    ├── task/                 # the agent's world (task_dir)
    │   ├── dataset/          # interactor.py, evaluate.py, pools, dev.csv,
    │   │                     # statement.md  — NO test1.csv
    │   ├── artifacts/        # big files (precompute tables, logs)
    │   └── submission/       # best-so-far solution.py (+ its data files)
    └── private/              # runner-only: never shown to the agent
        ├── dataset_pristine/ # untouched download (final-eval harness source)
        └── test1.csv         # held-out final-eval split

Usage:
    python -m benchmarks.ioai.data.prepare_animal_deduction --root <dir> \
        [--pristine-src <dir>]   # reuse an existing download (skips gdown)
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import gdown

DRIVE_FOLDER_ID = "1YheHvGfQw5YUa7MjdUF0hQC4sdtLZ5UC"
NOTEBOOK_RAW_URL = (
    "https://raw.githubusercontent.com/IOAI-official/IOAI-2026/main/"
    "Home%20Task/Home-Task-3.ipynb"
)
DATASET_FILES = [
    "interactor.py",
    "evaluate.py",
    "animals_pool.txt",
    "questions_pool.txt",
    "dev.csv",
    "test1.csv",
]
PRIVATE_FILES = {"test1.csv"}

STATEMENT_PREAMBLE = """\
# IOAI 2026 Home Task 3 — task statement

This is the official task notebook rendered as markdown. It was written for
Google Colab; in THIS environment the dataset and helper code are already on
disk in the `dataset/` directory next to this file (no gdown, no /content
paths), and the GPU is a local CUDA device. Everything else applies verbatim.

---
"""


def render_statement(notebook_path: Path) -> str:
    """Render the task notebook (markdown + code cells, in full) to markdown."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts = [STATEMENT_PREAMBLE]
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if "colab-badge" in source:
            continue
        if cell["cell_type"] == "markdown":
            parts.append(source)
        else:
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Run root to create")
    parser.add_argument(
        "--pristine-src",
        default=None,
        help="Directory already holding the dataset files (skips the Drive "
        "download; used to ship a cached copy to an offline-ish VM)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pristine = root / "private" / "dataset_pristine"
    dataset = root / "task" / "dataset"
    for sub in (pristine, dataset, root / "task" / "artifacts",
                root / "task" / "submission", root / "private"):
        sub.mkdir(parents=True, exist_ok=True)

    if args.pristine_src:
        src = Path(args.pristine_src).resolve()
        for name in DATASET_FILES:
            shutil.copy2(src / name, pristine / name)
    else:
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=str(pristine),
                              quiet=True, use_cookies=False)

    missing = [n for n in DATASET_FILES if not (pristine / n).is_file()]
    if missing:
        sys.exit(f"dataset download incomplete, missing: {missing}")

    notebook = pristine / "Home-Task-3.ipynb"
    if not notebook.is_file():
        urllib.request.urlretrieve(NOTEBOOK_RAW_URL, notebook)
    (dataset / "statement.md").write_text(render_statement(notebook),
                                          encoding="utf-8")

    for name in DATASET_FILES:
        if name in PRIVATE_FILES:
            shutil.copy2(pristine / name, root / "private" / name)
        else:
            shutil.copy2(pristine / name, dataset / name)

    # Leak check: nothing under task/ may name or contain the private split.
    leaked = [p for p in (root / "task").rglob("*") if p.name in PRIVATE_FILES]
    if leaked:
        sys.exit(f"LEAK: private files under task/: {leaked}")

    print(f"run root ready: {root}")
    print(f"  task_dir:    {root / 'task'}")
    print(f"  private csv: {root / 'private' / 'test1.csv'}")


if __name__ == "__main__":
    main()
