#!/usr/bin/env python3
"""URL -> run-root preflight for Kaggle competitions.

Turns a competition URL into the run root the runner consumes:
    <root>/task/dataset/    competition data + statement.md
    <root>/task/kaggle.json {"competition": <slug>}

The mechanical scaffolding lives here (slug parse, authenticated
`kaggle competitions download`, kaggle.json). The one agentic step is a single
`codex exec` that inspects the downloaded files plus the competition's own
pages and authors dataset/statement.md per preflight_spec.md — the sole
per-competition variable, since config.yaml carries everything else.

Usage:
    python -m benchmarks.kaggle.preflight \
        --url https://www.kaggle.com/competitions/<slug>/overview \
        --root ~/kaggle_run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import yaml
from dotenv import load_dotenv

load_dotenv()

from kapso.execution.coding_agents.factory import CodingAgentFactory

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
SPEC_PATH = os.path.join(os.path.dirname(__file__), "preflight_spec.md")
SLUG_RE = re.compile(r"kaggle\.com/(?:c|competitions)/([A-Za-z0-9][A-Za-z0-9_-]*)")


def slug_from_url(url: str) -> str:
    """Extract the competition slug from a Kaggle competition URL."""
    match = SLUG_RE.search(url)
    if not match:
        raise ValueError(
            f"could not extract a competition slug from {url!r} — expected a "
            "https://www.kaggle.com/competitions/<slug>/... URL"
        )
    return match.group(1)


def download_competition(slug: str, dataset_dir: str) -> None:
    """Download + unzip the competition data via the authenticated kaggle CLI."""
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise FileNotFoundError("kaggle CLI not on PATH — cannot download data")
    os.makedirs(dataset_dir, exist_ok=True)
    proc = subprocess.run(
        [kaggle_bin, "competitions", "download", "-c", slug, "-p", dataset_dir],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`kaggle competitions download -c {slug}` failed (exit "
            f"{proc.returncode}). A rules-acceptance error means the "
            "competition rules must be accepted once on kaggle.com first.\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    for name in os.listdir(dataset_dir):
        if name.endswith(".zip"):
            zip_path = os.path.join(dataset_dir, name)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(dataset_dir)
            os.remove(zip_path)


def build_prompt(url: str, slug: str, dataset_dir: str, statement_path: str) -> str:
    """Dynamic per-competition header + the fixed statement-authoring spec."""
    spec = open(SPEC_PATH, encoding="utf-8").read()
    header = (
        "# Preflight assignment\n\n"
        f"Competition URL: {url}\n"
        f"Competition slug: `{slug}`\n"
        f"Dataset directory (data already downloaded here via the "
        f"authenticated kaggle CLI): {dataset_dir}\n"
        f"Write the statement to: {statement_path}\n\n"
    )
    return header + spec


def run_preflight(url: str, root: str, mode: str = "KAGGLE") -> str:
    slug = slug_from_url(url)
    root = os.path.abspath(root)
    task_dir = os.path.join(root, "task")
    dataset_dir = os.path.join(task_dir, "dataset")
    statement_path = os.path.join(dataset_dir, "statement.md")
    os.makedirs(task_dir, exist_ok=True)

    download_competition(slug, dataset_dir)
    with open(os.path.join(task_dir, "kaggle.json"), "w") as f:
        json.dump({"competition": slug}, f, indent=2)

    with open(CONFIG_PATH) as f:
        preflight_cfg = yaml.safe_load(f)["modes"][mode]["preflight"]

    config = CodingAgentFactory.build_config(
        agent_type="codex",
        model=preflight_cfg["model"],
        agent_specific={
            "effort": preflight_cfg["effort"],
            "timeout": preflight_cfg["timeout"],
            "web_search": preflight_cfg["web_search"],
            "streaming": True,
        },
    )
    agent = CodingAgentFactory.create(config)
    agent.initialize(task_dir)
    result = agent.generate_code(
        build_prompt(url, slug, dataset_dir, statement_path),
        timeout_seconds=preflight_cfg["timeout"],
    )

    if not result.success:
        sys.exit(f"preflight codex run failed: {result.error}")
    if not (os.path.isfile(statement_path) and os.path.getsize(statement_path) > 0):
        sys.exit(f"{statement_path} missing or empty after preflight")
    data_entries = [n for n in os.listdir(dataset_dir) if n != "statement.md"]
    if not data_entries:
        sys.exit(f"{dataset_dir} holds no competition data beyond statement.md")

    print(f"\nrun root ready: {root}")
    print(f"  competition: {slug}")
    print(f"  statement:   {statement_path} "
          f"({os.path.getsize(statement_path)} bytes)")
    print(f"  data:        {sorted(data_entries)}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="Kaggle competition URL")
    parser.add_argument("--root", required=True,
                        help="Run root to create (consumed by runner.py --root)")
    parser.add_argument("--mode", default="KAGGLE")
    args = parser.parse_args()
    run_preflight(args.url, args.root, args.mode)


if __name__ == "__main__":
    main()
