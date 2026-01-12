#!/usr/bin/env python3
"""
Generate Workflow Repository Wiki Pages

Parses top repos from data/repo_research_results.csv and generates
wiki page files in data/workflow_repos/ directory.

Usage:
    python scripts/generate_workflow_repo_pages.py --top-k 10
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_csv(csv_path: str, top_k: int) -> List[Dict]:
    """Parse top repos from CSV file."""
    repos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= top_k:
                break
            
            # Skip failed research
            if row.get("research_status") != "success":
                continue
            
            repos.append(row)
    
    return repos


def generate_usage(row: Dict) -> str:
    """Generate usage section from row data."""
    score_reasoning = row.get("score_reasoning", "")
    
    # Extract applicability sentence
    if "directly applicable" in score_reasoning.lower():
        idx = score_reasoning.lower().find("directly applicable")
        # Find sentence boundaries
        start = score_reasoning.rfind(".", 0, idx) + 1
        end = score_reasoning.find(".", idx) + 1
        if end > idx:
            return score_reasoning[start:end].strip()
    
    # Fallback
    tags = row.get("tags", "").split("|")[:3]
    return f"Use this repository when working on {', '.join(tags)} tasks."


def generate_wiki_page(row: Dict) -> str:
    """Generate wiki page content for a repo."""
    name = row["name"]
    url = row["url"]
    description = row.get("description", "")
    tags = row.get("tags", "")
    stars = int(row.get("stars", 0))
    usage = generate_usage(row)
    
    # Parse domains from tags
    domains = [t.strip().replace(" ", "_") for t in tags.split("|") if t.strip()][:5]
    domains_str = ", ".join(f"[[domain::{d}]]" for d in domains)
    
    # Truncate overview if too long
    overview = description[:200] + "..." if len(description) > 200 else description
    
    return f"""# Workflow: {name}

{{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|{name}|{url}]]
|-
! Domains
| {domains_str}
|-
! Stars
| {stars:,}
|-
! Last Updated
| [[last_updated::2026-01-12]]
|}}

== Overview ==
{overview}

=== Description ===
{description}

=== Usage ===
{usage}

== Github URL ==
{url}

== Related Pages ==
* Repository serves as a starter template for ML workflows
* Can be cloned and adapted for specific use cases
"""


def main():
    parser = argparse.ArgumentParser(description="Generate workflow repo wiki pages")
    parser.add_argument(
        "--csv", 
        default="data/repo_research_results.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/workflow_repos/workflows",
        help="Output directory for wiki pages (should end with /workflows)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top repos to generate pages for"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse repos
    print(f"Parsing top {args.top_k} repos from {args.csv}...")
    repos = parse_csv(args.csv, args.top_k)
    print(f"Found {len(repos)} repos")
    
    # Generate wiki pages
    for row in repos:
        name = row["name"]
        # Sanitize filename
        filename = name.replace("-", "_").replace("/", "_") + ".md"
        filepath = output_dir / filename
        
        content = generate_wiki_page(row)
        filepath.write_text(content, encoding="utf-8")
        print(f"  ✓ {filename}")
    
    print(f"\nGenerated {len(repos)} wiki pages in {output_dir}/")


if __name__ == "__main__":
    main()
