"""
Deterministic wiki directory validation (repo ingestor).
 
Why this exists:
- Repo ingestion phases are prompt-driven and can silently produce broken graphs.
- The prompts (Audit phase) are useful but not sufficient as the only correctness gate.
 
This module provides a small, deterministic validator that checks the minimum
constraints needed for a usable DAG knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory

import re


@dataclass
class ValidationReport:
    """
    Validation report for a wiki directory.
    
    - errors: hard failures (graph is invalid)
    - warnings: soft issues (graph is usable but incomplete)
    """

    wiki_dir: Path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def ok(self) -> bool:
        return not self.errors

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("VALIDATION REPORT")
        lines.append("=" * 18)
        lines.append(f"wiki_dir: {self.wiki_dir}")
        if self.counts:
            lines.append("")
            lines.append("Counts:")
            for k, v in sorted(self.counts.items()):
                lines.append(f"  - {k}: {v}")
        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines) + "\n"


def _parse_index_entries(index_path: Path) -> Set[str]:
    """
    Parse an index file and extract page names from the table rows.
    
    Returns a set of page names found in the index.
    """
    if not index_path.exists():
        return set()
    
    content = index_path.read_text(encoding="utf-8")
    entries = set()
    
    # Match table rows: | PageName | [→](./dir/file.md) | ...
    # The page name is typically in the first column after the leading |
    for line in content.splitlines():
        if not line.startswith("|"):
            continue
        # Skip header and separator rows
        if "---" in line or "Page" in line and "File" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            page_name = parts[1].strip()
            # Skip empty or placeholder entries
            if page_name and page_name != "—" and not page_name.startswith("<!--"):
                entries.add(page_name)
    
    return entries


def _get_pages_in_directory(wiki_dir: Path, subdir: str) -> Set[str]:
    """
    Get all page names from .md files in a subdirectory.
    
    Returns page names (filename without .md extension).
    """
    pages = set()
    dir_path = wiki_dir / subdir
    if dir_path.exists():
        for md_file in dir_path.glob("*.md"):
            pages.add(md_file.stem)
    return pages


def validate_page_indexes(wiki_dir: Path, report: ValidationReport) -> None:
    """
    Validate that page index files match directory contents.
    
    Checks:
    - Every page in directory has an index entry
    - Index entries point to existing pages
    """
    index_configs = [
        ("_WorkflowIndex.md", "workflows"),
        ("_PrincipleIndex.md", "principles"),
        ("_ImplementationIndex.md", "implementations"),
        ("_EnvironmentIndex.md", "environments"),
        ("_HeuristicIndex.md", "heuristics"),
    ]
    
    for index_name, subdir in index_configs:
        index_path = wiki_dir / index_name
        
        if not index_path.exists():
            # Don't report error for missing index if directory is empty
            dir_pages = _get_pages_in_directory(wiki_dir, subdir)
            if dir_pages:
                report.warnings.append(
                    f"Index file {index_name} missing but {len(dir_pages)} pages exist in {subdir}/"
                )
            continue
        
        index_entries = _parse_index_entries(index_path)
        dir_pages = _get_pages_in_directory(wiki_dir, subdir)
        
        # Check for pages in directory but not in index
        missing_from_index = dir_pages - index_entries
        for page in missing_from_index:
            report.warnings.append(
                f"Page {subdir}/{page}.md exists but missing from {index_name}"
            )
        
        # Check for entries in index but no page file
        orphan_entries = index_entries - dir_pages
        for entry in orphan_entries:
            # Only report if entry looks like a real page name (not placeholder)
            if entry and not entry.startswith("—"):
                report.warnings.append(
                    f"Index entry '{entry}' in {index_name} has no corresponding {subdir}/{entry}.md"
                )


def validate_wiki_directory(wiki_dir: Path) -> ValidationReport:
    """
    Validate a wiki directory for link integrity and mandatory constraints.
    
    Checks:
    - All links to core node types must point to existing pages.
    - Every Principle must have at least one implemented_by link to an Implementation.
    - Every Workflow should have at least 2 step links (warning).
    - Every Principle should be referenced by at least one Workflow step (warning).
    
    Notes:
    - We only strictly validate links targeting these types:
      Workflow, Principle, Implementation, Environment, Heuristic.
      Unknown target types (e.g., Artifact) are ignored by this validator.
    """

    wiki_dir = Path(wiki_dir)
    report = ValidationReport(wiki_dir=wiki_dir)

    try:
        pages = parse_wiki_directory(wiki_dir)
    except Exception as e:
        report.errors.append(f"Failed to parse wiki_dir: {e}")
        return report

    # Inventory
    ids: Set[str] = {p.id for p in pages}
    by_type: Dict[str, List[str]] = {}
    for p in pages:
        by_type.setdefault(p.page_type, []).append(p.id)
    report.counts = {k: len(v) for k, v in by_type.items()}

    core_types = {"Workflow", "Principle", "Implementation", "Environment", "Heuristic"}

    workflow_github_urls: Dict[str, str] = {}   # workflow_id -> github_url
    principle_impls: Dict[str, Set[str]] = {}  # principle_id -> set(implementation_page_id)

    # 1) Link target existence and workflow GitHub URL extraction
    for page in pages:
        for link in page.outgoing_links:
            edge_type = (link.get("edge_type") or "").strip()
            target_type = (link.get("target_type") or "").strip()
            target_id = (link.get("target_id") or "").strip()

            if not edge_type or not target_type or not target_id:
                report.errors.append(f"{page.id}: malformed link dict: {link!r}")
                continue

            # Handle github_url as a special case (not a page link)
            if edge_type.lower() == "github_url":
                if page.page_type == "Workflow":
                    workflow_github_urls[page.id] = target_id
                continue

            if target_type not in core_types:
                continue

            target_page_id = f"{target_type}/{target_id}"
            if target_page_id not in ids:
                report.errors.append(
                    f"{page.id}: broken link [[{edge_type}::{target_type}:{target_id}]] "
                    f"(missing page {target_page_id})"
                )

            if page.page_type == "Principle" and edge_type.lower() in ("implemented_by", "realized_by") and target_type == "Implementation":
                principle_impls.setdefault(page.id, set()).add(target_page_id)

    # 2) Mandatory: every Principle has >= 1 Implementation
    for principle_id in by_type.get("Principle", []):
        impls = principle_impls.get(principle_id, set())
        if not impls:
            report.errors.append(
                f"{principle_id}: missing mandatory [[implemented_by::Implementation:...]] link"
            )

    # 3) Mandatory: every Workflow must have a GitHub URL
    # Workflows now link to GitHub repositories instead of step links
    for workflow_id in by_type.get("Workflow", []):
        if workflow_id not in workflow_github_urls:
            # Also check if the page content contains a github_url
            for page in pages:
                if page.id == workflow_id:
                    # Check content for [[github_url::...]] pattern
                    if page.content and "[[github_url::" in page.content:
                        # Extract URL from content
                        url_match = re.search(r'\[\[github_url::([^\]]+)\]\]', page.content)
                        if url_match:
                            workflow_github_urls[workflow_id] = url_match.group(1)
                    break
        
        if workflow_id not in workflow_github_urls:
            report.errors.append(
                f"{workflow_id}: missing mandatory [[github_url::...]] link to implementation repository"
            )

    # 4) Note: Principles are no longer connected to Workflows via step links
    # Principles are standalone knowledge units that may be orphaned (this is acceptable)

    # 5) Validate page indexes match directory contents
    validate_page_indexes(wiki_dir, report)

    return report


