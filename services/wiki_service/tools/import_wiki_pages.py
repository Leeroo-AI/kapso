#!/usr/bin/env python3
"""
import_wiki_pages.py - Import .md wiki files into MediaWiki via API

This script reads wiki pages from the data/wikis directory and imports them
into MediaWiki using the Action API. The .md files contain MediaWiki markup.

Usage:
    # Import from default wiki directory
    python import_wiki_pages.py
    
    # Import from specific directory with custom URL
    python import_wiki_pages.py --wiki-dir data/wikis --base http://localhost:8090
    
    # Force reimport (overwrite existing pages)
    python import_wiki_pages.py --force
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import mw_client
sys.path.insert(0, str(Path(__file__).parent))
from mw_client import MWClient

# Add sync package to path for shared transforms
sys.path.insert(0, str(Path(__file__).parent.parent))
from sync.transforms import prepare_content_for_wiki


# Map folder names to MediaWiki namespaces
FOLDER_TO_NAMESPACE = {
    "heuristics": "Heuristic",
    "workflows": "Workflow",
    "principles": "Principle",
    "implementations": "Implementation",
    "environments": "Environment",
}

# Folders to skip
SKIP_FOLDERS = {"_staging", "_reports", "_files"}


def build_page_title(folder_name: str, filename: str) -> str:
    """Build MediaWiki page title from folder and filename."""
    name = filename
    if name.endswith(".md"):
        name = name[:-3]
    if name.endswith(".mediawiki"):
        name = name[:-10]
    
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    
    # Get namespace from folder
    namespace = FOLDER_TO_NAMESPACE.get(folder_name, "")
    
    if namespace:
        return f"{namespace}:{name}"
    else:
        return f"{folder_name}/{name}"


def import_wikis(wiki_dir, mw, force=False, dry_run=False):
    """Import all wiki pages from directory into MediaWiki."""
    imported = 0
    skipped = 0
    failed = 0
    
    md_files = list(wiki_dir.rglob("*.md"))
    print(f"Found {len(md_files)} .md files in {wiki_dir}")
    
    for filepath in md_files:
        rel_path = filepath.relative_to(wiki_dir)
        parts = rel_path.parts
        if len(parts) < 1:
            continue
            
        folder_name = parts[0]
        
        if folder_name in SKIP_FOLDERS or folder_name.startswith("_"):
            continue
        
        page_title = build_page_title(folder_name, filepath.name)
        
        if "domain_tag" in page_title.lower():
            continue
        
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ✗ Failed to read {filepath}: {e}")
            failed += 1
            continue
        
        if not content.strip():
            skipped += 1
            continue

        # Derive namespace for metadata injection
        namespace = FOLDER_TO_NAMESPACE.get(folder_name, "")
        page_name = filepath.stem  # filename without extension

        # Apply content transforms (strip H1, convert source links, add metadata)
        # This mirrors import_wikis.sh behaviour for consistent rendering
        content = prepare_content_for_wiki(
            content,
            namespace=namespace if namespace else None,
            page_name=page_name if namespace else None,
        )
        
        print(f"  📄 {page_title}")
        
        if dry_run:
            imported += 1
            continue
        
        try:
            result = mw.edit(
                page_title,
                text=content,
                summary=f"Auto-imported from {rel_path}",
                createonly=not force,
            )
            
            edit_result = result.get("edit", {}).get("result", "")
            if edit_result == "Success":
                imported += 1
            else:
                if force:
                    result = mw.edit(page_title, text=content, summary=f"Re-imported from {rel_path}")
                    if result.get("edit", {}).get("result") == "Success":
                        imported += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    
        except RuntimeError as e:
            if "articleexists" in str(e):
                skipped += 1
            else:
                print(f"    ⚠ Error: {e}")
                failed += 1
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            failed += 1
    
    return imported, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Import wiki pages into MediaWiki")
    parser.add_argument("--wiki-dir", type=Path, default=Path("data/wikis"))
    parser.add_argument("--base", default=os.getenv("MW_BASE", "http://localhost:8090"))
    parser.add_argument("--user", default=os.getenv("MW_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("MW_PASS", "adminpass123"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    
    args = parser.parse_args()
    
    wiki_dir = args.wiki_dir
    if not wiki_dir.is_absolute():
        if not wiki_dir.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            wiki_dir = project_root / args.wiki_dir
    
    if not wiki_dir.exists():
        print(f"Error: Wiki directory not found: {wiki_dir}")
        sys.exit(1)
    
    print(f"{'=' * 60}")
    print(f"MediaWiki Import")
    print(f"Wiki directory: {wiki_dir}")
    print(f"MediaWiki URL: {args.base}")
    print(f"User: {args.user}")
    print(f"Force: {args.force}, Dry run: {args.dry_run}")
    print(f"{'=' * 60}")
    
    if args.dry_run:
        print("\n[DRY RUN]\n")
        imported, skipped, failed = import_wikis(wiki_dir, None, args.force, True)
    else:
        mw = MWClient(args.base, args.user, args.password, verify=not args.insecure)
        
        print("\nLogging into MediaWiki...")
        try:
            mw.login()
            print(f"✓ Logged in (API: {mw.api_url})")
        except Exception as e:
            print(f"✗ Login failed: {e}")
            sys.exit(1)
        
        print("\nImporting pages...")
        imported, skipped, failed = import_wikis(wiki_dir, mw, args.force, False)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Imported: {imported}, ○ Skipped: {skipped}, ✗ Failed: {failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
