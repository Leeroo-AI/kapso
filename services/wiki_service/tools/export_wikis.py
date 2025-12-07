#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_wikis.py — Export wiki pages back to data/wikis directory.

This script provides two-way sync by exporting pages from MediaWiki back to
the source .mediawiki files. Pages are organized by their title prefix
(e.g., "repo_name/Page_Name" -> repo_name/Page_Name.mediawiki).

Usage:
    python3 export_wikis.py --base http://localhost:8080 --output ../../data/wikis

    # Export specific repo only
    python3 export_wikis.py --base http://localhost:8080 --output ../../data/wikis \
        --prefix huggingface_text-generation-inference

    # Use authentication for private wikis
    python3 export_wikis.py --base http://localhost:8080 --output ../../data/wikis \
        --user agent --password agentpass123
"""

import argparse
import os
import sys
import re
from pathlib import Path

# Import the MWClient from the same tools directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mw_client import MWClient


def get_all_pages(mw: MWClient, prefix: str | None = None) -> list[dict]:
    """
    Fetch all pages from the wiki using the API, including custom namespaces.
    
    Args:
        mw: Authenticated MWClient instance
        prefix: Optional prefix to filter pages (e.g., "repo_name/")
    
    Returns:
        List of page info dicts with 'title' key
    """
    # Namespaces to query: 0 (main) + custom namespaces (3000-3012)
    # Custom namespaces: Principle=3000, Workflow=3002, Implementation=3004,
    #                    Artifact=3006, Heuristic=3008, Environment=3010, Resource=3012
    namespaces = [0, 3000, 3002, 3004, 3006, 3008, 3010, 3012]
    
    all_pages = []
    
    for ns in namespaces:
        pages = []
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": str(ns),
            "format": "json",
        }
        
        # Add prefix filter if specified (only works in main namespace)
        if prefix and ns == 0:
            params["apprefix"] = prefix
        
        while True:
            r = mw.s.get(mw.api_url, params=params, timeout=mw.timeout)
            r.raise_for_status()
            data = r.json()
            
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            
            pages.extend(data.get("query", {}).get("allpages", []))
            
            # Handle pagination
            if "continue" in data:
                params["apcontinue"] = data["continue"]["apcontinue"]
            else:
                break
        
        all_pages.extend(pages)
    
    return all_pages


def get_page_content(mw: MWClient, title: str) -> str:
    """
    Fetch the wikitext content of a page.
    
    Args:
        mw: Authenticated MWClient instance
        title: Page title
    
    Returns:
        Wikitext content of the page
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    
    r = mw.s.get(mw.api_url, params=params, timeout=mw.timeout)
    r.raise_for_status()
    data = r.json()
    
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    
    # Extract content from nested response structure
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            raise RuntimeError(f"Page not found: {title}")
        revisions = page_data.get("revisions", [])
        if revisions:
            return revisions[0].get("slots", {}).get("main", {}).get("*", "")
    
    return ""


def title_to_filepath(title: str, output_dir: Path) -> Path:
    """
    Convert a wiki page title to a filesystem path.
    
    Handles namespace prefixes and converts them back to filename format:
        "Principle:repo_name/Token_Streaming" -> output_dir/repo_name/Principle_Token_Streaming.mediawiki
        "repo_name/Page_Name" -> output_dir/repo_name/Page_Name.mediawiki
        "Main Page" -> output_dir/_root/Main_Page.mediawiki
    
    Args:
        title: Wiki page title
        output_dir: Base output directory
    
    Returns:
        Path object for the output file
    """
    # Known namespaces that map to filename prefixes (1:1 mapping)
    known_namespaces = ["Principle", "Workflow", "Implementation", "Artifact", 
                        "Heuristic", "Environment", "Resource"]
    # No special mapping needed - namespace names match file prefixes directly
    ns_to_prefix = {}  # Empty - direct 1:1 mapping
    
    namespace = None
    rest = title
    
    # Check for namespace prefix (e.g., "Concept:...")
    if ":" in title:
        potential_ns, rest = title.split(":", 1)
        if potential_ns in known_namespaces:
            namespace = potential_ns
        else:
            # Not a known namespace, treat as regular title
            rest = title
    
    # Split the rest into repo name and page name
    if "/" in rest:
        parts = rest.split("/", 1)
        repo_name = parts[0]
        page_name = parts[1]
    else:
        # Pages without a repo prefix go into _root directory
        repo_name = "_root"
        page_name = rest
    
    # If there was a namespace, prepend it to the page name
    # Map namespace back to file prefix (e.g., Topic -> Concept)
    if namespace:
        file_prefix = ns_to_prefix.get(namespace, namespace)
        page_name = f"{file_prefix}_{page_name}"
    
    # Sanitize page name for filesystem (replace spaces with underscores)
    page_name = page_name.replace(" ", "_")
    
    # Remove any characters that are invalid in filenames
    page_name = re.sub(r'[<>:"|?*]', "_", page_name)
    
    return output_dir / repo_name / f"{page_name}.mediawiki"


def export_pages(mw: MWClient, output_dir: Path, prefix: str | None = None, 
                 dry_run: bool = False) -> tuple[int, int]:
    """
    Export all pages from the wiki to the output directory.
    
    Args:
        mw: Authenticated MWClient instance
        output_dir: Directory to write .mediawiki files
        prefix: Optional prefix to filter pages
        dry_run: If True, print what would be done without writing files
    
    Returns:
        Tuple of (exported_count, skipped_count)
    """
    print(f"📥 Fetching page list from {mw.api_url}...")
    pages = get_all_pages(mw, prefix)
    print(f"   Found {len(pages)} pages")
    
    exported = 0
    skipped = 0
    
    for page in pages:
        title = page["title"]
        
        # Skip system pages
        if title.startswith("MediaWiki:") or title.startswith("Special:"):
            skipped += 1
            continue
        
        filepath = title_to_filepath(title, output_dir)
        
        if dry_run:
            print(f"  [DRY RUN] Would export: {title} -> {filepath}")
            exported += 1
            continue
        
        try:
            content = get_page_content(mw, title)
            
            # Create directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            filepath.write_text(content, encoding="utf-8")
            print(f"  ✅ Exported: {title} -> {filepath.relative_to(output_dir)}")
            exported += 1
            
        except Exception as e:
            print(f"  ❌ Failed to export {title}: {e}")
            skipped += 1
    
    return exported, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Export wiki pages to data/wikis directory for two-way sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base", "-b",
        required=True,
        help="Base URL of the wiki (e.g., http://localhost:8080)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for .mediawiki files (e.g., ../../data/wikis)"
    )
    parser.add_argument(
        "--user", "-u",
        help="Username for authentication (optional for public wikis)"
    )
    parser.add_argument(
        "--password", "-p",
        help="Password for authentication"
    )
    parser.add_argument(
        "--prefix",
        help="Only export pages with this title prefix (e.g., 'repo_name/')"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without writing files"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).resolve()
    
    print("=" * 50)
    print("  Wiki Export Tool")
    print("=" * 50)
    print(f"  Source:  {args.base}")
    print(f"  Output:  {output_dir}")
    if args.prefix:
        print(f"  Prefix:  {args.prefix}")
    if args.dry_run:
        print("  Mode:    DRY RUN")
    print()
    
    # Create output directory if it doesn't exist
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    mw = MWClient(
        args.base,
        args.user or "",
        args.password or "",
        verify=not args.insecure
    )
    
    # Login if credentials provided
    if args.user and args.password:
        print("🔐 Logging in...")
        mw.login()
        print(f"   ✅ Logged in as {args.user}")
    else:
        # Still need to resolve API URL
        mw._resolve_and_get_login_token()
    
    print()
    
    # Export pages
    exported, skipped = export_pages(mw, output_dir, args.prefix, args.dry_run)
    
    print()
    print("=" * 50)
    print(f"  Export complete: {exported} exported, {skipped} skipped")
    print("=" * 50)


if __name__ == "__main__":
    main()

