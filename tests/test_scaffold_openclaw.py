# Test: Multi-language scaffold generation on openclaw/openclaw
#
# This tests that the repo ingestor correctly discovers and parses
# files across multiple languages (TypeScript, JavaScript, JSON, YAML, etc.)
# instead of only Python files.
#
# Usage:
#   conda run -n kapso_conda python tests/test_scaffold_openclaw.py
#
# What it does:
#   1. Clones https://github.com/openclaw/openclaw (shallow)
#   2. Runs generate_repo_scaffold() to build the RepoMap + _files/ details
#   3. Prints stats on discovered files by language
#   4. Cleans up the cloned repo

import shutil
import tempfile
from pathlib import Path

from kapso.knowledge_base.learners.ingestors.repo_ingestor.context_builder import (
    collect_source_files,
    generate_repo_scaffold,
    categorize_directories,
    find_key_files,
    parse_source_file,
    EXTENSION_TO_LANGUAGE,
    ALL_SUPPORTED_EXTENSIONS,
)
from kapso.knowledge_base.learners.ingestors.repo_ingestor.utils import (
    clone_repo,
    cleanup_repo,
)


def main():
    repo_url = "https://github.com/openclaw/openclaw"
    branch = "main"
    
    print(f"{'='*60}")
    print(f"Multi-language scaffold test: {repo_url}")
    print(f"{'='*60}\n")
    
    # Step 1: Clone the repo
    print("[1/4] Cloning repository (shallow)...")
    try:
        repo_path = clone_repo(repo_url, branch)
    except RuntimeError:
        # Fallback: try default branch
        print("  'main' branch failed, trying default branch...")
        repo_path = clone_repo(repo_url)
    print(f"  Cloned to: {repo_path}\n")
    
    # Step 2: Create a temp wiki dir for output
    wiki_dir = Path(tempfile.mkdtemp(prefix="kapso_test_wiki_"))
    print(f"[2/4] Wiki output dir: {wiki_dir}\n")
    
    try:
        # Step 3: Test categorize_directories
        print("[3/4] Analyzing repository structure...")
        categories = categorize_directories(repo_path)
        key_files = find_key_files(repo_path)
        
        print(f"\n  Directory categories:")
        for cat, dirs in categories.items():
            if dirs:
                print(f"    {cat}: {', '.join(sorted(dirs))}")
        
        print(f"\n  Key files detected:")
        for name, val in key_files.items():
            if val:
                print(f"    {name}: {val}")
        
        # Step 4: Collect source files and show stats
        print(f"\n[4/4] Collecting source files...")
        source_files = collect_source_files(repo_path, max_files=500)
        
        # Count by language
        lang_counts = {}
        lang_lines = {}
        for f in source_files:
            lang = f["ast_info"].get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            lang_lines[lang] = lang_lines.get(lang, 0) + f["ast_info"]["lines"]
        
        # Count by category
        cat_counts = {}
        for f in source_files:
            cat = f["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        print(f"\n  Total source files collected: {len(source_files)}")
        print(f"  Total lines: {sum(f['ast_info']['lines'] for f in source_files):,}")
        
        print(f"\n  Files by language:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            lines = lang_lines[lang]
            print(f"    {lang:15s} {count:5d} files  ({lines:>8,} lines)")
        
        print(f"\n  Files by category:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:15s} {count:5d} files")
        
        # Step 5: Generate scaffold
        print(f"\n  Generating scaffold (RepoMap + detail files)...")
        scaffold_content = generate_repo_scaffold(
            repo_path=repo_path,
            repo_name="Openclaw_Openclaw",
            repo_url=repo_url,
            branch=branch,
            wiki_dir=wiki_dir,
        )
        
        # Write the RepoMap file (generate_repo_scaffold returns content,
        # the caller is responsible for writing it — same as __init__.py does)
        repo_map = wiki_dir / "_RepoMap_Openclaw_Openclaw.md"
        repo_map.write_text(scaffold_content, encoding="utf-8")
        
        # Count generated files
        files_dir = wiki_dir / "_files"
        detail_files = list(files_dir.glob("*.md")) if files_dir.exists() else []
        
        print(f"  Detail files generated: {len(detail_files)}")
        print(f"  RepoMap exists: {repo_map.exists()}")
        if repo_map.exists():
            map_lines = len(repo_map.read_text().splitlines())
            print(f"  RepoMap lines: {map_lines}")
        
        # Show a few example detail files to verify content
        print(f"\n  Sample detail files:")
        for detail in detail_files[:5]:
            content = detail.read_text()
            # Extract the language from the detail file table
            lang_line = [l for l in content.splitlines() if "| Language |" in l]
            lang_str = lang_line[0].split("|")[2].strip() if lang_line else "n/a"
            # Extract the file name from the first line
            first_line = content.splitlines()[0] if content else ""
            print(f"    {detail.name} -> language: {lang_str}")
        
        # Verify: non-Python files were collected
        non_python = [f for f in source_files if f["ast_info"].get("language") != "python"]
        python_files = [f for f in source_files if f["ast_info"].get("language") == "python"]
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Python files:     {len(python_files)}")
        print(f"  Non-Python files: {len(non_python)}")
        print(f"  Detail pages:     {len(detail_files)}")
        
        if len(non_python) > 0:
            print(f"\n  ✅ PASS: Multi-language support working!")
            print(f"  Found {len(non_python)} non-Python files across {len(lang_counts)} languages.")
        else:
            print(f"\n  ❌ FAIL: No non-Python files found!")
            print(f"  This repo should have TypeScript, JSON, etc.")
        
    finally:
        # Cleanup
        print(f"\n  Cleaning up...")
        cleanup_repo(repo_path)
        shutil.rmtree(wiki_dir, ignore_errors=True)
        print(f"  Done.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
