# Context Builder for Repository Understanding
#
# Provides utilities to pre-compute repository structure and context
# that helps the agent work more efficiently in subsequent phases.
#
# Structure:
# - _RepoMap_{repo_name}.md: Compact index with file list and status
# - _files/{filename}.md: Per-file detail with AST info and Understanding
#
# This split design makes it easy to:
# - See at a glance what's explored vs remaining
# - Edit individual files without touching the whole index
# - Navigate by following references
#
# Key functions:
# - generate_repo_scaffold(): Create compact index + per-file detail files
# - parse_python_file(): Extract classes, functions, imports from a Python file

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_python_file(file_path: Path) -> Dict:
    """
    Parse a Python file and extract structural information.
    
    Returns dict with:
    - lines: Line count
    - classes: List of class names with their public methods
    - functions: List of top-level function names
    - imports: List of imported modules/symbols
    - has_main: Whether file has if __name__ == "__main__"
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = len(content.splitlines())
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        imports = []
        has_main = False
        
        for node in ast.walk(tree):
            # Extract class definitions
            if isinstance(node, ast.ClassDef):
                # Get public methods (not starting with _)
                methods = [
                    n.name for n in node.body 
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not n.name.startswith("_")
                ]
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "line": node.lineno,
                })
        
        # Get top-level functions only (not nested in classes)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Include public functions
                if not node.name.startswith("_"):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                    })
            
            # Check for if __name__ == "__main__"
            if isinstance(node, ast.If):
                try:
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == "__name__"):
                        has_main = True
                except:
                    pass
            
            # Extract imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])
        
        # Deduplicate imports
        imports = sorted(set(imports))
        
        return {
            "lines": lines,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "has_main": has_main,
            "parse_error": None,
        }
        
    except SyntaxError as e:
        return {
            "lines": 0,
            "classes": [],
            "functions": [],
            "imports": [],
            "has_main": False,
            "parse_error": str(e),
        }
    except Exception as e:
        return {
            "lines": 0,
            "classes": [],
            "functions": [],
            "imports": [],
            "has_main": False,
            "parse_error": str(e),
        }


def categorize_directories(repo_path: Path) -> Dict[str, List[str]]:
    """
    Categorize directories in the repo by their likely purpose.
    
    Returns dict with:
    - package_dirs: Main source code directories
    - example_dirs: Example/demo directories
    - test_dirs: Test directories
    - doc_dirs: Documentation directories
    """
    categories = {
        "package_dirs": [],
        "example_dirs": [],
        "test_dirs": [],
        "doc_dirs": [],
    }
    
    # Common patterns
    example_patterns = {"example", "examples", "demo", "demos", "sample", "samples", "notebook", "notebooks", "scripts"}
    test_patterns = {"test", "tests", "testing", "spec", "specs"}
    doc_patterns = {"doc", "docs", "documentation", "wiki"}
    skip_patterns = {"__pycache__", ".git", ".github", "node_modules", "venv", ".venv", "env", ".env", "build", "dist", "egg-info"}
    
    for item in repo_path.iterdir():
        if not item.is_dir():
            continue
        
        name_lower = item.name.lower()
        
        # Skip hidden and build directories
        if item.name.startswith(".") or any(p in name_lower for p in skip_patterns):
            continue
        
        if name_lower in example_patterns or "example" in name_lower:
            categories["example_dirs"].append(item.name)
        elif name_lower in test_patterns:
            categories["test_dirs"].append(item.name)
        elif name_lower in doc_patterns:
            categories["doc_dirs"].append(item.name)
        elif (item / "__init__.py").exists():
            # It's a Python package
            categories["package_dirs"].append(item.name)
        elif any(item.glob("*.py")):
            # Has Python files, might be a package or scripts
            categories["package_dirs"].append(item.name)
    
    return categories


def find_key_files(repo_path: Path) -> Dict[str, Optional[str]]:
    """
    Find important files in the repository.
    """
    key_files = {
        "readme": None,
        "setup": None,
        "requirements": None,
        "pyproject": None,
        "dockerfile": None,
    }
    
    for f in repo_path.iterdir():
        if not f.is_file():
            continue
        
        name_lower = f.name.lower()
        
        if name_lower.startswith("readme"):
            key_files["readme"] = f.name
        elif name_lower == "setup.py":
            key_files["setup"] = f.name
        elif name_lower == "requirements.txt":
            key_files["requirements"] = f.name
        elif name_lower == "pyproject.toml":
            key_files["pyproject"] = f.name
        elif name_lower == "dockerfile":
            key_files["dockerfile"] = f.name
    
    return key_files


def collect_python_files(repo_path: Path, max_files: int = 200) -> List[Dict]:
    """
    Collect all Python files in the repository with their AST info.
    
    Returns list of dicts with:
    - path: Relative path from repo root
    - category: package/example/test/other
    - ast_info: Parsed AST information
    """
    categories = categorize_directories(repo_path)
    
    # Build set of categorized directories
    example_dirs = set(categories["example_dirs"])
    test_dirs = set(categories["test_dirs"])
    package_dirs = set(categories["package_dirs"])
    
    files = []
    
    # Skip patterns
    skip_patterns = {"__pycache__", ".git", "node_modules", "venv", ".venv", "build", "dist"}
    
    for py_file in repo_path.rglob("*.py"):
        # Skip unwanted directories
        if any(p in py_file.parts for p in skip_patterns):
            continue
        
        rel_path = py_file.relative_to(repo_path)
        rel_str = str(rel_path)
        
        # Determine category
        first_dir = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
        
        if first_dir in example_dirs:
            category = "example"
        elif first_dir in test_dirs:
            category = "test"
        elif first_dir in package_dirs:
            category = "package"
        else:
            category = "other"
        
        # Parse the file
        ast_info = parse_python_file(py_file)
        
        files.append({
            "path": rel_str,
            "category": category,
            "ast_info": ast_info,
        })
        
        if len(files) >= max_files:
            logger.warning(f"Reached max file limit ({max_files}), stopping collection")
            break
    
    # Sort by category priority then path
    category_order = {"package": 0, "example": 1, "test": 2, "other": 3}
    files.sort(key=lambda f: (category_order.get(f["category"], 99), f["path"]))
    
    return files


def _file_path_to_detail_name(file_path: str) -> str:
    """Convert a file path to a safe detail filename."""
    # Replace path separators and dots with underscores
    safe_name = file_path.replace("/", "_").replace("\\", "_").replace(".", "_")
    return f"{safe_name}.md"


def generate_file_detail(
    file_path: str,
    ast_info: Dict,
    category: str,
) -> str:
    """
    Generate the markdown content for a single file's detail page.
    
    This is a small, focused file (~30 lines) that's easy to edit.
    """
    lines = []
    
    lines.append(f"# File: `{file_path}`")
    lines.append("")
    lines.append(f"**Category:** {category}")
    lines.append("")
    
    # AST info table
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Lines | {ast_info['lines']} |")
    
    if ast_info["classes"]:
        class_names = [c["name"] for c in ast_info["classes"]]
        lines.append(f"| Classes | `{'`, `'.join(class_names)}` |")
    
    if ast_info["functions"]:
        func_names = [f["name"] for f in ast_info["functions"]]
        if len(func_names) > 8:
            func_display = func_names[:8] + [f"... +{len(func_names)-8} more"]
        else:
            func_display = func_names
        lines.append(f"| Functions | `{'`, `'.join(func_display)}` |")
    
    if ast_info["imports"]:
        imports = ast_info["imports"][:10]
        if len(ast_info["imports"]) > 10:
            imports.append(f"... +{len(ast_info['imports'])-10} more")
        lines.append(f"| Imports | {', '.join(imports)} |")
    
    if ast_info["has_main"]:
        lines.append("| Executable | Yes (`__main__`) |")
    
    lines.append("")
    
    # Understanding section (to be filled by agent)
    lines.append("## Understanding")
    lines.append("")
    lines.append("**Status:** ⬜ Not explored")
    lines.append("")
    lines.append("**Purpose:** <!-- What does this file do? -->")
    lines.append("")
    lines.append("**Mechanism:** <!-- How does it accomplish its purpose? -->")
    lines.append("")
    lines.append("**Significance:** <!-- Why does this file exist? Core component or utility? -->")
    lines.append("")
    
    # Relationships
    lines.append("## Relationships")
    lines.append("")
    lines.append("**Depends on:** <!-- What files in this repo does it import? -->")
    lines.append("")
    lines.append("**Used by:** <!-- What files import this? -->")
    lines.append("")
    
    return "\n".join(lines)


def generate_repo_index(
    repo_name: str,
    repo_url: str,
    branch: str,
    categories: Dict[str, List[str]],
    key_files: Dict[str, Optional[str]],
    python_files: List[Dict],
    total_lines: int,
) -> str:
    """
    Generate a compact index file listing all files with links to details.
    
    This file is small (~100-200 lines) and easy to scan.
    """
    lines = []
    
    # Header
    lines.append(f"# Repository Map: {repo_name}")
    lines.append("")
    lines.append("> **Compact index** of repository files.")
    lines.append("> Each file has a detail page in `_files/` with Understanding to fill.")
    lines.append("> Mark files as ✅ explored in the table below as you complete them.")
    lines.append("")
    
    # Metadata
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Repository | {repo_url} |")
    lines.append(f"| Branch | {branch} |")
    lines.append(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")
    lines.append(f"| Python Files | {len(python_files)} |")
    lines.append(f"| Total Lines | {total_lines:,} |")
    lines.append(f"| Explored | 0/{len(python_files)} |")
    lines.append("")
    
    # Directory structure
    lines.append("## Structure")
    lines.append("")
    if categories["package_dirs"]:
        lines.append(f"📦 **Packages:** {', '.join(sorted(categories['package_dirs']))}")
    if categories["example_dirs"]:
        lines.append(f"📝 **Examples:** {', '.join(sorted(categories['example_dirs']))}")
    if categories["test_dirs"]:
        lines.append(f"🧪 **Tests:** {', '.join(sorted(categories['test_dirs']))}")
    lines.append("")
    
    if key_files.get("readme"):
        lines.append(f"📖 README: `{key_files['readme']}`")
    if key_files.get("pyproject") or key_files.get("setup"):
        setup_file = key_files.get("pyproject") or key_files.get("setup")
        lines.append(f"⚙️ Setup: `{setup_file}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # File index table grouped by category
    category_icons = {
        "package": "📦",
        "example": "📝",
        "test": "🧪",
        "other": "📄",
    }
    
    current_category = None
    
    for file_info in python_files:
        cat = file_info["category"]
        path = file_info["path"]
        ast_info = file_info["ast_info"]
        detail_name = _file_path_to_detail_name(path)
        
        # Category header with new table
        if cat != current_category:
            if current_category is not None:
                lines.append("")  # End previous section
            current_category = cat
            icon = category_icons.get(cat, "📄")
            category_title = cat.title()
            lines.append(f"## {icon} {category_title} Files")
            lines.append("")
            # Columns:
            # - Status: ⬜ pending → ✅ explored (Phase 0)
            # - File: path to the file
            # - Lines: line count
            # - Purpose: brief description filled by Phase 0 (3-5 words)
            # - Coverage: natural language showing which wiki pages cover this file
            # - Details: link to per-file detail page
            lines.append("| Status | File | Lines | Purpose | Coverage | Details |")
            lines.append("|--------|------|-------|---------|----------|---------|")
        
        # Table row
        # - Purpose: Initially "—", filled during Phase 0 with brief description
        # - Coverage: Initially "—", updated by later phases with page names that cover this file
        lines.append(f"| ⬜ | `{path}` | {ast_info['lines']} | — | — | [→](./_files/{detail_name}) |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Reference to page indexes (replacing inline Agent Notes)
    lines.append("## Page Indexes")
    lines.append("")
    lines.append("Each page type has its own index file for tracking and integrity checking:")
    lines.append("")
    lines.append("| Index | Description |")
    lines.append("|-------|-------------|")
    lines.append("| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |")
    lines.append("| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |")
    lines.append("| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |")
    lines.append("| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |")
    lines.append("| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |")
    lines.append("")
    
    return "\n".join(lines)


def generate_repo_scaffold(
    repo_path: Path,
    repo_name: str,
    repo_url: str,
    branch: str = "main",
    wiki_dir: Optional[Path] = None,
) -> str:
    """
    Generate repository scaffold with compact index + per-file details.
    
    Creates:
    - _RepoMap_{repo_name}.md: Compact index (~150 lines)
    - _files/{filename}.md: Per-file detail (~30 lines each)
    
    This split design makes navigation and editing much easier for the agent.
    
    Args:
        repo_path: Path to the repository
        repo_name: Name of the repository
        repo_url: URL of the repository
        branch: Git branch
        wiki_dir: Where to write the _files/ directory (if provided)
    
    Returns:
        Content of the index file (_RepoMap.md)
    """
    categories = categorize_directories(repo_path)
    key_files = find_key_files(repo_path)
    python_files = collect_python_files(repo_path)
    
    total_lines = sum(f["ast_info"]["lines"] for f in python_files)
    
    # Generate and write per-file detail files if wiki_dir provided
    if wiki_dir:
        files_dir = wiki_dir / "_files"
        files_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in python_files:
            detail_content = generate_file_detail(
                file_path=file_info["path"],
                ast_info=file_info["ast_info"],
                category=file_info["category"],
            )
            detail_name = _file_path_to_detail_name(file_info["path"])
            detail_path = files_dir / detail_name
            detail_path.write_text(detail_content, encoding="utf-8")
        
        logger.info(f"Wrote {len(python_files)} file detail pages to {files_dir}")
        
        # Generate page index files (for tracking wiki pages by type)
        generate_page_indexes(wiki_dir, repo_name)
        
        # Create reports directory for phase execution summaries
        ensure_reports_directory(wiki_dir)
    
    # Generate index
    index_content = generate_repo_index(
        repo_name=repo_name,
        repo_url=repo_url,
        branch=branch,
        categories=categories,
        key_files=key_files,
        python_files=python_files,
        total_lines=total_lines,
    )
    
    return index_content


def get_repo_map_path(wiki_dir: Path, repo_name: str) -> Path:
    """Get the path to the _RepoMap index file for a repository."""
    return wiki_dir / f"_RepoMap_{repo_name}.md"


def get_files_dir_path(wiki_dir: Path) -> Path:
    """Get the path to the _files directory containing per-file details."""
    return wiki_dir / "_files"


def ensure_reports_directory(wiki_dir: Path) -> Path:
    """
    Create the _reports directory for phase execution summaries.
    
    Each phase writes a report that the next phase can read for context.
    """
    reports_dir = wiki_dir / "_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def generate_page_indexes(wiki_dir: Path, repo_name: str) -> None:
    """
    Generate empty index files for each page type.
    
    Creates:
    - _WorkflowIndex.md
    - _PrincipleIndex.md
    - _ImplementationIndex.md
    - _EnvironmentIndex.md
    - _HeuristicIndex.md
    
    Each index tracks pages of that type with cross-references.
    """
    indexes = {
        "Workflow": {
            "file": "_WorkflowIndex.md",
            "columns": "| Page | File | Steps (Principles) | Source Files | Notes |",
            "separator": "|------|------|-------------------|--------------|-------|",
            "desc": "Tracks all Workflow pages and their step connections to Principles.",
        },
        "Principle": {
            "file": "_PrincipleIndex.md",
            "columns": "| Page | File | Implemented By | In Workflows | Notes |",
            "separator": "|------|------|----------------|--------------|-------|",
            "desc": "Tracks all Principle pages and their implementation/workflow connections.",
        },
        "Implementation": {
            "file": "_ImplementationIndex.md",
            "columns": "| Page | File | Source | Implements (Principle) | Notes |",
            "separator": "|------|------|--------|------------------------|-------|",
            "desc": "Tracks all Implementation pages and their source code locations.",
        },
        "Environment": {
            "file": "_EnvironmentIndex.md",
            "columns": "| Page | File | Required By | Notes |",
            "separator": "|------|------|-------------|-------|",
            "desc": "Tracks all Environment pages and which implementations require them.",
        },
        "Heuristic": {
            "file": "_HeuristicIndex.md",
            "columns": "| Page | File | Applies To | Notes |",
            "separator": "|------|------|------------|-------|",
            "desc": "Tracks all Heuristic pages and which pages they apply to.",
        },
    }
    
    for page_type, config in indexes.items():
        content = f"""# {page_type} Index: {repo_name}

> {config['desc']}
> Update this file after creating {page_type} pages.

{config['columns']}
{config['separator']}
<!-- Add rows as you create pages -->

---

## Integrity Notes

<!-- Record any cross-reference issues or TODOs here -->
"""
        index_path = wiki_dir / config["file"]
        index_path.write_text(content, encoding="utf-8")
    
    logger.info(f"Generated 5 page index files in {wiki_dir}")


def check_exploration_progress(repo_map_path: Path) -> Tuple[int, int, List[str]]:
    """
    Check how many files have been explored vs total.
    
    Parses the _RepoMap index file and counts ✅ vs ⬜ markers.
    
    Args:
        repo_map_path: Path to the _RepoMap_{repo_name}.md file
        
    Returns:
        Tuple of (explored_count, total_count, list_of_unexplored_file_paths)
    """
    if not repo_map_path.exists():
        logger.warning(f"Repo map not found: {repo_map_path}")
        return (0, 0, [])
    
    content = repo_map_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    explored = 0
    total = 0
    unexplored = []
    
    for line in lines:
        # Look for table rows with status markers
        # Format: | ✅ | `path/to/file.py` | ... or | ⬜ | `path/to/file.py` | ...
        if line.startswith("|") and ("`" in line):
            # Skip header rows
            if "Status" in line or "----" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                status = parts[1]
                file_cell = parts[2]
                
                # Extract file path from backticks
                if "`" in file_cell:
                    # Extract path between backticks
                    start = file_cell.find("`") + 1
                    end = file_cell.rfind("`")
                    if start > 0 and end > start:
                        file_path = file_cell[start:end]
                        total += 1
                        
                        if "✅" in status:
                            explored += 1
                        else:
                            unexplored.append(file_path)
    
    logger.info(f"Exploration progress: {explored}/{total} files explored")
    return (explored, total, unexplored)

