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
    Generate index files for each page type.
    
    Creates:
    - _WorkflowIndex.md (detailed template with per-step structure)
    - _PrincipleIndex.md
    - _ImplementationIndex.md
    - _EnvironmentIndex.md
    - _HeuristicIndex.md
    
    Each index tracks pages of that type with cross-references.
    The WorkflowIndex uses a comprehensive structure that Phase 2 depends on.
    """
    # Generate the detailed WorkflowIndex template first
    # This structure is CRITICAL for Phase 2 (Excavation+Synthesis) to work correctly
    workflow_index_content = f"""# Workflow Index: {repo_name}

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Implementation APIs |
|----------|-------|------------|---------------------|
<!-- ADD ROW FOR EACH WORKFLOW: | WorkflowName | N | N | API1, API2, ... | -->

---

<!-- 
================================================================================
FOR EACH WORKFLOW, ADD A SECTION LIKE THIS:
================================================================================

## Workflow: {repo_name}_WorkflowName

**File:** [→](./workflows/{repo_name}_WorkflowName.md)
**Description:** One-line description of the workflow.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Step Name | Principle_Name | `API.method()` | ⬜ |
| 2 | Step Name | Principle_Name | `API.method()` | ⬜ |

### Step 1: Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `{repo_name}_Model_Loading` |
| **Implementation** | `{repo_name}_FastLanguageModel_From_Pretrained` |
| **API Call** | `FastLanguageModel.from_pretrained(model_name, ...)` |
| **Source Location** | `path/to/file.py:L100-200` |
| **External Dependencies** | `transformers`, `torch` |
| **Environment** | `{repo_name}_CUDA_11_Requirements` |
| **Key Parameters** | `model_name: str`, `max_seq_length: int` |
| **Inputs** | Model name or path |
| **Outputs** | Loaded model and tokenizer |

(Repeat Step N: ... for each step)

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Model_Loading | `FastLanguageModel_From_Pretrained` | `from_pretrained` | `loader.py` | API Doc |
| LoRA_Configuration | `Get_Peft_Model_Wrapper` | `peft.get_peft_model` | External | Wrapper Doc |

================================================================================
-->

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
"""
    workflow_index_path = wiki_dir / "_WorkflowIndex.md"
    workflow_index_path.write_text(workflow_index_content, encoding="utf-8")
    
    # Other indexes use simpler structure
    # NOTE: Example page names follow WikiMedia naming conventions:
    # - First letter capitalized
    # - Underscores only (no hyphens)
    # - Case-sensitive after first character
    other_indexes = {
        "Principle": {
            "file": "_PrincipleIndex.md",
            "folder": "principles",
            "desc": "Tracks Principle pages and their connections to Implementations, Workflows, etc.",
            "connection_hint": f"e.g., `✅Impl:{repo_name}_FastLanguageModel_From_Pretrained, ✅Workflow:{repo_name}_QLoRA_Finetuning`",
        },
        "Implementation": {
            "file": "_ImplementationIndex.md",
            "folder": "implementations",
            "desc": "Tracks Implementation pages and their connections to Principles, Environments, etc.",
            "connection_hint": f"e.g., `⬜Principle:{repo_name}_LoRA_Configuration, ⬜Env:{repo_name}_CUDA_11_Requirements`",
        },
        "Environment": {
            "file": "_EnvironmentIndex.md",
            "folder": "environments",
            "desc": "Tracks Environment pages and which pages require them.",
            "connection_hint": f"e.g., `✅Impl:{repo_name}_FastLanguageModel_From_Pretrained`",
        },
        "Heuristic": {
            "file": "_HeuristicIndex.md",
            "folder": "heuristics",
            "desc": "Tracks Heuristic pages and which pages they apply to.",
            "connection_hint": f"e.g., `✅Impl:{repo_name}_SFTTrainer_Train, ✅Workflow:{repo_name}_QLoRA_Finetuning`",
        },
    }
    
    for page_type, config in other_indexes.items():
        content = f"""# {page_type} Index: {repo_name}

> {config['desc']}
> **Update IMMEDIATELY** after creating or modifying a {page_type} page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
<!-- {config['connection_hint']} -->

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
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


# =============================================================================
# ORPHAN MINING: Deterministic Triage and Verification
# =============================================================================


def _parse_repo_map_for_orphans(repo_map_path: Path) -> List[Dict]:
    """
    Parse the RepoMap to find files with Coverage = '—' (orphan candidates).
    
    Returns list of dicts with: path, lines, purpose, coverage
    """
    if not repo_map_path.exists():
        logger.warning(f"Repo map not found: {repo_map_path}")
        return []
    
    content = repo_map_path.read_text(encoding="utf-8")
    lines_list = content.splitlines()
    
    orphan_files = []
    
    for line in lines_list:
        # Look for table rows: | Status | `path` | lines | purpose | coverage | details |
        if not line.startswith("|") or "`" not in line:
            continue
        
        # Skip header rows
        if "Status" in line or "----" in line or "File" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        # Expected: ['', 'Status', '`path`', 'lines', 'purpose', 'coverage', 'details', '']
        if len(parts) < 7:
            continue
        
        file_cell = parts[2]
        lines_cell = parts[3]
        purpose_cell = parts[4]
        coverage_cell = parts[5]
        
        # Extract file path from backticks
        if "`" not in file_cell:
            continue
        
        start = file_cell.find("`") + 1
        end = file_cell.rfind("`")
        if start <= 0 or end <= start:
            continue
        
        file_path = file_cell[start:end]
        
        # Parse line count
        try:
            line_count = int(lines_cell.replace(",", ""))
        except ValueError:
            line_count = 0
        
        # Check if coverage is empty (orphan candidate)
        # Coverage column shows "—" for uncovered files
        is_orphan = coverage_cell.strip() in ("—", "-", "")
        
        if is_orphan:
            orphan_files.append({
                "path": file_path,
                "lines": line_count,
                "purpose": purpose_cell if purpose_cell != "—" else "",
            })
    
    return orphan_files


def _apply_orphan_filter_rules(file_info: Dict) -> Tuple[str, str]:
    """
    Apply deterministic filter rules to classify an orphan file.
    
    Returns:
        Tuple of (category, rule) where:
        - category: "AUTO_DISCARD", "AUTO_KEEP", or "MANUAL_REVIEW"
        - rule: The rule that matched (e.g., "D1", "K1")
    """
    path = file_info["path"]
    lines = file_info["lines"]
    filename = path.split("/")[-1] if "/" in path else path
    
    # =========================================================================
    # AUTO_DISCARD RULES (no agent judgment needed)
    # =========================================================================
    
    # Rule D1: Empty or near-empty files (≤20 lines)
    if lines <= 20:
        return ("AUTO_DISCARD", "D1: ≤20 lines")
    
    # Rule D2: Small __init__.py files (<100 lines)
    if filename == "__init__.py" and lines < 100:
        return ("AUTO_DISCARD", "D2: Small __init__.py")
    
    # Rule D3: Test files (by path pattern)
    if "/tests/" in path or "/test_" in path or "_test.py" in path or path.startswith("tests/"):
        return ("AUTO_DISCARD", "D3: Test file")
    
    # Rule D4: Benchmark files
    if "/benchmark/" in path or "/benchmarks/" in path:
        return ("AUTO_DISCARD", "D4: Benchmark file")
    
    # Rule D5: Scripts directory (not core library)
    if path.startswith("scripts/"):
        return ("AUTO_DISCARD", "D5: Scripts directory")
    
    # =========================================================================
    # AUTO_KEEP RULES (no agent judgment needed)
    # =========================================================================
    
    # Rule K1: Large files (≥300 lines) - likely substantial code
    if lines >= 300:
        return ("AUTO_KEEP", f"K1: {lines} lines (≥300)")
    
    # Rule K2: Kernel files (performance-critical code)
    if "/kernels/" in path and lines >= 100:
        return ("AUTO_KEEP", f"K2: Kernel file ({lines} lines)")
    
    # Rule K3: Model files (user-facing implementations)
    if "/models/" in path and lines >= 200:
        return ("AUTO_KEEP", f"K3: Model file ({lines} lines)")
    
    # =========================================================================
    # MANUAL_REVIEW: Everything else (agent judgment needed)
    # =========================================================================
    return ("MANUAL_REVIEW", "")


def generate_orphan_candidates(
    repo_map_path: Path,
    wiki_dir: Path,
    repo_name: str,
) -> Path:
    """
    Generate _orphan_candidates.md with deterministic classification.
    
    This is Step 6a of orphan mining. It reads the RepoMap, applies
    deterministic filter rules, and writes a candidates file that
    the agent will use in subsequent steps.
    
    Categories:
    - AUTO_KEEP: Files that MUST be documented (no agent judgment)
    - AUTO_DISCARD: Files to skip (no agent judgment)
    - MANUAL_REVIEW: Files requiring agent evaluation
    
    Args:
        repo_map_path: Path to _RepoMap_{repo_name}.md
        wiki_dir: Directory to write _orphan_candidates.md
        repo_name: Repository name for display
        
    Returns:
        Path to the generated _orphan_candidates.md file
    """
    logger.info(f"Generating orphan candidates from {repo_map_path}")
    
    # Parse RepoMap to find orphan files (Coverage = "—")
    orphan_files = _parse_repo_map_for_orphans(repo_map_path)
    logger.info(f"Found {len(orphan_files)} orphan candidates")
    
    # Classify each file using deterministic rules
    auto_keep = []
    auto_discard = []
    manual_review = []
    
    for file_info in orphan_files:
        category, rule = _apply_orphan_filter_rules(file_info)
        
        if category == "AUTO_DISCARD":
            auto_discard.append((file_info["path"], file_info["lines"], rule))
        elif category == "AUTO_KEEP":
            auto_keep.append((file_info["path"], file_info["lines"], rule))
        else:
            manual_review.append((
                file_info["path"],
                file_info["lines"],
                file_info["purpose"],
            ))
    
    # Sort each list by path for consistent output
    auto_keep.sort(key=lambda x: x[0])
    auto_discard.sort(key=lambda x: x[0])
    manual_review.sort(key=lambda x: x[0])
    
    # Generate markdown content
    content_lines = []
    
    # Header
    content_lines.append(f"# Orphan Candidates: {repo_name}")
    content_lines.append("")
    content_lines.append("> Generated by deterministic triage (Step 6a).")
    content_lines.append("> Agent reviews MANUAL_REVIEW section only.")
    content_lines.append("")
    
    # Summary table
    content_lines.append("## Summary")
    content_lines.append("")
    content_lines.append("| Category | Count | Action |")
    content_lines.append("|----------|-------|--------|")
    content_lines.append(f"| AUTO_KEEP | {len(auto_keep)} | Create pages (no judgment needed) |")
    content_lines.append(f"| AUTO_DISCARD | {len(auto_discard)} | Skip (no judgment needed) |")
    content_lines.append(f"| MANUAL_REVIEW | {len(manual_review)} | Agent evaluates each file |")
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    
    # AUTO_KEEP section
    content_lines.append("## AUTO_KEEP (Must Document)")
    content_lines.append("")
    content_lines.append("These files MUST have wiki pages. No agent judgment required.")
    content_lines.append("")
    content_lines.append("| # | File | Lines | Rule | Status |")
    content_lines.append("|---|------|-------|------|--------|")
    for i, (path, lines, rule) in enumerate(auto_keep, 1):
        content_lines.append(f"| {i} | `{path}` | {lines} | {rule} | ⬜ PENDING |")
    if not auto_keep:
        content_lines.append("| — | (none) | — | — | — |")
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    
    # AUTO_DISCARD section
    content_lines.append("## AUTO_DISCARD (Skip)")
    content_lines.append("")
    content_lines.append("These files are skipped. No wiki pages needed.")
    content_lines.append("")
    content_lines.append("| File | Lines | Rule |")
    content_lines.append("|------|-------|------|")
    for path, lines, rule in auto_discard:
        content_lines.append(f"| `{path}` | {lines} | {rule} |")
    if not auto_discard:
        content_lines.append("| (none) | — | — |")
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    
    # MANUAL_REVIEW section
    content_lines.append("## MANUAL_REVIEW (Agent Evaluates)")
    content_lines.append("")
    content_lines.append("Agent must evaluate each file and write decision.")
    content_lines.append("")
    content_lines.append("| # | File | Lines | Purpose | Decision | Reasoning |")
    content_lines.append("|---|------|-------|---------|----------|-----------|")
    for i, (path, lines, purpose) in enumerate(manual_review, 1):
        purpose_display = purpose if purpose else "—"
        content_lines.append(f"| {i} | `{path}` | {lines} | {purpose_display} | ⬜ PENDING | |")
    if not manual_review:
        content_lines.append("| — | (none) | — | — | — | — |")
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    
    # Decision guide for agent
    content_lines.append("## Decision Guide for Agent")
    content_lines.append("")
    content_lines.append("For MANUAL_REVIEW files, evaluate:")
    content_lines.append("")
    content_lines.append("1. **Does it have a public API?** (class or function without `_` prefix)")
    content_lines.append("2. **Is it user-facing?** (would a user import/call this?)")
    content_lines.append("3. **Does it implement a distinct algorithm?** (not just glue code)")
    content_lines.append("")
    content_lines.append("Write decision as:")
    content_lines.append("- `✅ APPROVED` — Create wiki page")
    content_lines.append("- `❌ REJECTED` — Skip (with reasoning)")
    content_lines.append("")
    
    # Write to file
    output_path = wiki_dir / "_orphan_candidates.md"
    output_path.write_text("\n".join(content_lines), encoding="utf-8")
    
    logger.info(
        f"Wrote orphan candidates: {len(auto_keep)} AUTO_KEEP, "
        f"{len(auto_discard)} AUTO_DISCARD, {len(manual_review)} MANUAL_REVIEW"
    )
    
    return output_path


def get_orphan_candidates_path(wiki_dir: Path) -> Path:
    """Get the path to the _orphan_candidates.md file."""
    return wiki_dir / "_orphan_candidates.md"


def _parse_orphan_candidates(candidates_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse _orphan_candidates.md to get file lists and statuses.
    
    Returns dict with:
    - auto_keep: List of {path, status} dicts
    - manual_review: List of {path, decision} dicts
    """
    if not candidates_path.exists():
        return {"auto_keep": [], "manual_review": []}
    
    content = candidates_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    auto_keep = []
    manual_review = []
    current_section = None
    
    for line in lines:
        # Detect section headers
        if "## AUTO_KEEP" in line:
            current_section = "auto_keep"
            continue
        elif "## AUTO_DISCARD" in line:
            current_section = "auto_discard"
            continue
        elif "## MANUAL_REVIEW" in line:
            current_section = "manual_review"
            continue
        elif line.startswith("## "):
            current_section = None
            continue
        
        # Skip non-table rows
        if not line.startswith("|") or "`" not in line:
            continue
        if "----" in line or "File" in line or "Status" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        
        if current_section == "auto_keep" and len(parts) >= 6:
            # | # | File | Lines | Rule | Status |
            file_cell = parts[2]
            status_cell = parts[5]
            
            # Extract path from backticks
            if "`" in file_cell:
                start = file_cell.find("`") + 1
                end = file_cell.rfind("`")
                if start > 0 and end > start:
                    path = file_cell[start:end]
                    is_done = "DONE" in status_cell or "✅" in status_cell
                    auto_keep.append({"path": path, "done": is_done})
        
        elif current_section == "manual_review" and len(parts) >= 7:
            # | # | File | Lines | Purpose | Decision | Reasoning |
            file_cell = parts[2]
            decision_cell = parts[5]
            
            # Extract path from backticks
            if "`" in file_cell:
                start = file_cell.find("`") + 1
                end = file_cell.rfind("`")
                if start > 0 and end > start:
                    path = file_cell[start:end]
                    # Check decision status
                    is_pending = "PENDING" in decision_cell or "⬜" in decision_cell
                    is_approved = "APPROVED" in decision_cell or "✅" in decision_cell
                    is_rejected = "REJECTED" in decision_cell or "❌" in decision_cell
                    
                    manual_review.append({
                        "path": path,
                        "pending": is_pending,
                        "approved": is_approved,
                        "rejected": is_rejected,
                    })
    
    return {"auto_keep": auto_keep, "manual_review": manual_review}


def verify_orphan_completion(wiki_dir: Path, repo_name: str) -> Tuple[bool, str]:
    """
    Verify all orphan candidates were processed correctly.
    
    This is Step 6d of orphan mining. It checks:
    1. All AUTO_KEEP files have DONE status
    2. All MANUAL_REVIEW files have decisions (not PENDING)
    3. All approved files have wiki pages in implementations/
    
    Args:
        wiki_dir: Wiki directory containing _orphan_candidates.md
        repo_name: Repository name for page name formatting
        
    Returns:
        Tuple of (success: bool, report: str)
        - success: True if all checks pass
        - report: Human-readable report of findings
    """
    candidates_path = get_orphan_candidates_path(wiki_dir)
    
    if not candidates_path.exists():
        return (False, "ERROR: _orphan_candidates.md not found")
    
    # Parse the candidates file
    parsed = _parse_orphan_candidates(candidates_path)
    auto_keep = parsed["auto_keep"]
    manual_review = parsed["manual_review"]
    
    errors = []
    warnings = []
    
    # Check 1: All AUTO_KEEP files have DONE status
    auto_keep_not_done = [f["path"] for f in auto_keep if not f["done"]]
    if auto_keep_not_done:
        for path in auto_keep_not_done:
            errors.append(f"AUTO_KEEP not completed: {path}")
    
    # Check 2: All MANUAL_REVIEW files have decisions
    manual_pending = [f["path"] for f in manual_review if f["pending"]]
    if manual_pending:
        for path in manual_pending:
            errors.append(f"MANUAL_REVIEW missing decision: {path}")
    
    # Check 3: All approved files have wiki pages
    # Approved = AUTO_KEEP (all) + MANUAL_REVIEW with APPROVED
    approved_files = [f["path"] for f in auto_keep]
    approved_files.extend([f["path"] for f in manual_review if f["approved"]])
    
    implementations_dir = wiki_dir / "implementations"
    principles_dir = wiki_dir / "principles"
    
    for file_path in approved_files:
        # Derive expected page name from file path
        # e.g., "unsloth/kernels/geglu.py" -> possible page names:
        # - {repo_name}_geglu_kernel
        # - {repo_name}_geglu
        # - etc.
        # We check if ANY page exists that could correspond to this file
        
        filename = file_path.split("/")[-1].replace(".py", "")
        
        # Check if any implementation page might cover this file
        # Look for pages containing the filename (case-insensitive)
        found_page = False
        
        if implementations_dir.exists():
            for page_file in implementations_dir.glob("*.md"):
                page_name = page_file.stem.lower()
                if filename.lower() in page_name:
                    found_page = True
                    break
        
        if not found_page and principles_dir.exists():
            for page_file in principles_dir.glob("*.md"):
                page_name = page_file.stem.lower()
                if filename.lower() in page_name:
                    found_page = True
                    break
        
        if not found_page:
            warnings.append(f"No wiki page found for approved file: {file_path}")
    
    # Build report
    report_lines = []
    report_lines.append("# Orphan Completion Verification Report")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- AUTO_KEEP files: {len(auto_keep)}")
    report_lines.append(f"- MANUAL_REVIEW files: {len(manual_review)}")
    report_lines.append(f"- Errors: {len(errors)}")
    report_lines.append(f"- Warnings: {len(warnings)}")
    report_lines.append("")
    
    if errors:
        report_lines.append("## Errors (Must Fix)")
        report_lines.append("")
        for error in errors:
            report_lines.append(f"- {error}")
        report_lines.append("")
    
    if warnings:
        report_lines.append("## Warnings")
        report_lines.append("")
        for warning in warnings:
            report_lines.append(f"- {warning}")
        report_lines.append("")
    
    if not errors and not warnings:
        report_lines.append("## Result: PASS")
        report_lines.append("")
        report_lines.append("All orphan candidates processed successfully.")
    elif not errors:
        report_lines.append("## Result: PASS (with warnings)")
        report_lines.append("")
        report_lines.append("All required steps completed. Review warnings above.")
    else:
        report_lines.append("## Result: FAIL")
        report_lines.append("")
        report_lines.append("Fix the errors above and re-run verification.")
    
    report = "\n".join(report_lines)
    success = len(errors) == 0
    
    logger.info(f"Orphan verification: {'PASS' if success else 'FAIL'} ({len(errors)} errors, {len(warnings)} warnings)")
    
    return (success, report)

