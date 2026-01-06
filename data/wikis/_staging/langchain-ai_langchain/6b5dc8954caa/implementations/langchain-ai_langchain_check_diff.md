{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Testing]], [[domain::Build_System]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

CI script that analyzes git diffs to determine which LangChain packages need testing, linting, or building based on changed files.

=== Description ===

The `check_diff.py` script is a CI optimization tool that intelligently determines which packages in the LangChain monorepo need to be tested when changes are made. It builds a dependency graph between packages, maps file changes to affected directories, and generates test matrix configurations for GitHub Actions workflows. The script handles special cases like Pydantic version testing and performance benchmarks.

=== Usage ===

Use this script in CI pipelines to selectively run tests only on packages affected by code changes, reducing CI time and resource usage. It is invoked by the `check_diffs` GitHub Actions workflow.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/.github/scripts/check_diff.py .github/scripts/check_diff.py]
* '''Lines:''' 1-340

=== Signature ===
<syntaxhighlight lang="python">
def all_package_dirs() -> Set[str]:
    """Return all package directories in the monorepo."""

def dependents_graph() -> dict:
    """Construct a mapping of package -> dependents for test propagation."""

def add_dependents(dirs_to_eval: Set[str], dependents: dict) -> List[str]:
    """Add dependent packages to the evaluation set."""

def _get_configs_for_single_dir(job: str, dir_: str) -> List[Dict[str, str]]:
    """Generate CI configs for a single directory."""

def _get_pydantic_test_configs(dir_: str, *, python_version: str = "3.12") -> List[Dict[str, str]]:
    """Generate Pydantic version test matrix configs."""

def _get_configs_for_multi_dirs(job: str, dirs_to_run: Dict[str, Set[str]], dependents: dict) -> List[Dict[str, str]]:
    """Generate CI configs for multiple directories."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal CI script - not importable as a module
# Used via: python .github/scripts/check_diff.py <file1> <file2> ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| files || List[str] || Yes || List of changed file paths from git diff
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| stdout || str || JSON configurations for each CI job type (lint, test, extended-tests, etc.)
|}

== Usage Examples ==

=== Basic Usage in CI ===
<syntaxhighlight lang="bash">
# Get list of changed files and pass to script
git diff --name-only origin/main...HEAD | xargs python .github/scripts/check_diff.py
</syntaxhighlight>

=== Output Format ===
<syntaxhighlight lang="json">
lint=[{"working-directory": "libs/core", "python-version": "3.10"}]
test=[{"working-directory": "libs/core", "python-version": "3.10"}, {"working-directory": "libs/core", "python-version": "3.14"}]
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

