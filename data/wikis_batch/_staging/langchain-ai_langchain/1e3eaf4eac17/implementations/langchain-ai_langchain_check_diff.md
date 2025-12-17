{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::CI/CD]], [[domain::Testing]], [[domain::DevOps]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
CI script that analyzes git diffs to determine which LangChain packages require testing, linting, and building based on file changes.

=== Description ===
The check_diff.py script is a critical component of the LangChain monorepo's CI/CD workflow. It intelligently maps changed files to affected package directories, builds dependency graphs to include dependent packages when core components change, and generates test matrix configurations with appropriate Python versions. This script ensures efficient CI pipeline execution by testing only the packages that are affected by code changes, while respecting package dependencies and special testing requirements.

The script handles the monorepo's complex structure with multiple independently versioned packages (libs/core, libs/partners/*, etc.), understands the dependency relationships between packages, and can configure different test types (standard tests, extended tests, Pydantic version tests, and performance benchmarks via codspeed).

=== Usage ===
This script is automatically invoked by the check_diffs GitHub Actions workflow when pull requests are created or updated. It receives a list of changed files as command-line arguments and outputs JSON configurations for various CI jobs (lint, test, extended-tests, compile-integration-tests, dependencies, test-pydantic, codspeed).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/.github/scripts/check_diff.py .github/scripts/check_diff.py]
* '''Lines:''' 1-341

=== Signature ===
<syntaxhighlight lang="python">
# Main functions
def all_package_dirs() -> Set[str]
def dependents_graph() -> dict
def add_dependents(dirs_to_eval: Set[str], dependents: dict) -> List[str]
def _get_configs_for_single_dir(job: str, dir_: str) -> List[Dict[str, str]]
def _get_pydantic_test_configs(dir_: str, *, python_version: str = "3.12") -> List[Dict[str, str]]
def _get_configs_for_multi_dirs(job: str, dirs_to_run: Dict[str, Set[str]], dependents: dict) -> List[Dict[str, str]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This script is not imported as a module; it runs as a standalone script
# Invoked by: python .github/scripts/check_diff.py <file1> <file2> ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| files || List[str] || Yes || Command-line arguments containing paths to changed files from git diff
|-
| IGNORE_CORE_DEPENDENTS || bool (global) || No || Flag to control whether core dependents are included when core changes
|-
| IGNORED_PARTNERS || List[str] (global) || No || List of partner packages to exclude from dependent testing
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| stdout || str || JSON configurations for CI jobs in format "job_name={json_array}"
|}

Output format for each job:
<syntaxhighlight lang="text">
lint=[{"working-directory": "libs/core", "python-version": "3.10"}, ...]
test=[{"working-directory": "libs/partners/openai", "python-version": "3.10"}, ...]
extended-tests=[...]
compile-integration-tests=[...]
dependencies=[...]
test-pydantic=[{"working-directory": "libs/core", "pydantic-version": "2.8.0", "python-version": "3.12"}, ...]
codspeed=[{"working-directory": "libs/core", "python-version": "3.13"}, ...]
</syntaxhighlight>

== Core Functions ==

=== all_package_dirs ===
Returns all package directories in the monorepo by finding pyproject.toml files, excluding CLI and standard-tests packages.

=== dependents_graph ===
Constructs a mapping of package names to their dependent packages by parsing pyproject.toml dependencies and extended_testing_deps.txt files. This ensures that when a package changes, all packages that depend on it are also tested.

=== add_dependents ===
Given a set of directories to evaluate, adds all dependent packages from the dependents graph. Special handling for libs/core which has many dependents and can be controlled by IGNORE_CORE_DEPENDENTS flag.

=== _get_configs_for_single_dir ===
Generates test matrix configurations for a single directory, determining appropriate Python versions based on the directory and job type:
* codspeed: Python 3.13 only
* libs/core: Python 3.10, 3.11, 3.12, 3.13, 3.14
* libs/partners/chroma: Python 3.10, 3.13
* default: Python 3.10, 3.14

=== _get_pydantic_test_configs ===
Generates Pydantic version test matrix by determining the compatible Pydantic version range from uv.lock files and pyproject.toml constraints. Tests against all minor versions of Pydantic 2.x that are compatible with both the package and core.

=== _get_configs_for_multi_dirs ===
Aggregates configurations for multiple directories based on job type, applying dependency resolution rules.

== Directory Mapping Rules ==

=== Infrastructure Changes ===
Changes to .github/workflows, .github/tools, .github/actions, or check_diff.py itself trigger extended tests on all core packages (libs/core, libs/text-splitters, libs/langchain, libs/langchain_v1) as a safety measure.

=== Core Packages ===
Changes to LANGCHAIN_DIRS (core, text-splitters, langchain, langchain_v1, model-profiles) trigger extended tests on that directory and all subsequent directories in the list, implementing a dependency cascade.

=== Partner Packages ===
Changes to libs/partners/* trigger tests on the specific partner package and optionally add it to codspeed benchmarks (unless in IGNORED_PARTNERS).

=== Standard Tests ===
Changes to libs/standard-tests trigger tests on standard-tests itself and key partner packages (mistralai, openai, anthropic, fireworks, groq).

=== CLI ===
Changes to libs/cli trigger lint and test jobs for the CLI package only.

== Special Handling ==

=== Large Diffs ===
If 300 or more files are changed (max diff length), all package directories are added to lint, test, and extended-test jobs to ensure comprehensive testing.

=== Ignored Partners ===
Partners in IGNORED_PARTNERS (huggingface, prompty) are removed from dependents lists but still run if directly edited. This handles CI instability issues specific to those packages.

=== Tombstone Directories ===
Skips partner directories that only contain README.md files (deleted packages with tombstone documentation).

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
# Invoked by GitHub Actions workflow
# Example: Changes to core package
python .github/scripts/check_diff.py libs/core/langchain_core/runnables.py

# Output:
# lint=[{"working-directory": "libs/core", "python-version": "3.10"}, ...]
# test=[{"working-directory": "libs/core", "python-version": "3.10"}, ...]
# extended-tests=[{"working-directory": "libs/core", "python-version": "3.10"}, ...]
# compile-integration-tests=[...]
# dependencies=[...]
# test-pydantic=[{"working-directory": "libs/core", "pydantic-version": "2.8.0", "python-version": "3.12"}, ...]
# codspeed=[{"working-directory": "libs/core", "python-version": "3.13"}]
</syntaxhighlight>

=== Partner Package Change ===
<syntaxhighlight lang="python">
# Changes to OpenAI partner package
python .github/scripts/check_diff.py libs/partners/openai/langchain_openai/chat_models.py

# Output includes test matrix for openai package and its dependents
# test=[{"working-directory": "libs/partners/openai", "python-version": "3.10"},
#       {"working-directory": "libs/partners/openai", "python-version": "3.14"}]
# codspeed=[{"working-directory": "libs/partners/openai", "python-version": "3.13"}]
</syntaxhighlight>

=== Workflow Integration ===
<syntaxhighlight lang="yaml">
# In .github/workflows/check_diffs.yml
- name: Get changed files
  id: changed-files
  run: |
    git diff --name-only origin/master...HEAD > changed_files.txt

- name: Determine test matrix
  id: matrix
  run: |
    python .github/scripts/check_diff.py $(cat changed_files.txt)
</syntaxhighlight>

== Related Pages ==
* [[uses::Concept:Dependency_Graph]]
* [[uses::Concept:CI_Test_Matrix]]
