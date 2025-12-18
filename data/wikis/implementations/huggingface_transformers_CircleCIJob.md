{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Build_Tools]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

A dynamic CircleCI configuration generator that creates YAML config files for running parallel test suites across multiple docker images and resource classes.

=== Description ===

The `create_circleci_config.py` module provides a flexible framework for generating CircleCI configuration files dynamically based on test files discovered in the `test_preparation` directory. The main class `CircleCIJob` encapsulates all configuration needed for a test job including docker images, environment variables, pytest options, parallelism settings, and test execution steps.

The module defines multiple pre-configured test jobs (torch, generate, tokenization, processors, pipelines, hub, examples, etc.) that can be composed into comprehensive CI workflows. It handles complex scenarios like flaky test retries, test splitting across parallel nodes, hub cache management, and result aggregation.

Key features include:
* **Dataclass-based job configuration** with sensible defaults
* **Automatic test file discovery** from `test_preparation/{job_name}_test_list.txt`
* **Parallel test execution** with CircleCI's test splitting
* **Flaky test handling** using pytest-rerunfailures with pattern matching
* **Docker image customization** per job type
* **Environment variable management** including conditional HF tokens
* **Result collection and reporting** with artifact storage

The generated YAML config includes steps for checkout, dependency installation, test execution with retries, crash detection, and comprehensive logging/artifact storage.

=== Usage ===

Use this module when you need to dynamically generate CircleCI configuration based on discovered test files, particularly in large repositories with multiple test suites that need to run in parallel across different environments. It's designed for the Transformers repository's CI/CD pipeline but demonstrates patterns applicable to any complex testing setup.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/.circleci/create_circleci_config.py .circleci/create_circleci_config.py]
* '''Lines:''' 1-413

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class CircleCIJob:
    name: str
    additional_env: dict[str, Any] = None
    docker_image: list[dict[str, str]] = None
    install_steps: list[str] = None
    marker: Optional[str] = None
    parallelism: Optional[int] = 0
    pytest_num_workers: int = 8
    pytest_options: dict[str, Any] = None
    resource_class: Optional[str] = "xlarge"
    tests_to_run: Optional[list[str]] = None
    num_test_files_per_worker: Optional[int] = 10
    command_timeout: Optional[int] = None

    def to_dict(self) -> dict

    @property
    def job_name(self) -> str


class EmptyJob:
    job_name = "empty"

    def to_dict(self) -> dict


def create_circleci_config(folder: Optional[str] = None) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from create_circleci_config import CircleCIJob, EmptyJob, create_circleci_config
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| name || str || Yes || Job name identifier
|-
| additional_env || dict[str, Any] || No || Additional environment variables to add to the job
|-
| docker_image || list[dict[str, str]] || No || Docker image configuration (defaults to cimg/python:3.8.12)
|-
| install_steps || list[str] || No || Custom installation commands (defaults to ["uv pip install ."])
|-
| marker || str || No || Pytest marker to filter tests (e.g., "is_pipeline_test")
|-
| parallelism || int || No || Number of parallel CircleCI nodes (0 = no parallelism)
|-
| pytest_num_workers || int || No || Number of pytest workers (-n flag), defaults to 8
|-
| pytest_options || dict[str, Any] || No || Additional pytest command-line options
|-
| resource_class || str || No || CircleCI resource class (small/medium/large/xlarge)
|-
| tests_to_run || list[str] || No || Specific test files to run (auto-discovered if not provided)
|-
| command_timeout || int || No || Timeout in seconds for test execution
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| CircleCI config || YAML file || Generated CircleCI configuration at {folder}/generated_config.yml
|-
| job dictionary || dict || Job configuration dictionary from to_dict() method
|-
| job_name || str || Formatted job name with "tests_" prefix where appropriate
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Create a basic test job
from create_circleci_config import CircleCIJob

torch_job = CircleCIJob(
    "torch",
    docker_image=[{"image": "huggingface/transformers-torch-light"}],
    marker="not generate",
    parallelism=6,
)

# Convert to dictionary for CircleCI config
job_config = torch_job.to_dict()
print(job_config['docker'])  # [{'image': 'huggingface/transformers-torch-light'}]
print(job_config['resource_class'])  # 'xlarge'


# Example 2: Create a job with custom environment and installation
hub_job = CircleCIJob(
    "hub",
    additional_env={"HUGGINGFACE_CO_STAGING": True},
    docker_image=[{"image": "huggingface/transformers-torch-light"}],
    install_steps=[
        'uv pip install .',
        'git config --global user.email "ci@dummy.com"',
        'git config --global user.name "ci"',
    ],
    marker="is_staging_test",
    pytest_num_workers=2,
    resource_class="medium",
)


# Example 3: Generate complete CircleCI configuration
import os
from create_circleci_config import create_circleci_config

# Assumes test_preparation directory exists with test list files
# e.g., test_preparation/tests_torch_test_list.txt
os.makedirs("test_preparation", exist_ok=True)
with open("test_preparation/tests_torch_test_list.txt", "w") as f:
    f.write("tests/test_modeling_bert.py\n")
    f.write("tests/test_modeling_gpt2.py\n")

create_circleci_config(folder=".")
# Creates ./generated_config.yml with full CircleCI workflow


# Example 4: Create a custom documentation test job
doc_test_job = CircleCIJob(
    "pr_documentation_tests",
    docker_image=[{"image": "huggingface/transformers-consistency"}],
    additional_env={
        "TRANSFORMERS_VERBOSITY": "error",
        "DATASETS_VERBOSITY": "error",
        "SKIP_CUDA_DOCTEST": "1"
    },
    pytest_options={
        "-doctest-modules": None,
        "doctest-glob": "*.md",
        "dist": "loadfile",
    },
    command_timeout=1200,
    pytest_num_workers=1,
)


# Example 5: Inspect flaky test patterns
from create_circleci_config import FLAKY_TEST_FAILURE_PATTERNS

print("Tests matching these patterns will be retried:")
for pattern in FLAKY_TEST_FAILURE_PATTERNS:
    print(f"  - {pattern}")
# Output includes: OSError, Timeout, ConnectionError, HTTPError, etc.
</syntaxhighlight>

== Related Pages ==

* (Leave empty for orphan pages)
