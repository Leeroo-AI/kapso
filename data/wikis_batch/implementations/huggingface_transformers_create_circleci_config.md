# Implementation: huggingface_transformers_create_circleci_config

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Dynamic CircleCI configuration generator that creates workflow definitions based on repository state and test requirements.

=== Description ===

The `.circleci/create_circleci_config.py` module (412 lines) generates CircleCI configuration dynamically. Instead of a static config, it:
- Analyzes repository structure for test discovery
- Creates job definitions based on model types
- Configures parallelism and resource requirements
- Generates workflows for different test tiers (fast, slow, GPU)

=== Usage ===

Run before CI execution to generate the actual CircleCI config. Called automatically by CircleCI setup workflow.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/.circleci/create_circleci_config.py .circleci/create_circleci_config.py]
* '''Lines:''' 1-412

=== Signature ===
<syntaxhighlight lang="python">
def get_test_jobs() -> list[dict]:
    """Generate job definitions for test matrix."""

def create_workflow(jobs: list[dict]) -> dict:
    """Create CircleCI workflow from jobs."""

def generate_config() -> str:
    """Generate complete CircleCI config YAML."""

def main():
    """Generate and write config.yml."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python .circleci/create_circleci_config.py > .circleci/config.yml
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Repository state || git || Yes || Current repo structure
|-
| Test configuration || env || No || Override defaults
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config.yml || File || CircleCI workflow configuration
|}

== Usage Examples ==

=== Generate Config ===
<syntaxhighlight lang="bash">
# Generate CircleCI configuration
cd transformers
python .circleci/create_circleci_config.py

# Output written to .circleci/config.yml
</syntaxhighlight>

== Related Pages ==
