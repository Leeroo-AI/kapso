# Workflow Repository Builder
#
# Creates private GitHub repositories from workflow implementations.
# Each workflow gets a dedicated repository with:
# - Step-by-step implementation files (one Python file per step)
# - requirements.txt with pinned dependencies
# - README.md with setup and execution instructions
# - Proper Python package structure
#
# This module integrates with the repo ingestion pipeline to ensure
# deterministic, version-controlled workflow implementations.

import logging
import os
import re
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# GitHub API URL
GITHUB_API_URL = "https://api.github.com"


def _get_github_token() -> str:
    """
    Get GitHub Personal Access Token from environment.
    
    Returns:
        GitHub PAT
        
    Raises:
        ValueError: If GITHUB_PAT is not set
    """
    token = os.getenv("GITHUB_PAT")
    if not token:
        raise ValueError(
            "GITHUB_PAT environment variable is not set. "
            "Please add your GitHub Personal Access Token to .env file."
        )
    return token


def _sanitize_repo_name(name: str) -> str:
    """
    Convert a workflow name to a valid GitHub repository name.
    
    GitHub repo names must:
    - Be lowercase
    - Use hyphens instead of underscores/spaces
    - Not contain special characters
    
    Args:
        name: Workflow name (e.g., "unslothai_unsloth_QLoRA_Finetuning")
        
    Returns:
        Valid GitHub repo name (e.g., "workflow-unslothai-unsloth-qlora-finetuning")
    """
    # Convert to lowercase and replace underscores with hyphens
    sanitized = name.lower().replace("_", "-")
    # Remove any non-alphanumeric characters except hyphens
    sanitized = re.sub(r"[^a-z0-9-]", "", sanitized)
    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Add workflow prefix
    return f"workflow-{sanitized}"


def _create_github_repo(
    repo_name: str,
    description: str,
    private: bool = True,
    org: Optional[str] = None,
) -> str:
    """
    Create a new GitHub repository using the API.
    
    Args:
        repo_name: Repository name (e.g., "workflow-unsloth-qlora-finetuning")
        description: Repository description
        private: Whether the repo should be private (default: True)
        org: Optional organization name (creates under user if None)
        
    Returns:
        Repository URL (e.g., "https://github.com/username/repo-name")
        
    Raises:
        RuntimeError: If repository creation fails
    """
    import requests
    
    token = _get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    # Create repo payload
    payload = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False,  # We'll push our own content
    }
    
    # Determine API endpoint based on org or user
    if org:
        url = f"{GITHUB_API_URL}/orgs/{org}/repos"
    else:
        url = f"{GITHUB_API_URL}/user/repos"
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        repo_data = response.json()
        repo_url = repo_data["html_url"]
        logger.info(f"Created GitHub repository: {repo_url}")
        return repo_url
    elif response.status_code == 422:
        # Repository might already exist
        error_data = response.json()
        errors = error_data.get("errors", [])
        for error in errors:
            if error.get("message") == "name already exists on this account":
                # Return existing repo URL
                if org:
                    repo_url = f"https://github.com/{org}/{repo_name}"
                else:
                    # Get username from API
                    user_response = requests.get(
                        f"{GITHUB_API_URL}/user",
                        headers=headers,
                    )
                    if user_response.status_code == 200:
                        username = user_response.json()["login"]
                        repo_url = f"https://github.com/{username}/{repo_name}"
                        logger.info(f"Repository already exists: {repo_url}")
                        return repo_url
        raise RuntimeError(f"Failed to create repository: {error_data}")
    else:
        raise RuntimeError(
            f"GitHub API error ({response.status_code}): {response.text}"
        )


def _generate_step_file(
    step_num: int,
    step_name: str,
    step_info: Dict[str, Any],
    repo_namespace: str,
) -> str:
    """
    Generate a Python implementation file for a workflow step.
    
    Args:
        step_num: Step number (1-indexed)
        step_name: Step name (e.g., "Data_Preparation")
        step_info: Step information from WorkflowIndex
        repo_namespace: Repository namespace for imports
        
    Returns:
        Python source code for the step
    """
    # Extract info from step_info
    api_call = step_info.get("api_call", "")
    source_location = step_info.get("source_location", "")
    external_deps = step_info.get("external_dependencies", [])
    key_params = step_info.get("key_parameters", "")
    inputs = step_info.get("inputs", "")
    outputs = step_info.get("outputs", "")
    
    # Parse dependencies for imports
    imports = []
    if isinstance(external_deps, list):
        for dep in external_deps:
            # Common mapping of dependency names to import statements
            dep_lower = dep.lower().strip()
            if dep_lower == "transformers":
                imports.append("from transformers import AutoTokenizer, AutoModelForCausalLM")
            elif dep_lower == "torch":
                imports.append("import torch")
            elif dep_lower == "bitsandbytes":
                imports.append("# bitsandbytes is used internally by transformers")
            elif dep_lower in ("trl", "sfttrainer"):
                imports.append("from trl import SFTTrainer, SFTConfig")
            elif dep_lower == "peft":
                imports.append("from peft import get_peft_model, LoraConfig")
            elif dep_lower == "datasets":
                imports.append("from datasets import load_dataset")
            else:
                imports.append(f"import {dep}")
    
    # Convert step name to function name
    func_name = step_name.lower().replace(" ", "_")
    
    # Build the source code
    code = f'''"""
Step {step_num}: {step_name}

This module implements the {step_name} step of the workflow.

API Reference: {api_call}
Source Location: {source_location}

Inputs: {inputs}
Outputs: {outputs}
"""

import logging
from typing import Any, Dict, Optional, Tuple

{chr(10).join(imports) if imports else "# No external imports required"}

logger = logging.getLogger(__name__)


def {func_name}(
    *args,
    **kwargs,
) -> Any:
    """
    Execute the {step_name} step.
    
    This function implements the core logic for this workflow step.
    
    Args:
        *args: Positional arguments (see step documentation)
        **kwargs: Keyword arguments including:
            {key_params if key_params else "- See implementation for details"}
    
    Returns:
        {outputs if outputs else "Step output (see implementation)"}
    
    Example:
        >>> result = {func_name}(...)
    """
    logger.info("Starting {step_name}...")
    
    # TODO: Implement step logic based on:
    # API: {api_call}
    # Source: {source_location}
    
    raise NotImplementedError(
        "Step implementation pending. "
        "Refer to {source_location} for reference implementation."
    )


def main():
    """Run this step standalone for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="{step_name} - Step {step_num} of the workflow"
    )
    # Add arguments based on inputs
    args = parser.parse_args()
    
    # Run the step
    result = {func_name}()
    print(f"Step {step_num} completed: {{result}}")


if __name__ == "__main__":
    main()
'''
    return code


def _generate_requirements_txt(steps: List[Dict[str, Any]]) -> str:
    """
    Generate requirements.txt from workflow step dependencies.
    
    Args:
        steps: List of step information dictionaries
        
    Returns:
        Contents of requirements.txt with pinned versions
    """
    # Collect all unique dependencies
    all_deps = set()
    for step in steps:
        deps = step.get("external_dependencies", [])
        if isinstance(deps, list):
            all_deps.update(deps)
        elif isinstance(deps, str):
            all_deps.update(d.strip() for d in deps.split(","))
    
    # Map dependency names to pip package names with versions
    # These are common ML packages with reasonable version pins
    dep_mapping = {
        "transformers": "transformers>=4.40.0",
        "torch": "torch>=2.0.0",
        "bitsandbytes": "bitsandbytes>=0.43.0",
        "peft": "peft>=0.10.0",
        "trl": "trl>=0.8.0",
        "datasets": "datasets>=2.18.0",
        "accelerate": "accelerate>=0.28.0",
        "safetensors": "safetensors>=0.4.0",
        "sentencepiece": "sentencepiece>=0.2.0",
        "protobuf": "protobuf>=4.25.0",
        "numpy": "numpy>=1.24.0",
        "pandas": "pandas>=2.0.0",
        "scikit-learn": "scikit-learn>=1.4.0",
        "wandb": "wandb>=0.16.0",
        "tensorboard": "tensorboard>=2.15.0",
    }
    
    requirements = ["# Auto-generated requirements for this workflow", ""]
    
    for dep in sorted(all_deps):
        dep_lower = dep.lower().strip()
        if dep_lower in dep_mapping:
            requirements.append(dep_mapping[dep_lower])
        elif dep_lower:
            # Unknown dependency - add without version pin
            requirements.append(f"{dep_lower}")
    
    # Add common requirements that are always needed
    if not any("torch" in r for r in requirements):
        requirements.append("torch>=2.0.0")
    
    return "\n".join(requirements) + "\n"


def _generate_readme(
    workflow_name: str,
    description: str,
    steps: List[Dict[str, Any]],
    repo_namespace: str,
) -> str:
    """
    Generate README.md for the workflow repository.
    
    Args:
        workflow_name: Workflow name
        description: Workflow description
        steps: List of step information dictionaries
        repo_namespace: Source repository namespace
        
    Returns:
        Contents of README.md
    """
    # Build step list
    step_list = []
    for i, step in enumerate(steps, 1):
        step_name = step.get("name", f"Step {i}")
        step_desc = step.get("description", "")
        step_list.append(f"{i}. **{step_name}**: {step_desc}")
    
    readme = f'''# {workflow_name}

{description}

## Source Repository

This workflow implementation is derived from: `{repo_namespace}`

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Sufficient GPU memory (see individual steps for requirements)

## Installation

```bash
# Clone this repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow Steps

{chr(10).join(step_list)}

## Usage

### Run Individual Steps

Each step can be run independently:

```bash
python step_01_*.py --help
```

### Run Full Workflow

To run the complete workflow:

```bash
python run_workflow.py
```

## Configuration

Configuration options are available in each step module.
See the docstrings and argument parsers for details.

## License

This workflow implementation is subject to the license terms of the source repository.

## Generated

This repository was auto-generated by Praxium Repo Ingestor on {datetime.now().strftime("%Y-%m-%d")}.
'''
    return readme


def _generate_workflow_runner(
    workflow_name: str,
    steps: List[Dict[str, Any]],
) -> str:
    """
    Generate a main workflow runner script.
    
    Args:
        workflow_name: Workflow name
        steps: List of step information dictionaries
        
    Returns:
        Python source code for run_workflow.py
    """
    # Build step imports and calls
    step_imports = []
    step_calls = []
    
    for i, step in enumerate(steps, 1):
        step_name = step.get("name", f"step_{i}")
        func_name = step_name.lower().replace(" ", "_")
        module_name = f"step_{i:02d}_{func_name}"
        
        step_imports.append(f"from {module_name} import {func_name}")
        step_calls.append(f'''
    # Step {i}: {step_name}
    logger.info("=" * 60)
    logger.info("Step {i}: {step_name}")
    logger.info("=" * 60)
    try:
        result_{i} = {func_name}()
        logger.info(f"Step {i} completed successfully")
    except NotImplementedError:
        logger.warning("Step {i} not yet implemented, skipping...")
        result_{i} = None
    except Exception as e:
        logger.error(f"Step {i} failed: {{e}}")
        if not args.continue_on_error:
            raise
        result_{i} = None
''')
    
    code = f'''"""
{workflow_name} - Full Workflow Runner

This script orchestrates the complete workflow execution.
Run individual steps for more control.
"""

import argparse
import logging
import sys

{chr(10).join(step_imports)}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_workflow(args):
    """Execute the full workflow."""
    logger.info("Starting workflow: {workflow_name}")
    logger.info("=" * 60)
    {"".join(step_calls)}
    logger.info("=" * 60)
    logger.info("Workflow completed!")
    
    return {{
        {", ".join(f'"step_{i}": result_{i}' for i in range(1, len(steps) + 1))}
    }}


def main():
    parser = argparse.ArgumentParser(
        description="{workflow_name} - Full Workflow"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue execution even if a step fails",
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default="",
        help="Comma-separated list of step numbers to skip",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_workflow(args)
        logger.info(f"Final results: {{results}}")
    except Exception as e:
        logger.error(f"Workflow failed: {{e}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    return code


def _push_to_github(
    local_path: Path,
    repo_url: str,
    branch: str = "main",
) -> bool:
    """
    Push local repository content to GitHub.
    
    Args:
        local_path: Path to local git repository
        repo_url: GitHub repository URL
        branch: Branch name to push to
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    token = _get_github_token()
    
    # Add token to URL for authentication
    # https://github.com/user/repo -> https://token@github.com/user/repo
    auth_url = repo_url.replace("https://", f"https://{token}@")
    
    try:
        # Initialize git repo if not already
        subprocess.run(
            ["git", "init"],
            cwd=local_path,
            check=True,
            capture_output=True,
        )
        
        # Configure git user (required for commits)
        subprocess.run(
            ["git", "config", "user.email", "praxium-bot@example.com"],
            cwd=local_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Praxium Bot"],
            cwd=local_path,
            check=True,
            capture_output=True,
        )
        
        # Add all files
        subprocess.run(
            ["git", "add", "."],
            cwd=local_path,
            check=True,
            capture_output=True,
        )
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", "Initial workflow implementation"],
            cwd=local_path,
            check=True,
            capture_output=True,
        )
        
        # Add remote
        subprocess.run(
            ["git", "remote", "add", "origin", auth_url],
            cwd=local_path,
            capture_output=True,  # Don't check - might already exist
        )
        
        # Push
        result = subprocess.run(
            ["git", "push", "-u", "origin", branch, "--force"],
            cwd=local_path,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"Git push failed: {result.stderr}")
            return False
        
        logger.info(f"Successfully pushed to {repo_url}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False


def build_workflow_repo(
    workflow_name: str,
    workflow_description: str,
    steps: List[Dict[str, Any]],
    repo_namespace: str,
    org: Optional[str] = None,
    private: bool = True,
) -> Tuple[str, bool]:
    """
    Build and push a GitHub repository for a workflow.
    
    This is the main entry point for the repo builder.
    
    Args:
        workflow_name: Workflow page name (e.g., "unslothai_unsloth_QLoRA_Finetuning")
        workflow_description: One-line description of the workflow
        steps: List of step information dictionaries from WorkflowIndex
        repo_namespace: Source repository namespace
        org: Optional GitHub organization to create repo under
        private: Whether the repo should be private (default: True)
        
    Returns:
        Tuple of (repo_url, success_bool)
        
    Example:
        url, success = build_workflow_repo(
            workflow_name="unslothai_unsloth_QLoRA_Finetuning",
            workflow_description="QLoRA fine-tuning workflow",
            steps=[
                {"name": "Data_Preparation", "api_call": "get_chat_template()", ...},
                {"name": "Model_Loading", "api_call": "FastLanguageModel.from_pretrained()", ...},
            ],
            repo_namespace="unslothai_unsloth",
        )
    """
    logger.info(f"Building repository for workflow: {workflow_name}")
    
    # Sanitize repo name for GitHub
    repo_name = _sanitize_repo_name(workflow_name)
    
    # Create temporary directory for repo content
    temp_dir = Path(tempfile.mkdtemp(prefix="praxium_workflow_"))
    
    try:
        # Generate step files
        for i, step in enumerate(steps, 1):
            step_name = step.get("name", f"Step_{i}")
            func_name = step_name.lower().replace(" ", "_")
            filename = f"step_{i:02d}_{func_name}.py"
            
            step_code = _generate_step_file(i, step_name, step, repo_namespace)
            (temp_dir / filename).write_text(step_code, encoding="utf-8")
            logger.debug(f"Generated {filename}")
        
        # Generate requirements.txt
        requirements = _generate_requirements_txt(steps)
        (temp_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
        
        # Generate README.md
        readme = _generate_readme(
            workflow_name, workflow_description, steps, repo_namespace
        )
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")
        
        # Generate workflow runner
        runner = _generate_workflow_runner(workflow_name, steps)
        (temp_dir / "run_workflow.py").write_text(runner, encoding="utf-8")
        
        # Generate __init__.py for package structure
        init_content = f'"""Auto-generated workflow package: {workflow_name}"""\n'
        (temp_dir / "__init__.py").write_text(init_content, encoding="utf-8")
        
        # Create .gitignore
        gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project
*.log
.env
wandb/
outputs/
checkpoints/
"""
        (temp_dir / ".gitignore").write_text(gitignore, encoding="utf-8")
        
        # Create GitHub repository
        try:
            repo_url = _create_github_repo(
                repo_name=repo_name,
                description=workflow_description[:100],  # GitHub limits description length
                private=private,
                org=org,
            )
        except Exception as e:
            logger.error(f"Failed to create GitHub repository: {e}")
            return "", False
        
        # Push to GitHub
        success = _push_to_github(temp_dir, repo_url)
        
        if success:
            logger.info(f"Successfully created workflow repository: {repo_url}")
        else:
            logger.error(f"Failed to push to GitHub: {repo_url}")
        
        return repo_url, success
        
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_workflow_index_for_steps(
    workflow_index_path: Path,
    workflow_name: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse the WorkflowIndex to extract step information for a specific workflow.
    
    Args:
        workflow_index_path: Path to _WorkflowIndex.md
        workflow_name: Name of the workflow to extract
        
    Returns:
        Tuple of (description, steps_list)
    """
    if not workflow_index_path.exists():
        logger.warning(f"WorkflowIndex not found: {workflow_index_path}")
        return "", []
    
    content = workflow_index_path.read_text(encoding="utf-8")
    
    # Find the workflow section
    # Pattern: ## Workflow: {workflow_name}
    workflow_pattern = rf"## Workflow:\s*{re.escape(workflow_name)}"
    workflow_match = re.search(workflow_pattern, content)
    
    if not workflow_match:
        logger.warning(f"Workflow not found in index: {workflow_name}")
        return "", []
    
    # Extract the section until the next workflow or end
    section_start = workflow_match.end()
    next_workflow = re.search(r"\n## Workflow:", content[section_start:])
    section_end = section_start + next_workflow.start() if next_workflow else len(content)
    section = content[section_start:section_end]
    
    # Extract description
    desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", section)
    description = desc_match.group(1).strip() if desc_match else ""
    
    # Extract steps from attribute tables
    # Pattern: ### Step N: Step_Name
    steps = []
    step_pattern = r"### Step (\d+):\s*(\w+)"
    
    for step_match in re.finditer(step_pattern, section):
        step_num = int(step_match.group(1))
        step_name = step_match.group(2)
        
        # Find the attribute table for this step
        step_start = step_match.end()
        next_step = re.search(r"\n### Step", section[step_start:])
        step_end = step_start + next_step.start() if next_step else len(section)
        step_section = section[step_start:step_end]
        
        # Parse attribute table
        step_info = {"name": step_name}
        
        # Extract attributes from table rows
        # Pattern: | **Attribute** | Value |
        attr_patterns = [
            (r"\|\s*\*\*API Call\*\*\s*\|\s*(.+?)\s*\|", "api_call"),
            (r"\|\s*\*\*Source Location\*\*\s*\|\s*(.+?)\s*\|", "source_location"),
            (r"\|\s*\*\*External Dependencies\*\*\s*\|\s*(.+?)\s*\|", "external_dependencies"),
            (r"\|\s*\*\*Key Parameters\*\*\s*\|\s*(.+?)\s*\|", "key_parameters"),
            (r"\|\s*\*\*Inputs\*\*\s*\|\s*(.+?)\s*\|", "inputs"),
            (r"\|\s*\*\*Outputs\*\*\s*\|\s*(.+?)\s*\|", "outputs"),
        ]
        
        for pattern, key in attr_patterns:
            match = re.search(pattern, step_section)
            if match:
                value = match.group(1).strip()
                # Clean up backticks
                value = value.replace("`", "")
                step_info[key] = value
        
        # Parse external dependencies as list
        if "external_dependencies" in step_info:
            deps = step_info["external_dependencies"]
            step_info["external_dependencies"] = [
                d.strip() for d in deps.split(",") if d.strip()
            ]
        
        steps.append(step_info)
    
    # Sort by step number (in case they're out of order)
    steps.sort(key=lambda s: int(re.search(r"\d+", s.get("name", "0")) or 0))
    
    return description, steps
