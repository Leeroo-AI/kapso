# Workflow Repository Builder
#
# Generates workflow repository files for GitHub deployment.
# Each workflow gets a dedicated repository structure with:
# - Step-by-step implementation files (one Python file per step)
# - requirements.txt with pinned dependencies
# - README.md with setup and execution instructions
# - Proper Python package structure
#
# File generation is deterministic. GitHub operations are handled
# by the agentic repo_builder phase which can adapt to edge cases
# like naming conflicts.

import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def sanitize_repo_name(name: str) -> str:
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


def _generate_gitignore() -> str:
    """Generate .gitignore content."""
    return """# Python
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


def prepare_workflow_repo(
    workflow_name: str,
    workflow_description: str,
    steps: List[Dict[str, Any]],
    repo_namespace: str,
) -> Path:
    """
    Generate all files for a workflow repository in a temporary directory.
    
    This function creates the repository structure but does NOT handle
    GitHub operations (repo creation, git init/commit/push). Those are
    handled by the agentic repo_builder phase.
    
    Args:
        workflow_name: Workflow page name (e.g., "unslothai_unsloth_QLoRA_Finetuning")
        workflow_description: One-line description of the workflow
        steps: List of step information dictionaries from WorkflowIndex
        repo_namespace: Source repository namespace
        
    Returns:
        Path to temporary directory containing generated files
        
    Example:
        temp_dir = prepare_workflow_repo(
            workflow_name="unslothai_unsloth_QLoRA_Finetuning",
            workflow_description="QLoRA fine-tuning workflow",
            steps=[
                {"name": "Data_Preparation", "api_call": "get_chat_template()", ...},
                {"name": "Model_Loading", "api_call": "FastLanguageModel.from_pretrained()", ...},
            ],
            repo_namespace="unslothai_unsloth",
        )
        # temp_dir now contains all files ready for GitHub push
    """
    logger.info(f"Preparing repository files for workflow: {workflow_name}")
    
    # Create temporary directory for repo content
    temp_dir = Path(tempfile.mkdtemp(prefix="praxium_workflow_"))
    
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
    logger.debug("Generated requirements.txt")
    
    # Generate README.md
    readme = _generate_readme(
        workflow_name, workflow_description, steps, repo_namespace
    )
    (temp_dir / "README.md").write_text(readme, encoding="utf-8")
    logger.debug("Generated README.md")
    
    # Generate workflow runner
    runner = _generate_workflow_runner(workflow_name, steps)
    (temp_dir / "run_workflow.py").write_text(runner, encoding="utf-8")
    logger.debug("Generated run_workflow.py")
    
    # Generate __init__.py for package structure
    init_content = f'"""Auto-generated workflow package: {workflow_name}"""\n'
    (temp_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    # Create .gitignore
    gitignore = _generate_gitignore()
    (temp_dir / ".gitignore").write_text(gitignore, encoding="utf-8")
    
    logger.info(f"Generated {len(steps)} step files + supporting files in {temp_dir}")
    
    return temp_dir


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
