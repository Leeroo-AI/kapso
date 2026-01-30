# Workflow Repository Builder - Utilities
#
# Provides utility functions for the agentic repository builder phase.
# The actual repository structure and file generation is handled by
# the coding agent based on the domain-appropriate prompt.

import logging
import re

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
