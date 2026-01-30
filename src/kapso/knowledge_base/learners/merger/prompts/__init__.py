# Merger Prompts Module
#
# Provides prompt templates for the hierarchical knowledge merger.
#
# Usage:
#     from kapso.knowledge_base.learners.merger.prompts import load_prompt
#     
#     prompt_template = load_prompt("hierarchical_merge")

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """
    Load a prompt template by name.
    
    Args:
        name: Prompt name (without .md extension)
        
    Returns:
        Prompt template content as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


__all__ = ["load_prompt", "PROMPTS_DIR"]
