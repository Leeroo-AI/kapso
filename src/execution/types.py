# Execution Types
#
# Common types used across the execution module.

from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextData:
    """
    Output data structure from context gathering.
    
    Contains all context needed for solution generation:
    - problem: The problem description
    - additional_info: Experiment history and other info
    - kg_results: Knowledge graph text results
    - kg_code_results: Knowledge graph code snippets
    """
    problem: str
    additional_info: str
    kg_results: Optional[str] = ""
    kg_code_results: Optional[str] = ""
