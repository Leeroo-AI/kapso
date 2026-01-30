"""
Repository Memory (RepoMemory)
==============================

This module provides "memory about the repository at hand" that is:

- **Branch-scoped**: memory is committed into each experiment branch under `.kapso/`,
  so when Kapso continues from a branch, it naturally continues with the memory
  that corresponds to that exact code state.
- **Evidence-backed**: semantic claims (e.g., algorithms/architecture) must cite
  exact quotes from files in the repo. This prevents "hallucinated" repo memories.
- **Experiment-aware**: memory tracks what idea/spec was implemented and what diffs
  occurred, so the system can update its understanding as the experiment tree evolves.

The primary public entrypoint is `RepoMemoryManager`.
"""

from kapso.execution.memories.repo_memory.manager import RepoMemoryManager

__all__ = ["RepoMemoryManager"]

