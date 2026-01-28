# Knowledge Merger Module
#
# Hierarchical sub-graph-aware knowledge merger.
# Uses Claude Code agent with wiki MCP tools for intelligent merge decisions.
#
# Usage:
#     from src.knowledge.learners.merger import KnowledgeMerger, MergeResult
#     
#     merger = KnowledgeMerger()
#     result = merger.merge(pages, wiki_dir=Path("data/wikis"))

from src.knowledge.learners.merger.knowledge_merger import (
    KnowledgeMerger,
    MergeResult,
)

__all__ = [
    "KnowledgeMerger",
    "MergeResult",
]
