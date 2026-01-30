# Knowledge Learners Module
#
# Complete knowledge learning pipeline for acquiring knowledge from various sources.
#
# Architecture:
#   Source → Ingestor → WikiPages → Merger → Updated KG
#
# Main Usage:
#     from kapso.knowledge_base.learners import KnowledgePipeline
#     from kapso.knowledge.types import Source
#     
#     pipeline = KnowledgePipeline()
#     result = pipeline.run(Source.Repo("https://github.com/user/repo"))
#     result = pipeline.run(Source.Idea(query="...", source="...", content="..."))

# Source types (from unified types module)
from kapso.knowledge_base.learners.sources import Source

# Main pipeline orchestrator
from kapso.knowledge_base.learners.knowledge_learner_pipeline import (
    KnowledgePipeline,
    PipelineResult,
)

# Knowledge merger (Stage 2) - hierarchical sub-graph-aware merger
from kapso.knowledge_base.learners.merger import (
    KnowledgeMerger,
    MergeResult,
)

# Ingestors (Stage 1) - import for registration
from kapso.knowledge_base.learners.ingestors import (
    Ingestor,
    IngestorFactory,
    register_ingestor,
    RepoIngestor,
    ExperimentIngestor,
    IdeaIngestor,
    ImplementationIngestor,
    ResearchReportIngestor,
)

__all__ = [
    # Main pipeline
    "KnowledgePipeline",
    "PipelineResult",
    # Source types
    "Source",
    # Merger (Stage 2)
    "KnowledgeMerger",
    "MergeResult",
    # Ingestors (Stage 1)
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
    "RepoIngestor",
    "ExperimentIngestor",
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
