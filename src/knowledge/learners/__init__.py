# Knowledge Learners Module
#
# Complete knowledge learning pipeline for acquiring knowledge from various sources.
#
# Architecture:
#   Source → Ingestor → WikiPages → Merger → Updated KG
#
# Main Usage:
#     from src.knowledge.learners import KnowledgePipeline, Source
#     
#     pipeline = KnowledgePipeline()
#     result = pipeline.run(Source.Repo("https://github.com/user/repo"))
#
# Components:
#   - KnowledgePipeline: Main orchestrator
#   - Source: Typed wrappers for knowledge sources (Repo, Paper, Doc, etc.)
#   - Ingestors: Stage 1 - extract WikiPages from sources (in ingestors/)
#   - KnowledgeMerger: Stage 2 - merge pages into existing KG

# Source types
from src.knowledge.learners.sources import Source

# Main pipeline orchestrator
from src.knowledge.learners.knowledge_learner_pipeline import (
    KnowledgePipeline,
    PipelineResult,
)

# Knowledge merger (Stage 2)
from src.knowledge.learners.knowledge_merger import (
    KnowledgeMerger,
    MergeResult,
    MergeAction,
)

# Ingestors (Stage 1) - import for registration
from src.knowledge.learners.ingestors import (
    Ingestor,
    IngestorFactory,
    register_ingestor,
    RepoIngestor,
    PaperIngestor,
    ExperimentIngestor,
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
    "MergeAction",
    # Ingestors (Stage 1)
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
    "RepoIngestor",
    "PaperIngestor",
    "ExperimentIngestor",
]
