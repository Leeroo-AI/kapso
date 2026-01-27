# Knowledge Learners Module
#
# Complete knowledge learning pipeline for acquiring knowledge from various sources.
#
# Architecture:
#   Source → Ingestor → WikiPages → Merger → Updated KG
#
# The KG is stored in:
#   - Neo4j: Graph structure (nodes + edges) - THE INDEX
#   - Weaviate: Embeddings for semantic search
#   - Source files: Ground truth .md files
#
# Main Usage:
#     from src.knowledge.learners import KnowledgePipeline, Source
#     
#     pipeline = KnowledgePipeline()
#     result = pipeline.run(Source.Repo("https://github.com/user/repo"))
#
# Components:
#   - KnowledgePipeline: Main orchestrator
#   - Source: Typed wrappers for knowledge sources (Repo, Solution)
#   - Ingestors: Stage 1 - extract WikiPages from sources (in ingestors/)
#   - KnowledgeMerger: Stage 2 - merge pages into existing KG
#   - MergeHandlers: Type-specific merge logic (in merge_handlers/)

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
)

# Merge handlers
from src.knowledge.learners.merge_handlers import (
    MergeHandler,
    WorkflowMergeHandler,
    PrincipleMergeHandler,
    ImplementationMergeHandler,
    EnvironmentMergeHandler,
    HeuristicMergeHandler,
)

# Ingestors (Stage 1) - import for registration
from src.knowledge.learners.ingestors import (
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
    # Merge handlers
    "MergeHandler",
    "WorkflowMergeHandler",
    "PrincipleMergeHandler",
    "ImplementationMergeHandler",
    "EnvironmentMergeHandler",
    "HeuristicMergeHandler",
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

# Episodic Learner - learns from experiment history
from src.knowledge.learners.episodic_learner import (
    EpisodicLearner,
    LLMEpisodicLearner,
    EpisodicLearnerFactory,
    EpisodicLearnerResult,
    LearnedPattern,
    LearnedHeuristic,
    HarvestedWorkflow,
)

__all__ += [
    # Episodic Learner
    "EpisodicLearner",
    "LLMEpisodicLearner", 
    "EpisodicLearnerFactory",
    "EpisodicLearnerResult",
    "LearnedPattern",
    "LearnedHeuristic",
    "HarvestedWorkflow",
]
