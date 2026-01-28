# Knowledge Learners Module
#
# Complete knowledge learning pipeline for acquiring knowledge from various sources.
#
# Architecture:
#   Source → Ingestor → WikiPages → Merger → Updated KG
#
# Main Usage:
#     from src.knowledge.learners import KnowledgePipeline
#     from src.knowledge.types import Source
#     
#     pipeline = KnowledgePipeline()
#     result = pipeline.run(Source.Repo("https://github.com/user/repo"))
#     result = pipeline.run(Source.Idea(query="...", source="...", content="..."))

# Source types (from unified types module)
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
