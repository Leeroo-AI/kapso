# Knowledge Ingestors
#
# Stage 1 of the knowledge learning pipeline.
# Ingestors take a knowledge source and return proposed WikiPages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import IngestorFactory
#     
#     ingestor = IngestorFactory.create("repo")
#     pages = ingestor.ingest(Source.Repo("https://github.com/user/repo"))

from kapso.knowledge_base.learners.ingestors.base import Ingestor
from kapso.knowledge_base.learners.ingestors.factory import IngestorFactory, register_ingestor

# Import all ingestor implementations to register them
from kapso.knowledge_base.learners.ingestors.repo_ingestor import RepoIngestor
from kapso.knowledge_base.learners.ingestors.experiment_ingestor import ExperimentIngestor

# Research output ingestors (agentic, from research_ingestor package)
from kapso.knowledge_base.learners.ingestors.research_ingestor import (
    IdeaIngestor,
    ImplementationIngestor,
    ResearchReportIngestor,
)

__all__ = [
    # Base classes
    "Ingestor",
    # Factory
    "IngestorFactory",
    "register_ingestor",
    # Implementations
    "RepoIngestor",
    "ExperimentIngestor",
    # Research output ingestors
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
