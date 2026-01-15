# Knowledge Ingestors
#
# Stage 1 of the knowledge learning pipeline.
# Ingestors take a knowledge source and return proposed WikiPages.
#
# Usage:
#     from src.knowledge.learners.ingestors import IngestorFactory
#     
#     ingestor = IngestorFactory.create("repo")
#     pages = ingestor.ingest(Source.Repo("https://github.com/user/repo"))

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import IngestorFactory, register_ingestor

# Import all ingestor implementations to register them
from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
from src.knowledge.learners.ingestors.experiment_ingestor import ExperimentIngestor

__all__ = [
    # Base classes
    "Ingestor",
    # Factory
    "IngestorFactory",
    "register_ingestor",
    # Implementations
    "RepoIngestor",
    "ExperimentIngestor",
]

