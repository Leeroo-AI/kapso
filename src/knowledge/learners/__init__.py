# Knowledge Learners Module
#
# Modular learning system for knowledge acquisition.
# Each learner handles a specific source type (Repo, Paper, File, Experiment).
#
# Usage:
#     from src.knowledge.learners import Source, LearnerFactory
#     
#     expert.learn(Source.Repo("https://github.com/user/repo"), target_kg="https://skills.leeroo.com")

from src.knowledge.learners.base import Learner, KnowledgeChunk
from src.knowledge.learners.factory import LearnerFactory, register_learner
from src.knowledge.learners.sources import Source

# Import all learner implementations to register them
from src.knowledge.learners.repo_learner import RepoLearner
from src.knowledge.learners.paper_learner import PaperLearner
from src.knowledge.learners.experiment_learner import ExperimentLearner

__all__ = [
    # Source types
    "Source",
    # Base classes
    "Learner",
    "KnowledgeChunk",
    # Factory
    "LearnerFactory",
    "register_learner",
    # Implementations
    "RepoLearner",
    "PaperLearner",
    "ExperimentLearner",
]

