# Kapso Agent
#
# Build robust software from knowledge and experimentation.
#
# Usage:
#     from src import Kapso, Source, DeployStrategy
#     
#     kapso = Kapso()
#     kapso.learn(Source.Repo("https://github.com/user/repo"), wiki_dir="data/wikis")
#     solution = kapso.evolve(goal="Create a triage agent")
#     software = solution.deploy()
#     result = software.run({"symptoms": "headache"})
#
# CLI:
#     python -m src.cli --goal "Build a web scraper"

# Light imports (no heavy dependencies)
from src.knowledge.learners import Source
from src.execution.solution import SolutionResult
from src.deployment import (
    Software,
    DeployStrategy,
    DeployConfig,
    DeploymentFactory,
)

# Lazy load Kapso to avoid loading OrchestratorAgent (which has heavy deps)
def __getattr__(name):
    if name == "Kapso":
        from src.kapso import Kapso
        return Kapso
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core API
    "Kapso",
    "Source",
    "SolutionResult",
    # Deployment
    "Software",
    "DeployStrategy",
    "DeployConfig",
    "DeploymentFactory",
]

__version__ = "0.1.0"
