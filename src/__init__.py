# Tinkerer Agent
#
# Build robust software from knowledge and experimentation.
#
# Usage:
#     from src import Tinkerer, Source, DeployStrategy
#     
#     tinkerer = Tinkerer(domain="healthcare")
#     tinkerer.learn(Source.Paper("./triage.pdf"), target_kg="https://skills.leeroo.com")
#     solution = tinkerer.evolve(goal="Create a triage agent")
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

# Lazy load Tinkerer to avoid loading OrchestratorAgent (which has heavy deps)
def __getattr__(name):
    if name == "Tinkerer":
        from src.tinkerer import Tinkerer
        return Tinkerer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core API
    "Tinkerer",
    "Source",
    "SolutionResult",
    # Deployment
    "Software",
    "DeployStrategy",
    "DeployConfig",
    "DeploymentFactory",
]

__version__ = "0.1.0"
