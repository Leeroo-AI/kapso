# Expert Agent
#
# Build robust software from knowledge and experimentation.
#
# Usage:
#     from src import Expert, Source, DeployStrategy
#     
#     expert = Expert(domain="healthcare")
#     expert.learn(Source.Paper("./triage.pdf"), target_kg="https://skills.leeroo.com")
#     solution = expert.build(goal="Create a triage agent")
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

# Lazy load Expert to avoid loading OrchestratorAgent (which has heavy deps)
def __getattr__(name):
    if name == "Expert":
        from src.expert import Expert
        return Expert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core API
    "Expert",
    "Source",
    "SolutionResult",
    # Deployment
    "Software",
    "DeployStrategy",
    "DeployConfig",
    "DeploymentFactory",
]

__version__ = "0.1.0"
