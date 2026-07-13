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
#     kapso --goal "Build a web scraper"

# Suppress deprecation warnings from third-party dependencies (e.g., pydub's audioop)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydub")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="aider")

# All public symbols are loaded lazily via __getattr__.
#
# Why lazy? The MCP server subprocess (gated_mcp.server) imports from this
# package on every cold start. Eager imports of Source, SolutionResult, and
# DeployStrategy pull in the full execution stack (gitpython, weaviate, etc.)
# and add ~8 seconds to that cold start. With lazy loading the MCP subprocess
# boots in under 1 second and Claude Code's MCP connection check succeeds.
_LAZY_IMPORTS = {
    "Kapso":            ("kapso.kapso",              "Kapso"),
    "Source":           ("kapso.knowledge_base.learners", "Source"),
    "SolutionResult":   ("kapso.execution.solution", "SolutionResult"),
    "Software":         ("kapso.deployment",          "Software"),
    "DeployStrategy":   ("kapso.deployment",          "DeployStrategy"),
    "DeployConfig":     ("kapso.deployment",          "DeployConfig"),
    "DeploymentFactory":("kapso.deployment",          "DeploymentFactory"),
    "RunCheckpointError": (
        "kapso.execution.run_checkpoint",
        "RunCheckpointError",
    ),
    "RunCheckpointMissingError": (
        "kapso.execution.run_checkpoint",
        "RunCheckpointMissingError",
    ),
    "RunCheckpointCorruptError": (
        "kapso.execution.run_checkpoint",
        "RunCheckpointCorruptError",
    ),
    "RunCheckpointIncompatibleError": (
        "kapso.execution.run_checkpoint",
        "RunCheckpointIncompatibleError",
    ),
    "RunCheckpointCompletedError": (
        "kapso.execution.run_checkpoint",
        "RunCheckpointCompletedError",
    ),
    "IterationEvaluationContext": (
        "kapso.execution.iteration_evaluator",
        "IterationEvaluationContext",
    ),
    "IterationEvaluationResult": (
        "kapso.execution.iteration_evaluator",
        "IterationEvaluationResult",
    ),
    "IterationEvaluationError": (
        "kapso.execution.iteration_evaluator",
        "IterationEvaluationError",
    ),
    "IterationEvaluationValidationError": (
        "kapso.execution.iteration_evaluator",
        "IterationEvaluationValidationError",
    ),
    "IterationEvaluator": (
        "kapso.execution.iteration_evaluator",
        "IterationEvaluator",
    ),
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
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
    # Resume errors
    "RunCheckpointError",
    "RunCheckpointMissingError",
    "RunCheckpointCorruptError",
    "RunCheckpointIncompatibleError",
    "RunCheckpointCompletedError",
    # External candidate evaluation
    "IterationEvaluationContext",
    "IterationEvaluationResult",
    "IterationEvaluationError",
    "IterationEvaluationValidationError",
    "IterationEvaluator",
]

__version__ = "0.1.0"
