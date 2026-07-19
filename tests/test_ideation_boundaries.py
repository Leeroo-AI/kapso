"""Static and configuration boundaries for the ideation replacement."""

import ast
from pathlib import Path

from kapso.core.config import load_config
from kapso.execution.search_strategies.generic.ideation.analyzer import (
    AnalyzerSettings,
)
from kapso.execution.search_strategies.generic.ideation.embeddings import (
    EmbeddingSettings,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    CandidateGeneratorSettings,
    GenerationMemberSettings,
)

ROOT = Path(__file__).parents[1]
IDEATION = (
    ROOT / "src" / "kapso" / "execution" / "search_strategies" / "generic" / "ideation"
)


def test_shipped_candidate_pipeline_configuration_is_strict_and_shared():
    config = load_config(str(ROOT / "src" / "kapso" / "config.yaml"))
    defaults = config["ideation_defaults"]
    assert (
        config["modes"]["GENERIC"]["ideation"] is config["modes"]["MINIMAL"]["ideation"]
    )
    agents = defaults["coding_agents"]
    generator = CandidateGeneratorSettings.from_dict(agents["generator"])
    selector = GenerationMemberSettings.from_dict(agents["selector"])
    embeddings = EmbeddingSettings.from_dict(defaults["embeddings"])
    analyzer = AnalyzerSettings.from_dict(defaults["analyzer"])
    assert len(generator.members) == defaults["operators"]["candidate_quota"]
    assert selector.cli in {"codex", "claude_code"}
    assert embeddings.enabled
    assert analyzer.minimum_distinct_eligible <= len(generator.members)


def test_only_embedding_boundary_imports_the_openai_sdk():
    importers = []
    for path in IDEATION.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.split(".")[0] == "openai"
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name.split(".")[0] == "openai" for alias in node.names)
            )
            for node in ast.walk(tree)
        ):
            importers.append(path.name)
    assert importers == ["embeddings.py"]


def test_ideation_source_has_no_exception_swallowing_or_environment_reads():
    violations = []
    for path in IDEATION.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                violations.append(f"{path.name}:try")
            if isinstance(node, ast.Attribute) and (
                (isinstance(node.value, ast.Name) and node.value.id == "os")
                and node.attr in {"environ", "getenv"}
            ):
                violations.append(f"{path.name}:environment")
    assert violations == []


def test_candidate_reasoning_has_no_direct_model_api_or_prompt_truncation():
    sources = {
        path.name: path.read_text(encoding="utf-8") for path in IDEATION.glob("*.py")
    }
    reasoning_sources = "\n".join(
        sources[name]
        for name in (
            "coding_agents.py",
            "generator.py",
            "analyzer.py",
            "selector.py",
        )
    )
    for banned in (
        "LLMBackend",
        "litellm",
        "llm_completion",
        "chat.completions",
        "responses.create",
        "Anthropic(",
    ):
        assert banned not in reasoning_sources
    assert "prompt[:" not in reasoning_sources
    assert "prompt[-" not in reasoning_sources
