"""Static and configuration boundaries for the ideation replacement."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.search_strategies.generic.ideation.analyzer import (
    AnalyzerSettings,
)
from kapso.execution.search_strategies.generic.ideation.embeddings import (
    EmbeddingSettings,
)
from kapso.execution.search_strategies.generic.ideation.evidence_author import (
    EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    CANDIDATE_RESPONSE_SCHEMA,
    CandidateGeneratorSettings,
    GenerationMemberSettings,
)
from kapso.execution.search_strategies.generic.ideation.selector import (
    SELECTOR_RESPONSE_SCHEMA,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch

ROOT = Path(__file__).parents[1]
IDEATION = (
    ROOT / "src" / "kapso" / "execution" / "search_strategies" / "generic" / "ideation"
)


def test_shipped_candidate_pipeline_configuration_is_strict_and_shared():
    config = load_config(str(ROOT / "src" / "kapso" / "config.yaml"))
    assert "ideation_defaults" not in config
    defaults = config["ideation_profiles"]["DEFAULT"]
    agents = defaults["coding_agents"]
    generator = CandidateGeneratorSettings.from_dict(agents["generator"])
    selector = GenerationMemberSettings.from_dict(agents["selector"])
    evidence_author = GenerationMemberSettings.from_dict(agents["evidence_author"])
    embeddings = EmbeddingSettings.from_dict(defaults["embeddings"])
    analyzer = AnalyzerSettings.from_dict(defaults["analyzer"])
    assert len(generator.members) == defaults["operators"]["candidate_quota"]
    assert selector.cli in {"codex", "claude_code"}
    assert evidence_author.cli == "codex"
    assert evidence_author.allowed_tools == ("Read",)
    assert embeddings.enabled
    assert analyzer.minimum_distinct_eligible <= len(generator.members)


def test_superseded_generic_ideation_files_and_symbols_are_absent():
    generic = IDEATION.parent
    assert not (generic / "codex_ideation.py").exists()
    for name in (
        "ideation_claude_code.md",
        "ideation_ensemble_addendum.md",
        "ideation_selector.md",
    ):
        assert not (generic / "prompts" / name).exists()
    strategy_source = (generic / "strategy.py").read_text(encoding="utf-8")
    for symbol in (
        "_generate_solution_ensemble",
        "_select_from_candidates",
        "_salvage_ideation_output",
        "_fallback_solution",
        "normalize_parent_policy",
        "parent_policy",
        "ideation_ensemble",
    ):
        assert symbol not in strategy_source


def test_removed_generic_parameters_fail_instead_of_being_ignored():
    with pytest.raises(ValueError, match="unknown fields: parent_policy"):
        GenericSearch(SimpleNamespace(params={"parent_policy": "best"}))


def test_orchestrator_requires_one_named_generic_configuration_source():
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.mode_config = {
        "ideation_profile": "DEFAULT",
        "search_strategy": {"type": "generic", "params": {}},
    }
    strategy_type, params = orchestrator._resolve_search_strategy_config()
    assert strategy_type == "generic"
    assert (
        params["ideation"]
        == load_config(str(ROOT / "src" / "kapso" / "config.yaml"))[
            "ideation_profiles"
        ]["DEFAULT"]
    )

    orchestrator.mode_config = {"search_strategy": {"params": {}}}
    with pytest.raises(ValueError, match="explicit strategy type"):
        orchestrator._resolve_search_strategy_config()

    orchestrator.mode_config = {
        "ideation_profile": "DEFAULT",
        "search_strategy": {
            "type": "generic",
            "params": {"ideation": {}},
        },
    }
    with pytest.raises(ValueError, match="named canonical profile"):
        orchestrator._resolve_search_strategy_config()


def test_coding_agent_schemas_use_the_shared_cli_supported_subset():
    unsupported = {"minItems", "minLength", "minimum", "maximum", "uniqueItems"}

    def visit(value):
        if isinstance(value, dict):
            assert not (set(value) & unsupported)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(CANDIDATE_RESPONSE_SCHEMA)
    visit(SELECTOR_RESPONSE_SCHEMA)
    visit(EVIDENCE_AUTHOR_RESPONSE_SCHEMA)


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
            "evidence_author.py",
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
