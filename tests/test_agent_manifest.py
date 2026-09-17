"""All manifest consumers share one source and fail on invalid registries."""

import pytest
import yaml

from kapso.core import agent_manifest, api_endpoint, preflight
from kapso.execution.coding_agents import factory


def test_consumers_share_manifest_changes(monkeypatch, tmp_path):
    manifest = agent_manifest.load_agent_manifest()
    manifest["default_agent"] = "codex"
    manifest["agents"]["openai_compatible"]["agent_specific"]["timeout"] = 31
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(agent_manifest, "AGENTS_YAML_PATH", path)
    monkeypatch.setattr(factory.CodingAgentFactory, "_registry", {})
    monkeypatch.setattr(factory.CodingAgentFactory, "_agent_configs", {})
    monkeypatch.setattr(factory.CodingAgentFactory, "_default_agent", "")
    factory._register_from_yaml()
    assert factory.CodingAgentFactory.get_default_agent() == "codex"
    assert (
        factory.CodingAgentFactory.build_config(
            "openai_compatible"
        ).agent_specific["timeout"]
        == 31
    )
    assert api_endpoint.openai_compatible_options()["timeout"] == 31
    assert preflight.load_agent_manifest() == manifest


@pytest.mark.parametrize(
    "content,error",
    [
        ("[", yaml.YAMLError),
        ("[]", ValueError),
        ("agents: []", ValueError),
        ("agents: {}\ndefault_agent: unknown", ValueError),
    ],
)
def test_invalid_manifest_fails_loud(monkeypatch, tmp_path, content, error):
    path = tmp_path / "agents.yaml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(agent_manifest, "AGENTS_YAML_PATH", path)
    with pytest.raises(error):
        agent_manifest.load_agent_manifest()


def test_missing_manifest_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(
        agent_manifest, "AGENTS_YAML_PATH", tmp_path / "missing.yaml"
    )
    with pytest.raises(FileNotFoundError):
        agent_manifest.load_agent_manifest()
