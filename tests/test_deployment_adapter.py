"""The adapter's contract around the coding-agent session, with the session faked.

deploy() promised env_vars it never delivered, exported a validator it never
ran, copied .git into every adapted workspace, and gave every solution the
same container name. These pin the fixed behaviour; the session itself is a
fake that writes what the strategy expects.
"""

import json
import os
import stat
import types

import pytest

from kapso.deployment.adapter import agent as adapter_module
from kapso.deployment.adapter.agent import AdapterAgent, write_env_file
from kapso.deployment.adapter.validator import AdaptationValidator
from kapso.deployment.base import DeploymentSetting
from kapso.deployment.strategies import StrategyRegistry
from kapso.deployment.strategies.base import deployment_context
from kapso.execution.solution import SolutionResult


@pytest.fixture
def solution(tmp_path):
    src = tmp_path / "sol"
    src.mkdir()
    (src / "model.py").write_text("def score(x):\n    return x\n")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"\x00")
    return SolutionResult(goal="score things", code_path=str(src))


def setting_for(strategy):
    registry = StrategyRegistry.get()
    return DeploymentSetting(
        strategy=strategy, provider=registry.get_provider(strategy),
        resources={}, interface=registry.get_interface(strategy), reasoning="test",
    )


class ScriptedAgent:
    """Fake coding-agent session: runs a callable against the workspace."""
    def __init__(self, script):
        self.script = script
        self.prompts = []

    def initialize(self, workspace):
        self.workspace = workspace

    def generate_code(self, prompt):
        self.prompts.append(prompt)
        output = self.script(self.workspace, prompt) or ""
        return types.SimpleNamespace(success=True, files_changed=["main.py"], output=output, error=None)

    def cleanup(self):
        pass


@pytest.fixture
def scripted(monkeypatch):
    """Route CodingAgentFactory to a ScriptedAgent; returns a setter for the script."""
    holder = {}

    def build_config(**kwargs):
        return kwargs

    def create(config):
        return holder["agent"]

    monkeypatch.setattr(adapter_module.CodingAgentFactory, "build_config", staticmethod(build_config))
    monkeypatch.setattr(adapter_module.CodingAgentFactory, "create", staticmethod(create))

    def use(script):
        holder["agent"] = ScriptedAgent(script)
        return holder["agent"]
    return use


def writes_main(workspace, prompt):
    with open(os.path.join(workspace, "main.py"), "w") as f:
        f.write("def predict(inputs):\n    return {'ok': True}\n")
    return '<run_interface>{"type": "function", "module": "main", "callable": "predict"}</run_interface>'


# =============================================================================
# env_vars
# =============================================================================

def test_write_env_file_merges_keeps_comments_and_is_owner_only(tmp_path):
    (tmp_path / ".env").write_text("# keep me\nOLD=1\nAPI_KEY=stale\n")
    path = write_env_file(str(tmp_path), {"API_KEY": "fresh", "NEW": "2"})
    text = (tmp_path / ".env").read_text()
    assert path.endswith(".env")
    assert "# keep me" in text and "OLD=1" in text
    assert "API_KEY=fresh" in text and "API_KEY=stale" not in text and "NEW=2" in text
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert write_env_file(str(tmp_path), {}) is None


def test_env_var_names_reach_the_prompt_and_the_file_reaches_the_workspace(solution, scripted):
    agent = scripted(writes_main)
    result = AdapterAgent("fake", "m", max_retries=0).adapt(
        solution, setting_for("local"), env_vars={"API_KEY": "secret-value", "REGION": "eu"},
    )
    assert result.success
    prompt = agent.prompts[0]
    assert "API_KEY, REGION" in prompt
    assert "secret-value" not in prompt          # names only, never values
    assert open(os.path.join(result.adapted_path, ".env")).read().count("API_KEY=secret-value") == 1


# =============================================================================
# Workspace copy
# =============================================================================

def test_adapted_copy_leaves_git_and_caches_behind(solution, scripted):
    scripted(writes_main)
    result = AdapterAgent("fake", "m", max_retries=0).adapt(solution, setting_for("local"))
    assert os.path.exists(os.path.join(result.adapted_path, "model.py"))
    assert not os.path.exists(os.path.join(result.adapted_path, ".git"))
    # the validator's import check may create a fresh __pycache__; the copied one must be gone
    assert not os.path.exists(os.path.join(result.adapted_path, "__pycache__", "model.cpython-312.pyc"))


# =============================================================================
# Names and run interface
# =============================================================================

def test_docker_defaults_are_templated_per_solution_and_reach_the_prompt(solution, scripted, monkeypatch):
    # Docker's structure check wants the files the instruction asks for
    def writes_docker_files(workspace, prompt):
        for name in ("main.py", "app.py", "Dockerfile", "requirements.txt"):
            with open(os.path.join(workspace, name), "w") as f:
                f.write("# generated\n")
        return ""  # no run_interface from the agent → config defaults apply
    agent = scripted(writes_docker_files)
    expected = deployment_context(solution.code_path)
    result = AdapterAgent("fake", "m", max_retries=0).adapt(solution, setting_for("docker"))
    assert result.success
    iface = result.run_interface
    assert iface["container_name"] == f"kapso-{expected['deployment_name']}"
    assert iface["image_name"] == f"kapso-{expected['deployment_name']}"
    assert iface["port"] == expected["port"] and isinstance(iface["port"], int)
    assert iface["endpoint"] == f"http://localhost:{expected['port']}"
    assert iface["predict_path"] == "/predict"
    prompt = agent.prompts[0]
    assert f"kapso-{expected['deployment_name']}" in prompt and f"-p {expected['port']}:8000" in prompt
    assert "{deployment_name}" not in prompt and "{port}" not in prompt and "{env_file_flag}" not in prompt


def test_agent_run_interface_wins_and_path_is_normalised():
    adapter = AdapterAgent("fake", "m")
    iface = adapter._build_run_interface(
        "docker", endpoint="http://localhost:8123",
        agent_run_interface={"type": "http", "path": "/infer", "container_name": "kapso-x"},
        context={"deployment_name": "x", "port": 8123},
    )
    assert iface["predict_path"] == "/infer" and "path" not in iface
    assert iface["endpoint"] == "http://localhost:8123" and iface["container_name"] == "kapso-x"


# =============================================================================
# Validation and retries
# =============================================================================

def test_validation_failure_is_retried_with_the_finding_then_succeeds(solution, scripted):
    attempts = []

    def flaky(workspace, prompt):
        attempts.append(prompt)
        if len(attempts) == 1:
            return ""  # wrote nothing: main.py missing → structure check fails
        return writes_main(workspace, prompt)
    scripted(flaky)
    result = AdapterAgent("fake", "m", max_retries=1).adapt(solution, setting_for("local"))
    assert result.success
    assert len(attempts) == 2
    assert "previous attempt failed validation" in attempts[1]
    assert "main.py" in attempts[1]


def test_validation_failure_past_the_retries_fails_the_adaptation(solution, scripted):
    scripted(lambda workspace, prompt: "")
    result = AdapterAgent("fake", "m", max_retries=1).adapt(solution, setting_for("local"))
    assert not result.success
    assert "after 2 attempt(s)" in result.error and "main.py" in result.error


def test_validator_reads_required_files_from_config_and_gates_import_to_local(tmp_path):
    validator = AdaptationValidator()
    (tmp_path / "main.py").write_text("import torch_that_is_not_installed\n")
    # docker: syntax + structure only; the import would fail here and must not run
    for name in ("app.py", "Dockerfile", "requirements.txt"):
        (tmp_path / name).write_text("x = 1\n" if name.endswith(".py") else "")
    docker = validator.validate(str(tmp_path), setting_for("docker"))
    assert docker.success and any("skipped" in line for line in docker.logs)
    # local: the import runs and fails
    local = validator.validate(str(tmp_path), setting_for("local"))
    assert not local.success and "Import failed" in local.error
    # a strategy with nothing generated: every missing file is named
    empty = tmp_path / "empty"; empty.mkdir()
    missing = validator.validate(str(empty), setting_for("bentoml"))
    assert not missing.success and "service.py" in missing.error and "deploy.py" in missing.error


# =============================================================================
# Registry-driven lists
# =============================================================================

def test_every_shipped_strategy_declares_its_files_and_cloud_targets_their_tool():
    registry = StrategyRegistry.get()
    for name in registry.list_strategies():
        assert registry.get_strategy(name).get_required_files(), name
    assert registry.get_strategy("local").get_requires() is None
    assert registry.get_strategy("docker").get_requires()["binary"] == "docker"
    assert registry.get_strategy("langgraph").get_requires()["binary"] == "langgraph"


def test_preflight_and_cli_take_their_strategy_lists_from_the_registry():
    from kapso.core import preflight
    from kapso import cli
    tools = preflight.deploy_tools()
    assert set(tools) == {"docker", "modal", "bentoml", "langgraph"}
    assert cli.DEPLOY_STRATEGIES == ["auto"] + StrategyRegistry.get().list_strategies()
