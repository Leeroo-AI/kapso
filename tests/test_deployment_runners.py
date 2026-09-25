"""Runner contracts, driven with fakes for Docker, Modal and HTTP.

Each runner is the piece of deploy() that survives the coding-agent session:
it connects to what the session created and owns stop()/start(). None of
them had a test; the Docker one recreated containers without the
environment or GPU the first deploy had, and removed a container's logs with
it. These pin the lifecycle behaviour without a daemon or a cloud account.
"""

import sys
import types

import pytest

from kapso.deployment.strategies.base import deployment_context
from kapso.deployment.strategies.docker.runner import DockerRunner
from kapso.deployment.strategies.local.runner import LocalRunner


# =============================================================================
# Naming
# =============================================================================

def test_deployment_context_is_stable_readable_and_distinct(tmp_path):
    a = tmp_path / "churn model"
    b = tmp_path / "other" / "churn model"
    a.mkdir(); b.mkdir(parents=True)
    first, again, other = deployment_context(str(a)), deployment_context(str(a)), deployment_context(str(b))
    assert first == again
    assert first["deployment_name"].startswith("churn-model-")
    assert first["deployment_name"] != other["deployment_name"]  # same basename, different path
    assert 8000 <= first["port"] < 9000


# =============================================================================
# Local
# =============================================================================

def test_local_runner_imports_predict_and_restarts(tmp_path):
    (tmp_path / "main.py").write_text(
        "import os\n"
        "def predict(inputs):\n"
        "    return {'echo': inputs, 'key': os.environ.get('KAPSO_TEST_KEY')}\n"
    )
    runner = LocalRunner(code_path=str(tmp_path), env_vars={"KAPSO_TEST_KEY": "set-by-kapso"})
    assert runner.is_healthy()
    assert runner.run({"a": 1}) == {"echo": {"a": 1}, "key": "set-by-kapso"}
    runner.stop()
    assert not runner.is_healthy()
    runner.start()
    assert runner.run({"b": 2})["echo"] == {"b": 2}


# =============================================================================
# Docker (fake SDK)
# =============================================================================

class FakeContainer:
    def __init__(self, name, status="running"):
        self.name, self.status = name, status
        self.stopped = self.removed = False

    def logs(self, tail=None):
        return b"line1\nline2\n"

    def stop(self, timeout=None):
        self.stopped, self.status = True, "exited"

    def remove(self):
        self.removed = True

    def start(self):
        self.status = "running"


class FakeContainers:
    def __init__(self):
        self.by_name = {}
        self.run_calls = []

    def get(self, name):
        if name not in self.by_name:
            raise KeyError(name)
        return self.by_name[name]

    def run(self, image, **kwargs):
        self.run_calls.append((image, kwargs))
        container = FakeContainer(kwargs["name"])
        self.by_name[kwargs["name"]] = container
        return container


class FakeClient:
    def __init__(self):
        self.containers = FakeContainers()


@pytest.fixture
def fake_docker(monkeypatch):
    client = FakeClient()
    module = types.ModuleType("docker")
    module.from_env = lambda: client
    types_module = types.ModuleType("docker.types")

    class DeviceRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_module.DeviceRequest = DeviceRequest
    module.types = types_module
    monkeypatch.setitem(sys.modules, "docker", module)
    monkeypatch.setitem(sys.modules, "docker.types", types_module)
    # No HTTP in these tests: readiness is answered by the fake immediately
    monkeypatch.setattr(DockerRunner, "wait_for_ready", lambda self, timeout=60, interval=2: True)
    return client


def test_docker_runner_recreates_the_container_with_env_ports_and_restart_policy(fake_docker, tmp_path):
    runner = DockerRunner(
        endpoint="http://localhost:8123", port="8123", code_path=str(tmp_path),
        container_name="kapso-demo", image_name="kapso-demo",
        env_vars={"API_KEY": "k"}, resources={},
    )
    runner.start()
    image, kwargs = fake_docker.containers.run_calls[0]
    assert image == "kapso-demo"
    assert kwargs["name"] == "kapso-demo"
    assert kwargs["ports"] == {"8000/tcp": 8123}
    assert kwargs["environment"] == {"API_KEY": "k"}
    assert kwargs["restart_policy"] == {"Name": "unless-stopped"}
    assert "device_requests" not in kwargs


def test_docker_runner_requests_gpus_when_the_selector_granted_one(fake_docker, tmp_path):
    runner = DockerRunner(code_path=str(tmp_path), resources={"gpu": "T4"})
    runner.start()
    _, kwargs = fake_docker.containers.run_calls[0]
    assert kwargs["device_requests"][0].kwargs == {"count": -1, "capabilities": [["gpu"]]}


def test_docker_runner_keeps_container_logs_after_removing_it(fake_docker, tmp_path):
    runner = DockerRunner(code_path=str(tmp_path), container_name="kapso-demo")
    runner.start()
    runner.stop()
    container = fake_docker.containers.run_calls[0][1]["name"]
    assert fake_docker.containers.by_name[container].removed
    assert "line2" in runner.get_logs()


def test_docker_runner_names_default_to_the_per_solution_scheme_and_accept_path(fake_docker, tmp_path):
    runner = DockerRunner(code_path=str(tmp_path), path="/infer")
    expected = f"kapso-{deployment_context(str(tmp_path))['deployment_name']}"
    assert runner.container_name == expected and runner.image_name == expected
    assert runner.predict_path == "/infer"


def test_docker_runner_without_the_sdk_says_how_to_install_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "docker", None)  # import raises ImportError
    runner = DockerRunner(code_path=str(tmp_path))
    assert 'leeroo-kapso[deploy]' in runner.get_logs()


# =============================================================================
# Modal (fake SDK)
# =============================================================================

@pytest.fixture
def fake_modal(monkeypatch, tmp_path):
    calls = {"from_name": [], "hydrated": 0, "deploys": []}

    class Handle:
        def __init__(self, app, fn):
            self.app, self.fn = app, fn

        def hydrate(self):
            calls["hydrated"] += 1
            if self.app not in deployed_apps:
                raise RuntimeError(f"app {self.app} not found")

        def remote(self, inputs):
            return {"echo": inputs}

    class Function:
        @staticmethod
        def from_name(app, fn):
            calls["from_name"].append((app, fn))
            return Handle(app, fn)

    deployed_apps = set()
    module = types.ModuleType("modal")
    module.Function = Function
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setenv("MODAL_TOKEN_ID", "x")

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["modal", "app", "stop"]:
            calls.setdefault("stops", []).append(cmd)
            deployed_apps.discard(cmd[3])
        else:
            calls["deploys"].append((cmd, kwargs.get("cwd")))
            deployed_apps.add("kapso-demo")
        return types.SimpleNamespace(returncode=0, stdout="deployed", stderr="")

    from kapso.deployment.strategies.modal import runner as modal_runner
    monkeypatch.setattr(modal_runner.subprocess, "run", fake_run)
    return calls, deployed_apps


def test_modal_runner_uses_from_name_and_can_redeploy_on_start(fake_modal, tmp_path):
    from kapso.deployment.strategies.modal.runner import ModalRunner
    calls, deployed_apps = fake_modal
    (tmp_path / "modal_app.py").write_text("app = None\n")
    runner = ModalRunner(app_name="kapso-demo", code_path=str(tmp_path))
    assert calls["from_name"] == [("kapso-demo", "predict")]
    assert not runner.is_healthy()            # hydrate() failed: not deployed yet
    runner.start()                            # deploys, then looks up again
    assert calls["deploys"][0][0] == ["modal", "deploy", "modal_app.py"]
    assert runner.is_healthy()
    assert runner.run({"x": 1}) == {"echo": {"x": 1}}
    # stop() must not wait on the CLI's confirmation prompt (a real run hung on it)
    runner.stop()
    assert calls["stops"] == [["modal", "app", "stop", "kapso-demo", "--yes"]]
    assert not runner.is_healthy()
    runner.start()                            # gone → deployed again → connected
    assert len(calls["deploys"]) == 2 and runner.is_healthy()


def test_modal_runner_accepts_the_token_file_as_authentication(monkeypatch, tmp_path):
    from kapso.deployment.strategies.modal.runner import ModalRunner
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert not ModalRunner._authenticated()
    (tmp_path / ".modal.toml").write_text("[default]\n")
    assert ModalRunner._authenticated()


# =============================================================================
# BentoCloud (fake CLI + HTTP)
# =============================================================================

def test_bentoml_runner_finds_the_endpoint_from_the_deployment_name(monkeypatch, tmp_path):
    """The session's final message may omit <endpoint_url>; the name Kapso
    assigned is enough to ask BentoCloud (a real run needed exactly this)."""
    from kapso.deployment.strategies.bentoml import runner as bento_runner
    from kapso.deployment.strategies.bentoml.runner import BentoMLRunner

    state = {"status": "running"}

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["bentoml", "deployment", "get"] and cmd[3] == "kapso-demo"
        return types.SimpleNamespace(
            returncode=0, stderr="",
            stdout=f"name: kapso-demo\nstatus:\n  status: {state['status']}\n"
                   "endpoint_urls:\n- https://kapso-demo-abc123.mt-guc1.bentoml.ai\n",
        )
    monkeypatch.setattr(bento_runner.subprocess, "run", fake_run)
    runner = BentoMLRunner(deployment_name="kapso-demo", code_path=str(tmp_path))
    assert runner._endpoint == "https://kapso-demo-abc123.mt-guc1.bentoml.ai"
    assert runner._deployed
    # a terminated deployment still describes its URL; it must not count as connected
    state["status"] = "terminated"
    gone = BentoMLRunner(deployment_name="kapso-demo", code_path=str(tmp_path))
    assert gone._endpoint is None and not gone._deployed
    assert "terminated" in gone.get_logs()


def test_software_reports_a_runner_error_dict_as_an_error():
    from kapso.deployment.base import DeployConfig
    from kapso.deployment.software import DeployedSoftware
    from kapso.deployment.base import DeploymentInfo
    from kapso.execution.solution import SolutionResult

    class ErrRunner:
        def run(self, inputs):
            return {"error": "not deployed", "instructions": ["deploy first"]}
        def stop(self): pass
        def start(self): pass
        def is_healthy(self): return False

    software = DeployedSoftware(
        config=DeployConfig(solution=SolutionResult(goal="g", code_path="/nowhere")),
        runner=ErrRunner(), info=DeploymentInfo(strategy="bentoml"),
    )
    result = software.run({})
    assert result["status"] == "error" and result["error"] == "not deployed"
    assert result["instructions"] == ["deploy first"]
