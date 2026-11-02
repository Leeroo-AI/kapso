"""`notify:`: a build's stop reaches the person's command and webhook as one
JSON event; a block that could not deliver is refused before the build; a
failed delivery is reported, never raised."""

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import kapso.execution.orchestrator as orchestrator_module
import kapso.kapso as kapso_module
from kapso.core.config import ConfigError, deep_merge, load_config
from kapso.core.roles import check_notify
from kapso.execution.notify import STOP_EVENTS, build_event, post_webhook, run_command
from kapso.execution.observability import BuildStatus
from tests.handles_support import facade, fake_orchestrator, git_workspace, solve_result

RECORDER = "import sys; open(sys.argv[1], 'w').write(sys.stdin.read())"
REQUEST = {
    "id": 1, "node": 3, "key": "env:OPENAI_API_KEY", "for": "re-rank with embeddings",
    "hit": "AuthenticationError", "tried": "no key in .env", "fix": "add OPENAI_API_KEY to .env",
    "next_steps": "embed and re-rank", "previous_reply": None, "requested_at": "2026-10-30T09:00:00+00:00",
}


@pytest.mark.parametrize("overrides, refusal", [
    ({"events": ["paused"], "command": ["ping"]}, r"notify\.events: \['paused'\]; the events are waiting_for_user"),
    ({"events": ["failed"]}, r"notify\.events needs notify\.command or notify\.webhook"),
    ({"command": "ping --all"}, r"notify\.command must be an argument list"),
    ({"command": ["./scripts/ping.sh"]}, r"\./scripts/ping\.sh is a relative path; name it absolute"),
    ({"webhook": "hooks.example.com/kapso"}, r"notify\.webhook must be an http or https URL"),
    ({"timeout_seconds": 0}, r"notify\.timeout_seconds must be a positive number"),
])
def test_a_block_that_could_not_deliver_is_refused(overrides, refusal):
    with pytest.raises(ConfigError, match=refusal):
        check_notify({**load_config()["notify"], **overrides})


def test_every_stop_reason_the_orchestrator_writes_announces_an_event():
    source = Path(orchestrator_module.__file__).read_text(encoding="utf-8")
    written = set(re.findall(r'stopped_reason = "([a-z_]+)"', source))
    assert written and written <= set(STOP_EVENTS)
    assert build_event({"state": "running"}) is None
    with pytest.raises(ValueError, match="no notify event for stopped_reason 'stopped'"):
        build_event({"state": "done", "stopped_reason": "stopped"})


def test_a_command_that_fails_hangs_or_is_gone_is_reported_not_raised():
    failing = [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]
    assert run_command(failing, "{}", 10) == f"{sys.executable} exited 3: boom"
    started = time.monotonic()
    assert run_command([sys.executable, "-c", "import time; time.sleep(30)"], "{}", 0.5).endswith("did not finish within 0.5 s")
    assert time.monotonic() - started < 10
    assert run_command(["/nonexistent/ping.sh"], "{}", 10) == "/nonexistent/ping.sh is not an executable on PATH"


class _Receiver(BaseHTTPRequestHandler):
    received = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Receiver.received.append((self.path, self.headers["Content-Type"], body))
        self.send_response(_Receiver.status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def receiver():
    _Receiver.received, _Receiver.status = [], 200
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_the_webhook_receives_the_event_and_a_failure_hides_the_path(receiver):
    assert post_webhook(f"{receiver}/hooks/T0K3N", '{"event": "failed"}', 10) is None
    assert _Receiver.received == [("/hooks/T0K3N", "application/json", {"event": "failed"})]
    _Receiver.status = 500
    assert post_webhook(f"{receiver}/hooks/T0K3N", "{}", 10) == f"POST {receiver}: HTTPError: HTTP Error 500: Internal Server Error"
    closed = post_webhook("http://127.0.0.1:9/hooks/T0K3N", "{}", 10)
    assert closed.startswith("POST http://127.0.0.1:9: URLError") and "T0K3N" not in closed


def notifying_build(tmp_path, monkeypatch, events):
    workspace = tmp_path / "campaign"
    git_workspace(workspace)
    fake_orchestrator(monkeypatch, workspace, solve_result())
    kapso = facade()
    received = tmp_path / "event.json"
    command = [sys.executable, "-c", RECORDER, str(received)]
    kapso._config = deep_merge(kapso._config, {"notify": {"events": events, "command": command}})
    return kapso, workspace, received


def born(workspace):
    return BuildStatus(workspace / ".kapso" / "status.json")


def test_a_build_that_pauses_sends_the_requests_and_the_reply_line(tmp_path, monkeypatch, capsys):
    kapso, workspace, received = notifying_build(tmp_path, monkeypatch, ["waiting_for_user"])

    def solve(self, *args, finish, **kwargs):
        finish(solve_result(stopped_reason="waiting_for_user"))
        status = born(workspace)
        status.update(best={"score": 0.81, "node": 2})
        status.done(stopped_reason="waiting_for_user", requests=[REQUEST])

    monkeypatch.setattr(kapso_module.OrchestratorAgent, "solve", solve)
    kapso.build("Raise AUC above 0.85", output=str(workspace))

    event = json.loads(received.read_text())
    campaign = str(workspace.resolve())
    assert (event["event"], event["campaign"], event["objective"]) == ("waiting_for_user", campaign, "Raise AUC above 0.85")
    assert event["best"] == {"score": 0.81, "node": 2}
    assert event["requests"] == [{**REQUEST, "reply": f'kapso inbox reply {campaign} 1 "…"'}]
    assert "notify: waiting_for_user sent to the command" in capsys.readouterr().out


@pytest.mark.parametrize("where", ["construction", "solve"])
def test_a_build_that_dies_sends_failed_and_its_own_error_propagates(tmp_path, monkeypatch, where):
    kapso, workspace, received = notifying_build(tmp_path, monkeypatch, ["failed"])

    def construction_dies(self, *args, **kwargs):
        born(workspace)
        raise RuntimeError("the setup session died")

    def solve_dies(self, *args, **kwargs):
        born(workspace).failed(RuntimeError("the setup session died"))
        raise RuntimeError("the setup session died")

    if where == "construction":
        monkeypatch.setattr(kapso_module.OrchestratorAgent, "__init__", construction_dies)
    else:
        monkeypatch.setattr(kapso_module.OrchestratorAgent, "solve", solve_dies)
    with pytest.raises(RuntimeError, match="the setup session died"):
        kapso.build("Raise AUC above 0.85", output=str(workspace))
    event = json.loads(received.read_text())
    assert (event["event"], event["error"]) == ("failed", "RuntimeError: the setup session died")


@pytest.mark.parametrize("events, raised", [(["completed"], RuntimeError), (["failed"], KeyboardInterrupt)])
def test_an_unselected_event_and_a_persons_ctrl_c_send_nothing(tmp_path, monkeypatch, events, raised):
    kapso, workspace, received = notifying_build(tmp_path, monkeypatch, events)

    def solve(self, *args, **kwargs):
        born(workspace).failed(raised("stopped"))
        raise raised("stopped")

    monkeypatch.setattr(kapso_module.OrchestratorAgent, "solve", solve)
    with pytest.raises(raised):
        kapso.build("Raise AUC above 0.85", output=str(workspace))
    assert not received.exists()


def test_a_resume_that_dies_before_its_status_is_born_does_not_resend_the_last_stop(tmp_path, monkeypatch):
    kapso, workspace, received = notifying_build(tmp_path, monkeypatch, ["waiting_for_user"])
    born(workspace).done(stopped_reason="waiting_for_user", requests=[REQUEST])
    path = workspace / ".kapso" / "status.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "started_at": "2026-10-29T09:00:00+00:00"}))

    def refused(self, *args, **kwargs):
        raise RuntimeError("the checkpoint is incompatible")

    monkeypatch.setattr(kapso_module.OrchestratorAgent, "__init__", refused)
    with pytest.raises(RuntimeError, match="incompatible"):
        kapso.build("Raise AUC above 0.85", output=str(workspace))
    assert not received.exists()
