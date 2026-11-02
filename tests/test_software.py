"""`Software` from the record (docs/plans/redesign/wp5-deploy.md, change
3): the three run kinds, the lifecycle through the target's scripts with
the deployment's values, the state shared through the record, `open`."""

import json
import os
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kapso.deployment.errors import DeployError
from kapso.deployment.handle import Software
from kapso.deployment.record import FAILED, RUNNING, STOPPED, open_record, start_record

PORTS = {"port_base": 48200, "port_span": 30}
MAIN = '''import os
import subprocess
from pathlib import Path

import helper

Path("imports.log").open("a").write("imported\\n")


def predict(inputs):
    if inputs.get("boom"):
        raise ValueError("boom")
    print("noise on stdout")
    subprocess.run(["echo", "a child's noise on descriptor 1"])
    return {"value": helper.VALUE, "greeting": os.environ["GREETING"], "echo": inputs}
'''
RUN_PY = '''import json
import os
import sys
import time

inputs = json.load(sys.stdin)
if inputs.get("fail"):
    sys.stderr.write("script failed\\n")
    sys.exit(3)
if inputs.get("hang"):
    time.sleep(30)
print(json.dumps({"echo": inputs, "schedule": os.environ.get("KAPSO_OPTION_SCHEDULE"), "app": os.environ.get("KAPSO_VALUE_APP")}))
'''


def deployed(tmp_path, *, run, name="campaign", scripts=None, values=None, endpoint=None, options=None, helper="first", call_timeout=5.0, script_timeout=5.0):
    campaign = tmp_path / name
    campaign.mkdir(exist_ok=True)
    record = start_record(
        campaign, target_requested="t", host="claude_code:m", interpreter=sys.executable, env_names=["GREETING"],
        options=options or {}, commit="abc", branch="generic_exp_0", chosen_attempt=None, harness_built=None, harness=None, harness_root=None, **PORTS,
    )
    record.workspace.mkdir()
    (record.workspace / "main.py").write_text(MAIN)
    (record.workspace / "helper.py").write_text(f"VALUE = {helper!r}\n")
    (record.workspace / ".env").write_text("GREETING=hello\n")
    (record.target_dir / "scripts").mkdir(parents=True)
    for script, text in (scripts or {}).items():
        (record.target_dir / "scripts" / script).write_text(text)
    record.update(
        target="t", endpoint=endpoint, values=values or {}, observed={"y": 2},
        result={"target": "t", "reasoning": None, "endpoint": endpoint, "run": run, "values": values or {},
                "proof": {"inputs": {"x": 1}, "output": {"y": 2}}, "artifacts": [], "summary": "ok"},
    )
    record.set_state(RUNNING)
    return Software(record, call_timeout=call_timeout, script_timeout=script_timeout)


# ----------------------------------------------------------------- function

def test_a_function_runs_in_a_worker_that_imports_once_and_sees_the_env_values(tmp_path):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    assert software.run({"a": 1}) == {"status": "success", "output": {"value": "first", "greeting": "hello", "echo": {"a": 1}}}
    assert software({"a": 2})["output"]["echo"] == {"a": 2}
    assert (software.workspace / "imports.log").read_text() == "imported\n"
    assert software.is_healthy() and software.proof == {"inputs": {"x": 1}, "output": {"y": 2}} and software.observed == {"y": 2}
    # The solution's own prints, and a child process's, never corrupt the
    # channel; they are its log.
    assert "noise on stdout" in software.logs() and "a child's noise on descriptor 1" in software.logs()


def test_the_env_file_is_read_verbatim_as_docker_reads_it(tmp_path):
    """A quoted value keeps its quotes, `${NAME}` is not expanded, a `#`
    inside a value is the value's: what `--env-file` gives a container,
    the worker gives the function."""
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    (software.workspace / ".env").write_text('# a comment\n\nGREETING="hi ${USER} # not a comment"\n')
    assert software.run({})["output"]["greeting"] == '"hi ${USER} # not a comment"'


def test_a_raising_function_ends_its_worker_and_the_next_call_starts_one(tmp_path):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    software.run({})
    failed = software.run({"boom": True})
    assert failed["status"] == "error" and "ValueError: boom" in failed["error"]
    assert software.run({"after": True})["status"] == "success"
    assert (software.workspace / "imports.log").read_text() == "imported\nimported\n"


def test_two_function_deployments_keep_their_own_modules(tmp_path):
    first = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"}, name="first", helper="one")
    second = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"}, name="second", helper="two")
    assert first.run({})["output"]["value"] == "one" and second.run({})["output"]["value"] == "two"
    assert first.run({})["output"]["value"] == "one"


def test_stop_and_start_on_a_function_deployment_say_the_target_has_no_scripts(tmp_path):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    software.run({})
    assert software.stop() == ["the t target has nothing to stop", "stopped deployment 1"]
    assert software.state == STOPPED and software._worker is None and not software.is_healthy()
    with pytest.raises(DeployError, match="deployment 1 is stopped; start\\(\\) it"):
        software.run({})
    assert software.start() == ["the t target has nothing to start", "started deployment 1"]
    assert software.state == RUNNING and software.run({})["status"] == "success"
    with software:
        pass
    assert software.state == STOPPED
    # The engine's lines are kept beside the record, for any later handle.
    assert "the t target has nothing to stop" in Software.open(software.campaign).logs()


def test_close_ends_the_handles_worker_and_leaves_the_deployment_running(tmp_path):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    assert software.is_healthy() and software._worker is not None
    software.close()
    assert software._worker is None and software.state == RUNNING
    assert Software.open(software.campaign).run({})["status"] == "success"


def test_a_function_that_hangs_is_an_error_naming_the_timeout(tmp_path):
    (tmp_path / "campaign").mkdir()
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"}, call_timeout=1.0)
    (software.workspace / "main.py").write_text("import time\n\ndef predict(inputs):\n    time.sleep(30)\n")
    failed = software.run({})
    assert failed == {"status": "error", "error": "the function did not answer within 1 s; its worker was ended"}


def test_a_function_whose_import_fails_says_so(tmp_path):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    (software.workspace / "main.py").write_text("import nothing_of_the_kind\n")
    failed = software.run({})
    assert failed["status"] == "error" and "No module named 'nothing_of_the_kind'" in failed["error"]
    assert not software.is_healthy()
    with pytest.raises(DeployError, match="the function's worker did not start"):
        software.start()
    (software.workspace / "main.py").write_text("predict = 3\n")
    assert "main.py has no module-level callable predict" in software.run({})["error"]


def test_a_worker_killed_from_outside_is_an_error_not_a_broken_pipe(tmp_path):
    """The health check pings the worker each time, so a worker that died
    (killed, crashed) is found before the next call writes to it."""
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    software.run({})
    os.kill(software._worker.pid, 9)
    software._worker.wait()
    assert software.is_healthy()   # a worker started afresh answers the ping
    os.kill(software._worker.pid, 9)
    software._worker.wait()
    assert software.run({"again": True})["status"] == "success"


# --------------------------------------------------------------------- http

class _Echo(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("fail"):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"path": self.path, "echo": body}).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def echo_server():
    server = HTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_an_http_deployment_posts_the_inputs_under_the_urls_prefix_and_reports_a_bad_status(tmp_path, echo_server):
    software = deployed(tmp_path, run={"kind": "http", "url": f"{echo_server}/v1/", "path": "/predict"}, endpoint=echo_server)
    assert software.run({"a": 1}) == {"status": "success", "output": {"path": "/v1/predict", "echo": {"a": 1}}}
    failed = software.run({"fail": True})
    assert failed == {"status": "error", "error": f"500 from {echo_server}/v1//predict: boom"}
    assert software.is_healthy() and software.endpoint == echo_server


def test_an_http_deployment_that_is_gone_answers_with_the_connection_error(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    software = deployed(tmp_path, run={"kind": "http", "url": f"http://127.0.0.1:{port}", "path": "/predict"})
    failed = software.run({"a": 1})
    assert failed["status"] == "error" and failed["error"].startswith(f"POST http://127.0.0.1:{port}/predict: ConnectionRefusedError")


def test_a_start_scripts_new_endpoint_is_where_the_http_calls_go(tmp_path, echo_server):
    software = deployed(
        tmp_path, run={"kind": "http", "url": "http://127.0.0.1:1", "path": "/predict"}, endpoint="http://127.0.0.1:1",
        scripts={"start.sh": f"echo KAPSO_ENDPOINT={echo_server}\n"},
    )
    assert software.run({})["status"] == "error"
    software.start()
    assert software.endpoint == echo_server and software.run({"b": 2}) == {"status": "success", "output": {"path": "/predict", "echo": {"b": 2}}}
    assert open_record(software.campaign).data["result"]["run"]["url"] == echo_server


# ------------------------------------------------------------------- script

def test_a_script_deployment_runs_the_targets_run_py_with_the_values_and_options(tmp_path):
    software = deployed(
        tmp_path, run={"kind": "script"}, scripts={"run.py": RUN_PY}, values={"app": "kapso-x"},
        options={"schedule": "*/5 * * * *"}, call_timeout=1.0,
    )
    assert software.run({"a": 1}) == {"status": "success", "output": {"echo": {"a": 1}, "schedule": "*/5 * * * *", "app": "kapso-x"}}
    assert software.run({"fail": True}) == {"status": "error", "error": "script failed"}
    assert software.run({"hang": True}) == {"status": "error", "error": "run.py did not answer within 1 s and was killed"}


def test_a_target_without_run_py_refuses_the_call(tmp_path):
    software = deployed(tmp_path, run={"kind": "script"})
    with pytest.raises(DeployError, match="the t target has no scripts/run.py; see logs\\(\\)"):
        software.run({})


# ---------------------------------------------------------------- lifecycle

LIFECYCLE = {
    "start.sh": "env | grep '^KAPSO_' | sort > seen.env\necho started the thing\necho KAPSO_ENDPOINT=http://new:1\necho KAPSO_VALUE_IMAGE=img:2\n",
    "stop.sh": "echo stopping >&2\nexit 1\n",
    "health.sh": "test -f healthy.flag\n",
    "logs.sh": "echo the logs\n",
}


def test_the_lifecycle_runs_the_scripts_with_the_values_and_takes_their_updates(tmp_path):
    software = deployed(
        tmp_path, run={"kind": "http", "url": "http://x", "path": "/p"}, scripts=LIFECYCLE,
        values={"app": "kapso-x"}, options={"schedule": "daily"}, endpoint="http://old:1",
    )
    assert software.state == RUNNING and not software.is_healthy()      # no healthy.flag yet
    (software.workspace / "healthy.flag").write_text("")
    assert software.is_healthy()
    assert software.start() == ["start.sh: started the thing", "started deployment 1"]
    seen = dict(line.split("=", 1) for line in (software.workspace / "seen.env").read_text().splitlines())
    record = software._record
    assert seen["KAPSO_DEPLOYMENT_NAME"] == record.data["deployment_name"] and seen["KAPSO_PORT"] == str(record.data["port"])
    assert seen["KAPSO_NUMBER"] == "1" and seen["KAPSO_WORKSPACE"] == str(software.workspace)
    assert seen["KAPSO_RECORD_DIR"] == str(record.directory) and seen["KAPSO_TARGET_DIR"] == str(record.target_dir)
    assert seen["KAPSO_INTERPRETER"] == sys.executable and seen["KAPSO_ENV_FILE"] == str(software.workspace / ".env")
    assert seen["KAPSO_OPTION_SCHEDULE"] == "daily" and seen["KAPSO_VALUE_APP"] == "kapso-x" and seen["KAPSO_ENDPOINT"] == "http://old:1"
    # The start script's update lines wrote back into the record.
    assert software.endpoint == "http://new:1" and software.values == {"app": "kapso-x", "image": "img:2"}
    assert json.loads(record.path.read_text())["endpoint"] == "http://new:1"
    assert software.stop() == ["stop.sh exited 1: stopping", "stopped deployment 1"]
    assert software.state == STOPPED and "stop.sh exited 1: stopping" in software.logs() and "the logs" in software.logs()
    # Another handle on the same record sees the stop.
    other = Software.open(software.campaign)
    assert other.state == STOPPED and not other.is_healthy()
    (software.target_dir / "scripts" / "start.sh").write_text("echo cannot >&2\nexit 2\n")
    with pytest.raises(DeployError, match="start.sh exited 2: cannot"):
        software.start()


def test_two_handles_on_one_deployment_change_only_what_each_means_to(tmp_path):
    """A stop from a handle opened before another's start wrote a new
    endpoint keeps that endpoint: every change re-reads the record."""
    software = deployed(tmp_path, run={"kind": "http", "url": "http://x", "path": "/p"}, scripts={"start.sh": "echo KAPSO_ENDPOINT=http://new:1\n"}, endpoint="http://old:1")
    stale = Software.open(software.campaign)
    software.start()
    stale.stop()
    record = open_record(software.campaign)
    assert record.state == STOPPED and record.data["endpoint"] == "http://new:1"


def test_a_script_killed_at_its_timeout_takes_its_children_with_it(tmp_path):
    """A lifecycle script is its own process group: the engine ends what
    it started (a build, a runner), not the shell alone."""
    software = deployed(
        tmp_path, run={"kind": "script"}, script_timeout=1.0,
        scripts={"start.sh": "sleep 30 &\necho $! > child.pid\nwait\n"},
    )
    with pytest.raises(DeployError, match="start.sh exited 1: start.sh did not end within 1 s and was killed, with what it started"):
        software.start()
    child = int((software.workspace / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try_alive = os.path.exists(f"/proc/{child}") and "zombie" not in Path(f"/proc/{child}/status").read_text()
        if not try_alive:
            break
        time.sleep(0.1)
    assert not try_alive


def test_a_failed_deployment_with_no_target_opens_and_says_there_is_nothing_to_stop(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    record = start_record(
        campaign, target_requested="auto", host="claude_code:m", interpreter=sys.executable, env_names=[],
        options={}, commit="abc", branch=None, chosen_attempt=None, harness_built=None, harness=None, harness_root=None, **PORTS,
    )
    record.workspace.mkdir()
    record.set_state(FAILED, error="the session was cut at its deadline")
    software = Software.open(campaign)
    assert software.target is None and software.run_kind is None and software.proof is None
    with pytest.raises(DeployError, match="deployment 1 is failed; the session was cut at its deadline"):
        software.run({})
    noted = software.stop()
    assert noted[0].startswith("deployment 1 recorded no target (its session chose none the engine could read): nothing to stop")
    assert software.state == STOPPED and open_record(campaign).data["error"] == "the session was cut at its deadline"


def test_open_takes_the_highest_number_or_the_named_one_and_refuses_what_is_gone(tmp_path):
    first = deployed(tmp_path, run={"kind": "script"})
    second = deployed(tmp_path, run={"kind": "script"})
    assert Software.open(first.campaign).number == 2 and Software.open(first.campaign, 1).number == 1
    assert repr(Software.open(first.campaign, 1)) == f"Software(deployment 1 of {first.campaign}, t, running)"
    with pytest.raises(DeployError, match="has no deployment 3"):
        Software.open(first.campaign, 3)
    shutil.rmtree(second._record.target_dir)
    with pytest.raises(DeployError, match="deployment 2 of .*: its target directory is gone"):
        Software.open(first.campaign)
    with pytest.raises(DeployError, match=r"has no deployment \(looked at"):
        Software.open(tmp_path / "nowhere")
