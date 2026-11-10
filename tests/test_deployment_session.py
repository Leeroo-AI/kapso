"""The deployment session (docs/plans/redesign/wp5-deploy.md, change 5),
with the session's agent faked and everything else real: the workspace
as its own repository, the env file, the staging with the target skills,
the prompt, the record before and after, the engine's acceptance with
its one resume, the failures and what a failed deployment still lets a
person do. Then an attempt a person chose: deployed from its own commit,
with the campaign left where it was."""

import json
import os
import shutil
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import kapso.deployment.session as session_module
import kapso.kapso as kapso_module
from kapso.core.config import load_config
from kapso.deployment.errors import DeployError
from kapso.deployment.handle import Software
from kapso.deployment.record import DEPLOYING, FAILED, RUNNING, STOPPED, open_record, records
from kapso.deployment.session import place_commit, run_deployment
from kapso.execution.coding_agents.base import CodingResult
from kapso.handles import Solution
from kapso.harness import HarnessError, load_harness
from tests.handles_support import candidate_branch, deployable, facade, git, git_workspace, make_solution, recorded_campaign
from tests.test_target_scripts import RUN_PAPER

OBJECTIVE = "Raise the churn model's AUC above 0.85.\n\nRules: never touch dq/; keep retention above 0.98."
# The entry point answers with the record's state at the time of the
# call: what the acceptance's own proof call sees.
MAIN = '''import json
import os

def predict(inputs):
    if inputs.get('boom'):
        raise ValueError('boom')
    state = json.load(open(os.environ['KAPSO_RECORD_DIR'] + '/deployment.json'))['state']
    return {'ok': inputs, 'state': state}
'''


def result_of(target="local", **overrides):
    data = {
        "target": target, "reasoning": None, "endpoint": None,
        "run": {"kind": "function", "module": "main", "callable": "predict"}, "values": {},
        "proof": {"inputs": {"a": 1}, "output": {"ok": {"a": 1}, "state": "deploying"}}, "artifacts": ["main.py"],
        "summary": "main.predict answers in the worker",
    }
    data.update(overrides)
    return data


class FakeAgent:
    """The session's agent: writes what the test scripts, records what it was told."""
    calls = []
    on_generate = staticmethod(lambda workspace: None)
    on_resume = staticmethod(lambda workspace: None)
    generate_result = None

    def __init__(self, config):
        self.config = config

    def initialize(self, workspace):
        self.workspace = Path(workspace)

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        FakeAgent.calls.append({"kind": "fresh", "prompt": prompt, "config": self.config, "workspace": self.workspace})
        FakeAgent.on_generate(self.workspace)
        return FakeAgent.generate_result or CodingResult(success=True, output="done", metadata={"cli_session_id": "s1"})

    def resume(self, cli_session_id, follow_up, timeout_seconds=None, reason=None):
        FakeAgent.calls.append({"kind": "resume", "prompt": follow_up, "session": cli_session_id, "timeout": timeout_seconds, "reason": reason})
        FakeAgent.on_resume(self.workspace)
        return CodingResult(success=True, output="done", metadata={"cli_session_id": cli_session_id})

    def get_cumulative_cost(self):
        return 0.0

    def cleanup(self):
        pass


def writes(main=MAIN, text=None, **overrides):
    def write(workspace):
        if main is not None:
            (workspace / "main.py").write_text(main)
        (workspace / "kapso_deployment.json").write_text(text if text is not None else json.dumps(result_of(**overrides)))
    return write


@pytest.fixture
def agent(monkeypatch):
    FakeAgent.calls = []
    FakeAgent.on_generate = staticmethod(writes())
    FakeAgent.on_resume = staticmethod(lambda workspace: None)
    FakeAgent.generate_result = None
    monkeypatch.setattr(session_module.CodingAgentFactory, "create", staticmethod(lambda config: FakeAgent(config)))
    bundled = {"local": "present, needs nothing", "docker": "present, but `docker info` fails: start Docker Desktop, or: sudo systemctl start docker"}
    monkeypatch.setattr(
        session_module, "readiness",
        lambda target: bundled.get(target.name, f"missing: install {target.name}") if target.origin == "bundled" else "unverified",
    )
    return FakeAgent


def config_with(**deployment):
    config = load_config()
    config["deployment"] = {
        "target": "auto", "cli": None, "model": None, "effort": None, "timeout": 600,
        "call_timeout": 5, "script_timeout": 5, "port_base": 48300, "port_span": 20, **deployment,
    }
    return config


@pytest.fixture
def solution(tmp_path):
    """A campaign checked out at its delivered candidate, with untracked
    data, a tracked .env, and the engine's and a host's directories."""
    campaign = tmp_path / "churn-build"
    git_workspace(campaign)
    (campaign / ".env").write_text("OTHER=1\nAPI_KEY=old\n")
    git(campaign, "add", ".env")
    git(campaign, "commit", "-q", "-m", "env")
    commit = candidate_branch(campaign, "generic_exp_0", "predict.py", "print('model')\n")
    git(campaign, "checkout", "-q", "generic_exp_0")
    (campaign / "kapso_datasets").mkdir()
    (campaign / "kapso_datasets" / "train.csv").write_text("id,y\n1,0\n")
    (campaign / ".kapso").mkdir()
    (campaign / ".kapso" / "status.json").write_text("{}")
    (campaign / ".claude" / "skills" / "x").mkdir(parents=True)
    (campaign / "kapso_task.md").write_text("task")
    return make_solution(campaign, objective=OBJECTIVE, commit=commit, branch="generic_exp_0", score=0.85)


def deploy(solution, agent, **kwargs):
    arguments = dict(config=config_with(), target="local", env=None, options=None, coding_agent=None, harness=None)
    arguments.update(kwargs)
    return run_deployment(solution, **arguments)


def test_a_named_target_deploys_and_the_record_runs(solution, agent):
    software = deploy(solution, agent, env={"API_KEY": "secret"}, options={"schedule": "daily"})
    campaign = solution.campaign
    record = open_record(campaign)
    assert record.state == RUNNING and record.data["target"] == "local" and record.data["target_requested"] == "local"
    assert record.data["result"]["summary"] == "main.predict answers in the worker" and record.data["finished"]
    assert record.data["commit"] == solution.commit and record.data["branch"] == "generic_exp_0"
    assert record.data["host"] == "claude_code:claude-opus-5" and record.data["env_names"] == ["API_KEY"] and record.data["options"] == {"schedule": "daily"}
    assert record.data["offered"] == [{"name": "local", "ready": "present, needs nothing"}]
    # The engine's own proof call ran while the record was still
    # `deploying`: nothing else could call the deployment meanwhile.
    assert record.data["observed"] == {"ok": {"a": 1}, "state": "deploying"}
    # The workspace: the solution's tree with its untracked data, without
    # the engine's and the host's directories, its own repository.
    workspace = record.workspace
    assert (workspace / "predict.py").is_file() and (workspace / "kapso_datasets" / "train.csv").is_file()
    assert not (workspace / ".kapso").exists() and not (workspace / ".claude" / "skills" / "x").exists() and not (workspace / "kapso_task.md").exists()
    assert git(workspace, "log", "--oneline").count("\n") == 0 and git(workspace, "log", "-1", "--format=%s") == f"deployment 1 of churn-build at {solution.commit[:12]}"
    assert git(campaign, "status", "--porcelain", "--untracked-files=no") == ""   # the campaign untouched
    # The env file: the campaign's other lines kept, the call's keys winning, owner-only.
    env_file = workspace / ".env"
    assert env_file.read_text() == "OTHER=1\nAPI_KEY=secret\n" and stat.S_IMODE(env_file.stat().st_mode) == 0o600
    # The target skill staged into the session and copied beside the record.
    assert (workspace / ".claude" / "skills" / "local" / "scripts" / "deploy.sh").is_file()
    assert (record.target_dir / "scripts" / "deploy.sh").is_file()
    session = json.loads((campaign / ".kapso" / "sessions" / "deployment" / "1" / "session.json").read_text())
    assert session["engine_skills"] == ["local"] and session["harness"] is None
    # The session's prompt and configuration.
    [call] = agent.calls
    assert OBJECTIVE in call["prompt"] and str(workspace / "kapso_deployment.json") in call["prompt"]
    assert "The target is `local`. Open its skill with `/local`" in call["prompt"]
    assert "API_KEY" in call["prompt"] and "secret" not in call["prompt"] and "schedule = daily" in call["prompt"]
    env = call["config"].agent_specific["env_overrides"]
    assert env["KAPSO_DEPLOYMENT_NAME"] == record.data["deployment_name"] and env["KAPSO_PORT"] == str(record.data["port"])
    assert env["KAPSO_TARGET_DIR"] == str(workspace / ".claude" / "skills" / "local") and env["KAPSO_ENV_FILE"] == str(env_file)
    assert env["KAPSO_OPTION_SCHEDULE"] == "daily" and call["config"].agent_specific["timeout"] == 600
    # A campaign with no origin and no verification names neither.
    assert "KAPSO_SOURCE_REMOTE" not in env and "KAPSO_VERIFICATION_DIR" not in env
    assert "KAPSO_SOURCE_REMOTE" not in call["prompt"] and "KAPSO_VERIFICATION_DIR" not in call["prompt"]
    # The handle returned is the one that proved the deployment: its
    # worker is alive, and no other was left behind.
    assert software._worker is not None and software._worker.poll() is None
    assert software.run({"b": 2}) == {"status": "success", "output": {"ok": {"b": 2}, "state": "running"}} and software.proof["inputs"] == {"a": 1}


def test_a_deployment_is_told_where_the_code_came_from_and_what_verification_wrote(solution, agent, tmp_path):
    """The campaign is a clone of a working copy whose origin is the
    host: the session and the target's scripts get the host's URL, not
    the path in between, and the delivered attempt's verification
    directory."""
    campaign = solution.campaign
    working_copy = tmp_path / "working-copy"
    git_workspace(working_copy)
    git(working_copy, "remote", "add", "origin", "https://github.com/acme/churn.git")
    git(campaign, "remote", "add", "origin", str(working_copy))
    verification = campaign / ".kapso" / "verification" / "generic_exp_0"
    verification.mkdir(parents=True)
    (verification / "verdict.json").write_text('{"verdict": "valid", "summary": "holds", "findings": []}')
    software = deploy(solution, agent)
    [call] = agent.calls
    env = call["config"].agent_specific["env_overrides"]
    assert env["KAPSO_SOURCE_REMOTE"] == "https://github.com/acme/churn.git" and env["KAPSO_VERIFICATION_DIR"] == str(verification)
    assert "`KAPSO_SOURCE_REMOTE` = `https://github.com/acme/churn.git`" in call["prompt"]
    assert f"`KAPSO_VERIFICATION_DIR` = `{verification}`" in call["prompt"]
    scripts_see = software._environment()
    assert scripts_see["KAPSO_SOURCE_REMOTE"] == "https://github.com/acme/churn.git" and scripts_see["KAPSO_VERIFICATION_DIR"] == str(verification)
    software.close()
    assert software.logs().count("Traceback") == 0
    software.stop()


def test_the_solutions_own_env_file_reaches_the_session_and_the_runtime(solution, agent):
    """No `env=`, but the copied tree holds `.env`: the session is told,
    the scripts and the worker get `KAPSO_ENV_FILE`, the file is owner-only."""
    software = deploy(solution, agent)
    env_file = software.workspace / ".env"
    assert env_file.read_text() == "OTHER=1\nAPI_KEY=old\n" and stat.S_IMODE(env_file.stat().st_mode) == 0o600
    [call] = agent.calls
    assert "the solution's own `.env` is at the workspace root (`$KAPSO_ENV_FILE`)" in call["prompt"]
    assert call["config"].agent_specific["env_overrides"]["KAPSO_ENV_FILE"] == str(env_file)
    software.stop()


def test_under_auto_the_prompt_offers_the_targets_with_their_readiness(solution, agent):
    agent.on_generate = staticmethod(writes(reasoning="a plain Python model; local is present"))
    software = deploy(solution, agent, target=None)
    [call] = agent.calls
    prompt = call["prompt"]
    assert "Choose the target first." in prompt
    assert "- local — Runs the solution in a Python process on this machine" in prompt and "open with `/local` — present, needs nothing" in prompt
    assert "- docker — " in prompt and "— present, but `docker info` fails: start Docker Desktop" in prompt
    assert "- modal — " in prompt and "— missing: install modal" in prompt
    assert f"write `{software.workspace / 'kapso_deployment.json'}` at once with `target` and `reasoning` alone" in prompt
    assert "KAPSO_TARGET_DIR" not in call["config"].agent_specific["env_overrides"]
    record = open_record(solution.campaign)
    assert record.data["target_requested"] == "auto" and record.data["target"] == "local"
    assert record.data["reasoning"] == "a plain Python model; local is present"
    assert [(o["name"], o["ready"]) for o in record.data["offered"]] == [
        ("bentoml", "missing: install bentoml"), ("docker", "present, but `docker info` fails: start Docker Desktop, or: sudo systemctl start docker"),
        ("langgraph", "missing: install langgraph"), ("local", "present, needs nothing"), ("modal", "missing: install modal"),
    ]
    assert sorted(p.name for p in (record.workspace / ".claude" / "skills").iterdir()) == ["bentoml", "docker", "langgraph", "local", "modal"]
    software.stop()


def test_a_missing_reasoning_under_auto_comes_back_once(solution, agent):
    agent.on_resume = staticmethod(writes(reasoning="because"))
    software = deploy(solution, agent, target=None)
    assert [call["kind"] for call in agent.calls] == ["fresh", "resume"]
    assert agent.calls[1]["prompt"].startswith("The engine checked the deployment and found: kapso_deployment.json: `reasoning` must say why you chose local.")
    assert agent.calls[1]["session"] == "s1" and agent.calls[1]["reason"] == "deployment result" and 0 < agent.calls[1]["timeout"] <= 600
    assert open_record(solution.campaign).data["reasoning"] == "because"
    software.stop()


def test_a_result_that_is_not_json_comes_back_as_a_finding_then_passes(solution, agent):
    agent.on_generate = staticmethod(writes(text='{"target": "local", "run": {"kind": "function"'))
    agent.on_resume = staticmethod(writes())
    software = deploy(solution, agent)
    assert [call["kind"] for call in agent.calls] == ["fresh", "resume"]
    assert "kapso_deployment.json is not valid JSON (json.decoder.JSONDecodeError: " in agent.calls[1]["prompt"]
    assert "write it again as one JSON object" in agent.calls[1]["prompt"]
    assert open_record(solution.campaign).state == RUNNING
    software.stop()


@pytest.mark.parametrize("script, finding", [
    (writes(target="docker"), "kapso_deployment.json names target docker; the request was local"),
    (writes(artifacts=["main.py", "missing.py"]), "artifact missing.py is not in the workspace"),
    (writes(artifacts=["../predict.py"]), "artifact ../predict.py must be a path inside the workspace, relative to it"),
    (writes(run={"kind": "http", "url": "https://api.example", "path": "/x"}), "the local target's run kind is function; the result names http"),
    (writes(run={"kind": "function", "module": "json", "callable": "loads"}), "run.module json is not a module of the workspace (no json.py or json/__init__.py there)"),
    (writes(proof={"inputs": {"boom": True}, "output": {}}), "the proof call failed: "),
    (writes(main="def other(inputs):\n    return inputs\n"), "the deployment does not answer its health check"),
    (writes(run={"kind": "http"}), "kapso_deployment.json: `run.url` must be a non-empty string for the http kind"),
    (writes(text="not json at all"), "kapso_deployment.json is not valid JSON ("),
    (lambda workspace: None, "no kapso_deployment.json at the workspace root"),
])
def test_a_failing_check_comes_back_once_then_fails_the_deployment(solution, agent, script, finding):
    agent.on_generate = staticmethod(script)
    with pytest.raises(DeployError, match=r"deployment 1 of .* failed: after the resume: ") as error:
        deploy(solution, agent)
    assert finding in str(error.value) and "--stop --number 1" in str(error.value)
    assert [call["kind"] for call in agent.calls] == ["fresh", "resume"] and finding in agent.calls[1]["prompt"]
    record = open_record(solution.campaign)
    assert record.state == FAILED and finding in record.data["error"] and record.data["finished"]
    # The failed deployment is still a deployment: it opens, `--stop`
    # runs the target's scripts, and no worker of the acceptance is left.
    software = Software.open(solution.campaign, 1)
    assert software.target == "local" and software.stop()[-1] == "stopped deployment 1" and software.state == STOPPED
    assert not list(Path("/proc").glob("*/cmdline")) or all(
        "kapso.deployment.worker" not in path.read_bytes().decode(errors="replace") or str(record.workspace) not in path.read_bytes().decode(errors="replace")
        for path in Path("/proc").glob("*/cmdline") if path.is_file()
    )


def test_a_session_cut_at_its_deadline_is_failed_without_a_resume(solution, agent):
    agent.generate_result = CodingResult(
        success=False, output="", error="Deadline of 600s reached",
        metadata={"cli_session_id": "s1", "deadline_exceeded": True, "completed_before_kill": False},
    )
    with pytest.raises(DeployError, match="failed: the session was cut at its deadline"):
        deploy(solution, agent)
    assert [call["kind"] for call in agent.calls] == ["fresh"]
    record = open_record(solution.campaign)
    assert record.state == FAILED and record.data["target"] == "local" and (record.target_dir / "scripts" / "deploy.sh").is_file()


def test_a_session_that_fails_to_run_is_failed(solution, agent):
    agent.generate_result = CodingResult(success=False, output="", error="Quota exceeded", metadata={"cli_session_id": None})
    with pytest.raises(DeployError, match="failed: the session failed: Quota exceeded"):
        deploy(solution, agent)


def test_under_auto_a_failed_session_leaves_the_target_it_named_or_says_it_named_none(solution, agent):
    """A session cut after writing its choice: the target's scripts are
    beside the record for `--stop`. One that wrote nothing: the record says
    so, and `--stop` has nothing to run."""
    cut = CodingResult(success=False, output="", error="cut", metadata={"cli_session_id": "s1", "deadline_exceeded": True, "completed_before_kill": False})
    agent.generate_result = cut
    agent.on_generate = staticmethod(writes(main=None, text='{"target": "docker", "reasoning": "a service"}'))
    with pytest.raises(DeployError, match="--stop --number 1"):
        deploy(solution, agent, target=None)
    record = open_record(solution.campaign, 1)
    assert record.state == FAILED and record.data["target"] == "docker" and (record.target_dir / "scripts" / "stop.sh").is_file()
    agent.on_generate = staticmethod(lambda workspace: None)
    with pytest.raises(DeployError, match="its session named no target the engine could read, so there is nothing for `--stop` to run"):
        deploy(solution, agent, target=None)
    record = open_record(solution.campaign, 2)
    assert record.state == FAILED and record.data["target"] is None and not record.target_dir.exists()
    assert Software.open(solution.campaign, 2).stop()[0].startswith("deployment 2 recorded no target")


def test_a_dirty_campaign_and_bad_env_are_refused_before_any_record(solution, agent):
    with pytest.raises(DeployError, match="env: 'BAD NAME' is not a variable name"):
        deploy(solution, agent, env={"BAD NAME": "x"})
    with pytest.raises(DeployError, match="env: 'NAME\\\\n' is not a variable name"):
        deploy(solution, agent, env={"NAME\n": "x"})
    with pytest.raises(DeployError, match="env: the value of A holds a line break or a control character"):
        deploy(solution, agent, env={"A": "x\ny"})
    with pytest.raises(DeployError, match="env: the value of A holds a line break or a control character"):
        deploy(solution, agent, env={"A": "x\ry"})
    (solution.campaign / "predict.py").write_text("changed\n")
    with pytest.raises(DeployError, match=r"tracked files differ from the delivered commit \(predict.py\); restore the delivered tree \(`git -C .* checkout -- predict.py`\), or build again"):
        deploy(solution, agent)
    assert records(solution.campaign) == [] and agent.calls == []


def test_the_session_follows_the_hosts_own_layout_on_codex(solution, agent):
    software = deploy(solution, agent, coding_agent="codex")
    [call] = agent.calls
    assert call["config"].agent_type == "codex" and open_record(solution.campaign).data["host"] == "codex:gpt-5.6-sol"
    assert "Open its skill with `$local`" in call["prompt"]
    assert call["config"].agent_specific["env_overrides"]["KAPSO_TARGET_DIR"].endswith(".agents/skills/local")
    software.stop()


# ------------------------------------------------------------ with a harness

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"


def trading(tmp_path):
    copy = tmp_path / "harness" / "trading"
    shutil.copytree(TRADING, copy)
    return load_harness(str(copy), settings=dict(load_config()["harness"], store=str(tmp_path / "store")), say=lambda line: None)


def test_with_a_harness_the_session_carries_its_deployment_module_beside_the_target(solution, agent, tmp_path):
    harness = trading(tmp_path)
    software = deploy(solution, agent, harness=harness)
    [call] = agent.calls
    page = (harness.path / "instructions" / "deployment.md").read_text().strip()
    assert page in call["prompt"] and "- paper-trading — " in call["prompt"] and "- broker — " in call["prompt"]
    assert call["prompt"].index("The target is `local`") < call["prompt"].index(page)
    specific = call["config"].agent_specific
    assert set(specific["mcp_servers"]) >= {"market-data", "broker"} and specific["settings_path"].endswith(".claude/harness-hooks.json")
    assert specific["env_overrides"]["KAPSO_HARNESS_ROOT"] == str(harness.path)
    workspace = software.workspace
    assert sorted(p.name for p in (workspace / ".claude" / "skills").iterdir()) == ["local", "market-data", "paper-trading"]
    record = open_record(solution.campaign)
    assert record.data["harness"]["name"] == "trading" and record.data["harness_root"] == str(harness.path)
    session = json.loads((solution.campaign / ".kapso" / "sessions" / "deployment" / "1" / "session.json").read_text())
    assert session["skills"] == ["market-data", "paper-trading"] and session["engine_skills"] == ["local"]
    assert session["hooks"][0] == "protect-harness"
    software.stop()


def test_a_harness_changed_since_its_load_deploys_nothing(solution, agent, tmp_path):
    harness = trading(tmp_path)
    page = harness.path / "instructions" / "guardrails.md"
    page.write_text(page.read_text() + "\n- edited after the load\n")
    with pytest.raises(HarnessError, match="the harness changed since launch"):
        deploy(solution, agent, harness=harness)
    assert records(solution.campaign) == [] and agent.calls == []


def test_a_harness_target_deploys_through_its_skill_and_runs_one_cycle_on_demand(solution, agent, tmp_path):
    """The trading example's paper-trading target: the session (faked)
    writes the runner the skill asks for and runs the target's deploy
    script with the deployment's variables; the engine's acceptance runs
    the health check and the proof cycle; the handle runs a cycle on
    demand, reads the log and stops the runner."""

    def act(workspace):
        (workspace / "deploy").mkdir()
        (workspace / "deploy" / "run_paper.py").write_text(RUN_PAPER)
        env = {**os.environ, **agent.calls[-1]["config"].agent_specific["env_overrides"]}
        subprocess.run(["bash", f"{env['KAPSO_TARGET_DIR']}/scripts/deploy.sh"], cwd=workspace, env=env, check=True, capture_output=True)
        (workspace / "kapso_deployment.json").write_text(json.dumps(result_of(
            target="paper-trading", endpoint="deploy/paper.log", run={"kind": "script"},
            proof={"inputs": {}, "output": {"bar": "latest", "orders": [], "equity": 100000.0}},
            artifacts=["deploy/run_paper.py"], summary="the runner runs on the schedule",
        )))

    agent.on_generate = staticmethod(act)
    harness = trading(tmp_path)
    software = deploy(solution, agent, harness=harness, target="paper-trading", options={"schedule": "*/2 * * * *", "cash": "5000"})
    [call] = agent.calls
    env = call["config"].agent_specific["env_overrides"]
    assert env["KAPSO_TARGET_DIR"].endswith(".claude/skills/paper-trading") and env["KAPSO_OPTION_CASH"] == "5000"
    assert "The target is `paper-trading`. Open its skill with `/paper-trading`" in call["prompt"]
    record = open_record(solution.campaign)
    assert record.state == RUNNING and record.data["target"] == "paper-trading" and record.data["endpoint"] == "deploy/paper.log"
    # The engine's own cycle ran with the call's cash, which the session's
    # claimed proof did not: both are in the record, the engine's first.
    assert record.data["observed"] == {"bar": "latest", "orders": [], "equity": 5000.0}
    assert (record.target_dir / "scripts" / "run.py").is_file() and record.data["harness"]["name"] == "trading"
    assert record.data["offered"] == [{"name": "paper-trading", "ready": "unverified"}]
    session = json.loads((solution.campaign / ".kapso" / "sessions" / "deployment" / "1" / "session.json").read_text())
    assert session["skills"] == ["market-data", "paper-trading"] and session["engine_skills"] == []
    assert software.run({"at": "2024-01-05"}) == {"status": "success", "output": {"bar": "2024-01-05", "orders": [], "equity": 5000.0}}
    assert software.is_healthy() and '"bar": "latest"' in software.logs()
    assert software.stop()[0].startswith("stop.sh: the paper runner (pid ")
    assert not software.is_healthy() and not (software.workspace / "deploy" / "paper.pid").exists()


def test_deploy_runs_the_attempt_on_a_campaign_that_delivered_nothing_without_moving_it(tmp_path, agent):
    campaign, commit = deployable(tmp_path)
    head = git(campaign, "rev-parse", "HEAD")
    solution = Solution.open(campaign, branch="generic_exp_0")

    software = run_deployment(
        solution, config=config_with(), target="local", env=None, options=None, coding_agent=None, harness=None,
    )
    software.close()

    # The attempt's tree, not the checked-out one, with the campaign's
    # untracked data and without the directories no workspace carries.
    workspace = software.workspace
    assert (workspace / "model.py").read_text() == "improved = True\n"
    assert not (workspace / "old.py").exists()
    assert (workspace / "kapso_datasets" / "train.csv").is_file()
    assert not (workspace / ".claude" / "skills" / "from_the_attempt").exists() and not (workspace / ".kapso").exists()
    # The campaign is where it was: the head, the branch, the tracked files.
    assert git(campaign, "rev-parse", "HEAD") == head and git(campaign, "branch", "--show-current") == "main"
    assert git(campaign, "status", "--porcelain", "--untracked-files=no") == ""
    assert (campaign / "old.py").is_file() and not (campaign / "model.py").exists()

    record = json.loads(software.record.read_text(encoding="utf-8"))
    assert (record["commit"], record["branch"]) == (commit, "generic_exp_0")
    assert record["chosen_attempt"] == solution.chosen_attempt
    prompt = agent.calls[0]["prompt"]
    assert f"Chosen by the person, not delivered by the engine: branch `generic_exp_0` at commit `{commit}`, score 0.8454." in prompt
    assert "Delivered: branch" not in prompt
    # The engine's record still says nothing was delivered.
    assert Solution.open(campaign).commit is None


def test_a_dirty_campaign_does_not_leak_into_a_chosen_attempts_workspace(tmp_path, agent):
    """The attempt's tracked files come from its commit, so an edit in the
    campaign has nothing to do with what is deployed."""
    campaign, _commit = deployable(tmp_path)
    (campaign / "README.md").write_text("edited by hand\n")
    software = run_deployment(
        Solution.open(campaign, branch="generic_exp_0"), config=config_with(), target="local", env=None,
        options=None, coding_agent=None, harness=None,
    )
    software.close()
    assert (software.workspace / "README.md").read_text() == "# seed\n"
    assert (campaign / "README.md").read_text() == "edited by hand\n"


@pytest.mark.parametrize("attempt_has", ["a file", "a symlink", "nothing"])
def test_a_tracked_directory_the_attempt_replaced_leaves_nothing_behind(tmp_path, attempt_has):
    campaign = tmp_path / "campaign"
    git_workspace(campaign)
    (campaign / "model").mkdir()
    (campaign / "model" / "weights.py").write_text("old = True\n")
    git(campaign, "add", "model")
    git(campaign, "commit", "-q", "-m", "a package")
    git(campaign, "checkout", "-q", "-b", "attempt")
    git(campaign, "rm", "-rq", "model")
    if attempt_has == "a file":
        (campaign / "model").write_text("flat = True\n")
    if attempt_has == "a symlink":
        (campaign / "model").symlink_to("README.md")
    git(campaign, "add", "--all")
    git(campaign, "commit", "-q", "--allow-empty", "-m", "the attempt")
    commit = git(campaign, "rev-parse", "HEAD")
    git(campaign, "checkout", "-q", "-")
    workspace = tmp_path / "workspace"
    shutil.copytree(campaign, workspace, ignore=shutil.ignore_patterns(".git"), symlinks=True)
    place_commit(campaign, commit, workspace)
    model = workspace / "model"
    assert model.is_symlink() == (attempt_has == "a symlink")
    assert model.is_file() == (attempt_has != "nothing")
    assert model.exists() == (attempt_has != "nothing")


def test_an_untracked_symlink_is_not_written_through_for_a_chosen_attempt(tmp_path):
    campaign = tmp_path / "campaign"
    git_workspace(campaign)
    git(campaign, "checkout", "-q", "-b", "attempt")
    (campaign / "cfg.yaml").write_text("from_the_attempt: true\n")
    git(campaign, "add", "cfg.yaml")
    git(campaign, "commit", "-q", "-m", "the attempt")
    commit = git(campaign, "rev-parse", "HEAD")
    git(campaign, "checkout", "-q", "-")
    outside = tmp_path / "outside.yaml"
    outside.write_text("mine: true\n")
    workspace = tmp_path / "workspace"
    shutil.copytree(campaign, workspace, ignore=shutil.ignore_patterns(".git"), symlinks=True)
    (workspace / "cfg.yaml").symlink_to(outside)
    with pytest.raises(DeployError, match="cfg.yaml stands where commit"):
        place_commit(campaign, commit, workspace)
    assert outside.read_text() == "mine: true\n"


def test_the_engines_deliverable_deploys_with_no_chosen_attempt_on_its_record(solution, agent):
    software = run_deployment(
        solution, config=config_with(), target="local", env=None, options=None, coding_agent=None, harness=None,
    )
    software.close()
    assert json.loads(software.record.read_text(encoding="utf-8"))["chosen_attempt"] is None
    assert f"Delivered: branch `generic_exp_0` at commit `{solution.commit}`" in agent.calls[0]["prompt"]


def test_the_facade_does_not_ask_for_a_checkout_of_a_chosen_attempt_and_prints_its_verdict(tmp_path, monkeypatch, capsys):
    campaign, commit = deployable(tmp_path, "trading-void")
    seen = {}
    monkeypatch.setattr(kapso_module, "run_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(kapso_module, "run_deployment", lambda solution, **kwargs: seen.update(solution=solution) or "software")
    kapso = facade()
    monkeypatch.setattr(kapso, "_harness_for", lambda solution, verb: None)

    assert kapso.deploy(Solution.open(campaign, branch="generic_exp_0"), target="local") == "software"
    assert seen["solution"].commit == commit
    out = capsys.readouterr().out
    assert "chosen by you (the engine delivered nothing)" in out and "verification void: " in out


def test_a_chosen_attempt_whose_commit_left_the_repository_is_refused(tmp_path, monkeypatch):
    campaign, _commit = recorded_campaign(tmp_path, "churn-unjudged")
    solution = Solution.open(campaign, branch="generic_exp_0")
    gone = replace(solution, commit="f" * 40)
    with pytest.raises(DeployError, match=f"commit {'f' * 12} of the attempt on generic_exp_0 is not in {campaign}"):
        facade().deploy(gone)
