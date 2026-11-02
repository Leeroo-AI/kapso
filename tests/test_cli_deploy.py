"""`kapso deploy` (docs/plans/redesign/wp5-deploy.md, "The verb"): the
flags to the call, the summary a person reads, the lifecycle through the
record, a bare key refused by name, flags that contradict each other."""

import argparse
import sys
from types import SimpleNamespace

import pytest

import kapso.cli.deploy as deploy_module
from kapso.cli.deploy import add_parser, cmd_deploy, lifecycle, pairs, summary
from kapso.deployment.errors import DeployError
from kapso.deployment.record import FAILED, RUNNING, STOPPED, open_record, start_record
from tests.test_software import PORTS, deployed


def parser():
    root = argparse.ArgumentParser()
    add_parser(root.add_subparsers(dest="command"))
    return root


def arguments(campaign, **overrides):
    fields = dict(
        campaign=str(campaign), target=None, env=None, option=None, coding_agent=None, config=None, harness=None,
        status=False, stop=False, start=False, logs=False, number=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_pairs_take_key_equals_value_and_refuse_a_bare_key():
    assert pairs(["A=1", "B=x=y"], "--env") == {"A": "1", "B": "x=y"}
    assert pairs(None, "--option") == {}
    with pytest.raises(DeployError, match="--env takes KEY=VALUE, got 'FOO'"):
        pairs(["FOO"], "--env")


def test_the_target_is_not_an_argparse_choice_and_the_lifecycle_flags_exclude_each_other():
    """A harness target is known once the harness loads; the refusal of an
    unknown name is the engine's one line, not argparse's."""
    args = parser().parse_args(["deploy", "campaign", "--target", "paper-trading", "--option", "schedule=daily", "--env", "A=1"])
    assert args.target == "paper-trading" and args.option == ["schedule=daily"] and args.env == ["A=1"]
    with pytest.raises(SystemExit):
        parser().parse_args(["deploy", "campaign", "--strategy", "local"])
    with pytest.raises(SystemExit):
        parser().parse_args(["deploy", "campaign", "--stop", "--start"])


def test_a_deploy_flag_beside_a_lifecycle_flag_and_a_number_without_one_are_refused(tmp_path):
    with pytest.raises(DeployError, match="--target goes with a deploy, not with --stop"):
        cmd_deploy(arguments(tmp_path, stop=True, target="docker"))
    with pytest.raises(DeployError, match="--coding-agent goes with a deploy, not with --status"):
        cmd_deploy(arguments(tmp_path, status=True, coding_agent="codex"))
    with pytest.raises(DeployError, match="--number goes with --status, --stop, --start or --logs; a deploy takes the next number"):
        cmd_deploy(arguments(tmp_path, number=2))


def test_the_flags_reach_the_call_and_the_summary_pins_the_number(tmp_path, monkeypatch, capsys):
    software = deployed(tmp_path, run={"kind": "function", "module": "main", "callable": "predict"})
    seen = {}

    class FakeKapso:
        def __init__(self, config=None):
            seen["config"] = config

        def load_harness(self, reference):
            seen["harness"] = reference

        def deploy(self, solution, **kwargs):
            seen["solution"] = solution
            seen.update(kwargs)
            return software

    monkeypatch.setattr(deploy_module, "Kapso", FakeKapso)
    monkeypatch.setattr(deploy_module.Solution, "open", staticmethod(lambda path: f"solution at {path}"))
    cmd_deploy(arguments(
        software.campaign, target="docker", env=["API_KEY=x"], option=["schedule=daily"], coding_agent="codex",
        config="kapso config.yaml", harness="./trading",
    ))
    assert seen == {
        "config": "kapso config.yaml", "harness": "./trading", "solution": f"solution at {software.campaign}",
        "target": "docker", "env": {"API_KEY": "x"}, "options": {"schedule": "daily"}, "coding_agent": "codex",
    }
    out = capsys.readouterr().out
    assert f"DEPLOYED: deployment 1 of {software.campaign}" in out
    assert "  Target:    t\n" in out and "  State:     running (healthy: its module imports in a worker)" in out
    assert "  Run:       function — main.predict in a worker process" in out
    assert '  Proof:     {"x": 1} -> {"y": 2}\n' in out
    assert f"  Interpreter: {sys.executable} (runs the function; a kapso upgrade replaces it" in out
    # The next actions are copy-pastable as printed: Python literals in the
    # Python line, the path quoted for the shell, the config carried.
    assert f"  from kapso import Software; Software.open('{software.campaign}', 1, 'kapso config.yaml').run({{'x': 1}})   # from Python" in out
    assert f"  kapso deploy {software.campaign} --stop --number 1 --config 'kapso config.yaml'" in out
    software.stop()


def test_the_summary_shows_the_engines_own_answer_when_it_differs_from_the_sessions(tmp_path):
    software = deployed(tmp_path, run={"kind": "script"}, scripts={"run.py": "import sys\nprint('{}')\n"})
    record = software._record
    record.update(observed={"y": 3}, target_requested="auto", reasoning="the machine has docker")
    lines = summary(software, str(software.campaign))
    assert "  Target:    t (chosen: the machine has docker)" in lines
    assert "  Run:       script — the target's scripts/run.py" in lines
    assert '  Proof:     {"x": 1} -> {"y": 3}   (the engine\'s own call; the session\'s answered {"y": 2})' in lines
    assert "  from kapso import Software; Software.open('%s', 1).run({'x': 1})   # from Python" % software.campaign in lines


def test_the_lifecycle_flags_act_through_the_record(tmp_path, capsys):
    first = deployed(tmp_path, run={"kind": "script"}, scripts={"logs.sh": "echo the logs of the first\n", "stop.sh": "echo the runner ended\n"})
    second = deployed(tmp_path, run={"kind": "script"})
    lifecycle(arguments(first.campaign, status=True))
    rows = capsys.readouterr().out.splitlines()
    assert rows[0].startswith("  1  running (healthy)  t  ") and rows[1].startswith("  2  running (healthy)  t  ")
    lifecycle(arguments(first.campaign, status=True, number=2))
    assert len(capsys.readouterr().out.splitlines()) == 1
    lifecycle(arguments(first.campaign, stop=True, number=1))
    assert capsys.readouterr().out == f"stopped deployment 1 of {first.campaign}\n  stop.sh: the runner ended\n"
    assert open_record(first.campaign, 1).state == STOPPED and open_record(first.campaign, 2).state == RUNNING
    lifecycle(arguments(first.campaign, logs=True, number=1))
    assert "the logs of the first" in capsys.readouterr().out
    lifecycle(arguments(first.campaign, start=True, number=1))
    assert capsys.readouterr().out == f"started deployment 1 of {first.campaign} (healthy)\n  the t target has nothing to start\n"
    assert open_record(first.campaign, 1).state == RUNNING
    # A status question before the first deploy is answered, not refused: the
    # gate's cold engineer read the error exit as a wrong call of the package.
    lifecycle(arguments(tmp_path / "nowhere", status=True))
    assert capsys.readouterr().out == f"{tmp_path / 'nowhere'} has no deployment yet\n"
    with pytest.raises(DeployError, match="has no deployment 3"):
        lifecycle(arguments(first.campaign, status=True, number=3))


def test_the_status_rows_say_what_a_person_needs_while_deploying_and_after_a_failure(tmp_path, capsys):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    fields = dict(host="claude_code:m", interpreter=sys.executable, env_names=[], options={}, commit="abc", branch=None,
                  harness_built=None, harness=None, harness_root=None, **PORTS)
    start_record(campaign, target_requested="docker", **fields)
    failed = start_record(campaign, target_requested="auto", **fields)
    failed.set_state(FAILED, error="the session was cut at its deadline")
    gone = deployed(tmp_path, run={"kind": "script"})
    import shutil
    shutil.rmtree(gone.workspace)
    lifecycle(arguments(campaign, status=True))
    rows = capsys.readouterr().out.splitlines()
    assert rows[0].startswith("  1  deploying  docker (requested)  no endpoint  ")
    assert rows[1].startswith("  2  failed (the session was cut at its deadline)  auto (requested)  no endpoint  ")
    assert rows[2].startswith("  3  running (its files are gone)  t  ")
