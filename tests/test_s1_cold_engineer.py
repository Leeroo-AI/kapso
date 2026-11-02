"""S1's driver (docs/plans/redesign/wp9-semantic-suite.md, change 6): the
task table and the suite agree on the steps; the docs snapshot is the
pages docs.json navigates to and nothing else; a task view is the
repository's own README; a session's init event proves the sandbox cold;
the counted verb's executions are read from a session's stream (help,
doctor and reads never count; an errored execution followed by a success
is an error that led to its fix); the evidence predicates read the
artifacts a task leaves; the two counts the design names are metrics; the
released baseline reads the docs the packer staged, or fails loud."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "redesign"))
spec = importlib.util.spec_from_file_location("s1_cold_engineer", ROOT / "scripts" / "redesign" / "s1_cold_engineer.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)


def test_the_task_table_and_the_suite_agree_on_the_steps():
    assert [task["number"] for task in s1.TASKS] == [1, 2, 3, 4, 5, 6]
    assert all(task["verb"] in s1.VERB_PATTERNS["branch"] and task["verb"] in s1.VERB_PATTERNS["released"] for task in s1.TASKS)
    assert s1.STEP_NAMES == s1.SUITE["tests"]["S1"]["required_steps"]["claude_code"]


def test_the_docs_snapshot_is_the_navigated_pages_and_nothing_else(tmp_path):
    pages = s1.navigated_pages(ROOT)
    assert "index" in pages and "installation" in pages and "coding-agent-skills" in pages
    assert not any(page.startswith(("plans/", "research/")) for page in pages)
    source = tmp_path / "docs"
    (source / "plans").mkdir(parents=True)
    (source / "index.mdx").write_text("# start\n")
    (source / "plans" / "secret.mdx").write_text("# a plan\n")
    count = s1.snapshot_docs(source, ["index", "missing"], tmp_path / "snapshot")
    assert count == 1 and (tmp_path / "snapshot" / "index.mdx").is_file() and not (tmp_path / "snapshot" / "plans").exists()


def test_a_task_view_is_the_repositorys_own_readme(tmp_path):
    s1.task_view(ROOT / "examples" / "churn-first-build", tmp_path / "task", generate=False)
    readme = (tmp_path / "task" / "README.md").read_text()
    assert readme.startswith("# Churn model") and "kapso build" not in readme and "Recorded result" not in readme
    assert (tmp_path / "task" / "kapso-brief.md").is_file() and (tmp_path / "task" / "repo" / "predict.py").is_file()
    assert not (tmp_path / "task" / "run.py").exists() and not (tmp_path / "task" / "baseline.json").exists()


def test_the_init_event_proves_the_sandbox_cold():
    warm = {"type": "system", "subtype": "init", "plugins": [{"source": "kapso@kapso"}], "mcp_servers": [{"name": "plugin:kapso:kapso"}], "slash_commands": ["kapso:build", "help"], "tools": ["Bash", "WebFetch"]}
    init = s1.session_init([warm])
    assert init == {"plugins": ["kapso@kapso"], "mcp_servers": ["plugin:kapso:kapso"], "kapso_commands": ["kapso:build"], "web_tools": ["WebFetch"]} and not s1.is_cold(init)
    cold = {"type": "system", "subtype": "init", "plugins": [{"source": "cc-plugin-telemetry@builtin"}], "mcp_servers": [], "slash_commands": ["help"], "tools": ["Bash", "Read"]}
    assert s1.is_cold(s1.session_init([cold])) is True
    assert s1.is_cold(s1.session_init([])) is None, "no init row (a Codex session): coldness is not provable from the stream, not a silent pass"


def _stream(calls):
    """A Claude Code stream of Bash calls, each (command, errored)."""
    rows = []
    for number, (command, errored) in enumerate(calls):
        rows.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"t{number}", "name": "Bash", "input": {"command": command}}]}})
        rows.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{number}", "is_error": errored, "content": "Error: x" if errored else "ok"}]}})
    return rows


def test_the_counted_verbs_executions_are_read_from_the_stream(tmp_path):
    rows = _stream([("kapso build --help", False), ("kapso doctor build", False), ("cat docs/index.mdx", False), ("kapso build --brief b.md --output campaign-1", True), ("kapso build --brief b.md --repo repo --output campaign-1 --detach", False)])
    found = s1.executions(rows, "build", "cli", tmp_path)
    assert [entry["errored"] for entry in found] == [True, False]
    metrics = s1.attempt_metrics(found, evidence=True, started=0.0, evidence_at=90.0)
    assert metrics == {"first_try": False, "wrong_calls": 1, "minutes_to_launch": 1.5, "error_led_to_fix": True, "executions": 2, "evidence": True}
    assert s1.attempt_metrics(found[1:], evidence=True, started=0.0, evidence_at=60.0)["first_try"] is True
    assert s1.attempt_metrics([], evidence=False, started=0.0, evidence_at=None)["first_try"] is False
    # The Python surface counts a script that carries the call, run with python.
    (tmp_path / "run_build.py").write_text("from kapso import Kapso, Brief\nsolution = Kapso().build(Brief.read('task/kapso-brief.md'), repo='task/repo', output='campaign-1')\nprint(solution.explain())\n")
    (tmp_path / "look.py").write_text("print('hello')\n")
    rows = _stream([("./venv/bin/python look.py", False), ("cat run_build.py", False), ("./venv/bin/python run_build.py", False)])
    found = s1.executions(rows, "build", "python", tmp_path)
    assert len(found) == 1 and found[0]["errored"] is False
    assert s1.executions(_stream([("kapso build --output c --resume --detach", False)]), "resume", "cli", tmp_path)
    assert not s1.executions(_stream([("kapso build --output c --resume --detach", False)]), "build", "cli", tmp_path)


def _codex_stream(calls):
    """A Codex stream of command_execution items, each (command, exit_code)."""
    return [{"type": "item.completed", "item": {"type": "command_execution", "command": command, "exit_code": code, "aggregated_output": "done" if code == 0 else "Error: boom"}} for command, code in calls]


def test_executions_read_the_codex_stream_too(tmp_path):
    rows = _codex_stream([("kapso build --help", 0), ("kapso build --brief b.md --output c1", 1), ("kapso build --brief b.md --repo repo --output c1 --detach", 0)])
    found = s1.executions(rows, "build", "cli", tmp_path)
    assert [entry["errored"] for entry in found] == [True, False], "the --help is not counted; the first run errored (exit 1), the second succeeded"
    assert s1.attempt_metrics(found, evidence=True, started=0.0, evidence_at=60.0)["first_try"] is False


def test_error_led_to_fix_is_one_step_only(tmp_path):
    # One wrong call then a success is a one-step fix; two is not.
    one = s1.executions(_stream([("kapso build --brief b --output c", True), ("kapso build --brief b --repo r --output c --detach", False)]), "build", "cli", tmp_path)
    assert s1.attempt_metrics(one, evidence=True, started=0.0, evidence_at=1.0)["error_led_to_fix"] is True
    two = s1.executions(_stream([("kapso build --output c", True), ("kapso build --brief b --output c", True), ("kapso build --brief b --repo r --output c --detach", False)]), "build", "cli", tmp_path)
    metrics = s1.attempt_metrics(two, evidence=True, started=0.0, evidence_at=1.0)
    assert metrics["wrong_calls"] == 2 and metrics["error_led_to_fix"] is False, "two wrong calls is not a one-step fix"


def test_the_evidence_predicates_read_what_a_task_leaves(tmp_path):
    sandbox = tmp_path
    prereq = sandbox / "campaign-prereq"
    assert not s1.evidence_of(s1.TASKS[0], sandbox, prereq)
    (sandbox / "campaign-1" / ".kapso").mkdir(parents=True)
    (sandbox / "campaign-1" / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "harness": None}))
    assert s1.evidence_of(s1.TASKS[0], sandbox, prereq)
    (sandbox / "campaign-3" / ".kapso").mkdir(parents=True)
    (sandbox / "campaign-3" / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "harness": None}))
    assert not s1.evidence_of(s1.TASKS[2], sandbox, prereq), "the harness task needs the harness in the status"
    (sandbox / "campaign-3" / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "harness": "trading 0.1.0"}))
    assert s1.evidence_of(s1.TASKS[2], sandbox, prereq)
    (prereq / ".kapso" / "deployments" / "1").mkdir(parents=True)
    assert not s1.evidence_of(s1.TASKS[3], sandbox, prereq)
    (prereq / ".kapso" / "deployments" / "1" / "deployment.json").write_text("{}")
    assert s1.evidence_of(s1.TASKS[3], sandbox, prereq)
    (sandbox / "task" / "learning" / "status").mkdir(parents=True)
    assert not s1.evidence_of(s1.TASKS[4], sandbox, prereq)
    (sandbox / "task" / "learning" / "status" / "learn-1.json").write_text("{}")
    assert s1.evidence_of(s1.TASKS[4], sandbox, prereq)
    (sandbox / "campaign-killed" / ".kapso").mkdir(parents=True)
    (sandbox / "campaign-killed" / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    assert s1.evidence_of(s1.TASKS[5], sandbox, prereq)
    (sandbox / "campaign-killed" / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "pid": 2 ** 22 - 1}))
    assert not s1.evidence_of(s1.TASKS[5], sandbox, prereq)
    (sandbox / "campaign-killed" / ".kapso" / "status.json").write_text(json.dumps({"state": "done", "pid": 2 ** 22 - 1}))
    assert s1.evidence_of(s1.TASKS[5], sandbox, prereq), "a resume that already finished (done) is still proof it continued"
    (sandbox / "campaign-killed" / ".kapso" / "status.json").write_text(json.dumps({"state": "stopped", "pid": 2 ** 22 - 1}))
    assert not s1.evidence_of(s1.TASKS[5], sandbox, prereq), "a campaign left at its kill was never resumed"


def test_grade_parses_with_only_out_while_a_run_requires_host_and_clone(tmp_path):
    dims = s1.SUITE["tests"]["S1"]["judge"]["dimensions"]
    passes = [{"items": {"comprehension": {d: {"score": 1, "evidence": []} for d in dims}}} for _ in range(2)]
    (tmp_path / "record.json").write_text(json.dumps({"steps": [], "metrics": {}, "judge": {"failure": None, "passes": passes, "adjudication": {}}}))
    assert s1.main(["--grade", "--out", str(tmp_path)]) == 0  # the runner grades the fetched record with --out alone
    assert json.loads((tmp_path / "record.json").read_text())["metrics"]["comprehension_correct"] == 1.0
    with pytest.raises(SystemExit):  # a real cold-engineer run still needs the host and the clone
        s1.main(["--out", str(tmp_path)])


def test_the_two_counts_and_the_run_metrics(tmp_path):
    (tmp_path / "a.py").write_text("import kapso\n\n# a comment\nx = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    assert s1.python_lines(tmp_path, ["a.py", "b.py", "missing.py"]) == 3
    record = {"steps": [
        {"name": "task1-python", "kind": "attempt", "metrics": {"first_try": True, "wrong_calls": 0, "error_led_to_fix": None}},
        {"name": "task1-cli", "kind": "attempt", "metrics": {"first_try": False, "wrong_calls": 2, "error_led_to_fix": True}},
        {"name": "task2-cli", "kind": "attempt", "metrics": {"first_try": False, "wrong_calls": 1, "error_led_to_fix": False}},
        {"name": "flow-counts", "kind": "shell", "metrics": {"python_lines": 9, "cli_commands": 5}},
    ], "metrics": {"comprehension_correct": 1.0}}
    assert s1.run_metrics(record) == {"first_try_count": 1, "errored_calls": 2, "errors_led_to_fix": 0.5, "wrong_calls_max": 2, "comprehension_correct": 1.0, "python_lines": 9, "cli_commands": 5}
    assert set(s1.SUITE["tests"]["S1"]["bars"]) >= {"wrong_calls_max", "comprehension_correct"}
    # No errored call: the one-step rule is not exercised (None), and the
    # suite's bar names errored_calls as what exercises it.
    clean = {"steps": [record["steps"][0], record["steps"][3]], "metrics": {}}
    assert s1.run_metrics(clean)["errors_led_to_fix"] is None and s1.run_metrics(clean)["errored_calls"] == 0
    assert s1.SUITE["tests"]["S1"]["bars"]["errors_led_to_fix"]["exercised_by"] == "errored_calls"


def test_grade_projects_the_comprehension_dimension_into_the_metric():
    dim = s1.COMPREHENSION_DIMENSION
    other = [d for d in s1.SUITE["tests"]["S1"]["judge"]["dimensions"] if d != dim]

    def block(score):
        passes = [{"items": {"comprehension": {dim: {"score": score, "evidence": []}, **{d: {"score": 1, "evidence": []} for d in other}}}} for _ in range(2)]
        return {"failure": None, "passes": passes, "adjudication": {}}

    assert s1.grade({"steps": [], "judge": block(1), "metrics": {}})["metrics"]["comprehension_correct"] == 1.0
    assert s1.grade({"steps": [], "judge": block(0), "metrics": {}})["metrics"]["comprehension_correct"] == 0.0
    assert s1.grade({"steps": [], "judge": {"failure": "x", "passes": []}, "metrics": {}})["metrics"] == {}


def test_flow_counts_are_per_task_so_a_reused_filename_is_not_conflated(tmp_path):
    (tmp_path / "solve.py").write_text("a = 1\nb = 2\nc = 3\n")
    record = {"steps": [
        {"name": "task1-python", "kind": "attempt", "metrics": {"scripts": ["solve.py"], "commands": []}},
        {"name": "task2-python", "kind": "attempt", "metrics": {"scripts": ["solve.py"], "commands": []}},
        {"name": "task1-cli", "kind": "attempt", "metrics": {"scripts": [], "commands": ["kapso build a", "kapso build b"]}},
    ]}

    class _Driver:
        sandbox = tmp_path
        out = tmp_path
        artifact = staticmethod(lambda name, text: (tmp_path / name).write_text(text) or f"artifacts/{name}")

    step = s1.S1.step_flow_counts(_Driver(), record)
    assert step["metrics"] == {"python_lines": 3, "cli_commands": 2}, "the longest single task's flow, not the sum over tasks"


def test_the_released_baseline_needs_the_staged_docs(tmp_path):
    # The clone reaches the machine without .git, so the tag's docs come
    # staged as released-docs/ by semantic.sh; a pack without them must not
    # fall back to the branch's docs.
    args = argparse.Namespace(clone=tmp_path / "clone", out=tmp_path / "out", sandbox=None, host="claude_code", package="released")
    (tmp_path / "clone").mkdir()
    with pytest.raises(RuntimeError, match="no released docs"):
        s1.S1(args).step_sandbox()


def test_the_prerequisite_campaign_is_built_in_the_packages_own_verb():
    # The branch's `build` detaches itself; 0.7.10 has no `build`, `--detach`
    # or stop verb, so its own launch verb is backgrounded and signalled by pid.
    branch = s1.prerequisite_launch("branch", Path("/sb/campaign-prereq"), 60, 3)
    released = s1.prerequisite_launch("released", Path("/sb/campaign-prereq"), 60, 3)
    assert branch.startswith("kapso build ") and "--detach" in branch and "--budget-minutes 60 --budget-experiments 3" in branch
    assert released.startswith("nohup kapso ") and released.split()[2] != "build" and "-f kapso-brief.md" in released and released.endswith("&")
    assert "-o /sb/campaign-prereq -i 3 --time-budget-minutes 60" in released and "--initial-repo repo" in released


def test_an_execution_is_an_invocation_of_the_packages_own_verb_never_a_search_that_mentions_it(tmp_path):
    # s1-claude-1 had counted a `sed` over the docs as task 2's first run,
    # and the released arm ran 0.7.10's verb `evolve` without a single counted
    # execution — the headline comparison was the detector's. A command
    # counts when one of its segments invokes kapso with the verb (wrappers
    # and variable assignments stripped, a `bash -lc` opened); the released
    # arm is read under its own verbs.
    searches = _stream([("sed -n '1,40p' docs/reference/cli.mdx | grep 'kapso build'", False), ("grep -rn 'kapso build' docs", False), ("echo kapso build --help", False)])
    assert s1.executions(searches, "build", "cli", tmp_path) == []
    wrapped = _stream([("cd task && KAPSO_HOME=/tmp nohup kapso build --brief kapso-brief.md --output ../campaign-1 --detach", False)])
    assert len(s1.executions(wrapped, "build", "cli", tmp_path)) == 1
    assert s1.invocations("/bin/bash -lc 'cd task && ./venv/bin/python run.py'", "python") == ["python run.py"]
    released = _stream([("kapso evolve -f kapso-brief.md -o campaign-1 -i 1", False), ("kapso build --brief b.md --output c", True)])   # 0.7.10's verb
    assert [entry["command"][:12] for entry in s1.executions(released, "build", "cli", tmp_path, package="released")] == ["kapso evolve"]   # 0.7.10's verb
    assert s1.executions(released, "build", "cli", tmp_path, package="branch")[0]["command"].startswith("kapso build")
    assert s1.executions(_stream([("kapso evolve -o c --resume", False)]), "resume", "cli", tmp_path, package="released")   # 0.7.10's verb
    (tmp_path / "old.py").write_text("from kapso import Kapso\nsolution = Kapso().evolve('task/kapso-brief.md', output='campaign-1')\n")   # 0.7.10's verb
    assert len(s1.executions(_stream([("./venv/bin/python old.py", False)]), "build", "python", tmp_path, package="released")) == 1
    assert s1.executions(_stream([("./venv/bin/python old.py", False)]), "build", "python", tmp_path, package="branch") == []
    assert s1.executions(_stream([("python -c 'from kapso import Kapso; Kapso().build(1)'", False)]), "build", "python", tmp_path) == [], "a one-liner is not a script"


def test_the_flows_artifact_carries_each_attempts_script_or_commands_for_the_judge(tmp_path):
    (tmp_path / "run_build.py").write_text("from kapso import Kapso\n# the brief\nsolution = Kapso().build('b')\n")
    record = {"steps": [
        {"name": "task1-python", "kind": "attempt", "metrics": {"first_try": True, "wrong_calls": 0, "executions": 1, "evidence": True, "scripts": ["run_build.py"], "commands": []}},
        {"name": "task1-cli", "kind": "attempt", "metrics": {"first_try": False, "wrong_calls": 1, "executions": 2, "evidence": True, "scripts": [], "commands": ["kapso build --brief b", "kapso build --brief b --repo r"]}},
    ]}
    text = s1.flows_text(record, tmp_path)
    assert "## task1-python: first try True, wrong calls 0, executions 1, evidence True" in text
    assert "script run_build.py (2 code lines):" in text and "    solution = Kapso().build('b')" in text
    assert "command: kapso build --brief b --repo r" in text
    written = {}
    step = s1.flow_counts(record, tmp_path, lambda name, body: written.setdefault(name, body) and f"artifacts/{name}")
    assert step["metrics"] == {"python_lines": 2, "cli_commands": 2} and "flows.md" in written and step["artifacts"] == ["artifacts/flow-counts.json", "artifacts/flows.md"]


def test_the_splitter_reads_quotes_and_subshells_as_the_shell_does(tmp_path):
    # A `|` inside a quoted pattern is the pattern's, not a pipe: a search that
    # mentions the verb is never an execution (rel-s1-codex counted one as a wrong call).
    search = _codex_stream([("/bin/bash -lc \"rg -n 'config|budget|kapso build' docs/reference/cli.mdx && cat campaign-3.log\"", 0)])
    assert s1.executions(search, "build", "cli", tmp_path) == []
    assert s1.command_segments("rg -n 'a|b' x | grep c; (kapso build --output c &)") == ["rg -n 'a|b' x", "grep c", "kapso build --output c"]


def test_a_launch_after_a_heredoc_in_the_same_command_is_counted(tmp_path):
    # rel-s1-codex's second python flow wrote its script by heredoc and ran it
    # in the same command; the apostrophe in the script's docstring opened a
    # quote the splitter never closed, and the launch after the terminator was
    # read as part of the heredoc: no execution counted, the step's first try
    # lost. A heredoc's body is the command's, whatever quotes it holds.
    (tmp_path / "s.py").write_text("\"\"\"Kapso's API.\"\"\"\nfrom kapso import Kapso\nKapso().build(brief=1)\n", encoding="utf-8")
    heredoc = "cat > s.py <<'PY'\n\"\"\"Kapso's API.\"\"\"\nfrom kapso import Kapso\nKapso().build(brief=1)\nPY\n./venv/bin/python s.py && echo done"
    segments = s1.command_segments(heredoc)
    assert segments[-2:] == ["./venv/bin/python s.py", "echo done"]
    assert segments[0].startswith("cat > s.py <<'PY'") and segments[0].rstrip().endswith("PY") and len(segments) == 3
    rows = _codex_stream([(f"/bin/bash -lc \"{heredoc}\"", 0)])
    assert [entry["errored"] for entry in s1.executions(rows, "build", "python", tmp_path)] == [False]
    # Two heredocs on one line, and `<<<` (a here-string) is not a heredoc.
    assert s1._split_outside_quotes("cat <<A <<B\na\nA\nb\nB\nkapso build") == ["cat <<A <<B\na\nA\nb\nB", "kapso build"]
    assert s1.command_segments("grep x <<< 'a\nb'; kapso build") == ["grep x <<< 'a b'", "kapso build"]


def test_an_execution_errs_by_the_hosts_exit_status_not_by_its_outputs_words(tmp_path):
    # A deploy streams its agent's session log — probes, tracebacks — and exits 0.
    rows = _stream([("kapso deploy campaign-prereq --target local", False)])
    rows[1]["message"]["content"][0]["content"] = "Traceback (most recent call last):\n  ImportError: no sklearn\n...\nDeployed: 1"
    assert [entry["errored"] for entry in s1.executions(rows, "deploy", "cli", tmp_path)] == [False]
    codex = _codex_stream([("kapso deploy campaign-prereq --target local", 0)])
    codex[0]["item"]["aggregated_output"] = "Error: probe failed\nDeployed: 1"
    assert [entry["errored"] for entry in s1.executions(codex, "deploy", "cli", tmp_path)] == [False]
    # No exit status at all: the output's words decide.
    codex[0]["item"]["exit_code"] = None
    assert [entry["errored"] for entry in s1.executions(codex, "deploy", "cli", tmp_path)] == [True]


def test_a_command_still_running_at_the_turns_end_is_counted_with_no_outcome(tmp_path):
    started = {"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "kapso build --output c --resume", "exit_code": None}}
    found = s1.executions([started], "resume", "cli", tmp_path)
    assert found == [{"command": "kapso build --output c --resume", "errored": None}], "a foreground resume that outran the turn is an execution, its outcome unknown"
    assert s1.attempt_metrics(found, evidence=True, started=0.0, evidence_at=30.0)["first_try"] is False
    completed = {"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "kapso build --output c --resume", "exit_code": 0, "aggregated_output": "ok"}}
    assert s1.executions([started, completed], "resume", "cli", tmp_path) == [{"command": "kapso build --output c --resume", "errored": False}]


def test_a_flows_lines_are_logical_lines():
    call_across_lines = (
        "from kapso import Kapso, Brief, Budget\n"
        "solution = Kapso().build(\n    Brief.read('task/kapso-brief.md'),\n    repo='task/repo',\n    output='campaign-1',\n"
        "    budget=Budget(minutes=10, experiments=1),\n)\n"
        "print(solution.explain())\n"
    )
    assert s1.logical_lines(call_across_lines) == 3, "a call written across six physical lines is one line of the flow"
