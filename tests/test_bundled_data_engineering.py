"""The data-engineering harness's hooks and the parts of its scripts that need no warehouse: each hook decides
from the event and the workspace alone, on both hosts' event shapes, and stays silent on a repository that does
not look like the one it was written beside; the environment file takes a check that reports by exit status;
a script says that dbt is missing instead of ending in a traceback."""

import importlib.util
import json
import subprocess
import sys

import pytest
import yaml

from kapso.harness import BUNDLED_ROOT

HARNESS = BUNDLED_ROOT / "data-engineering"
EVIDENCE = HARNESS / "skills" / "validation-report" / "scripts" / "evidence.py"
FINGERPRINT = HARNESS / "skills" / "data-diff" / "scripts" / "fingerprint.py"
LINEAGE = HARNESS / "skills" / "lineage-bisection" / "scripts" / "lineage.py"
NOTICE = ("shared-layer-notice", "shared_layer_notice.py")


def run_hook(name, script, event, cwd):
    completed = subprocess.run(
        [sys.executable, str(HARNESS / "hooks" / name / script)],
        input=json.dumps(event), capture_output=True, text=True, check=True, cwd=str(cwd),
    )
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def run_script(script, arguments, cwd, env=None):
    return subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True, cwd=str(cwd), env=env)


@pytest.mark.parametrize("event", [
    {"tool_name": "Write", "tool_input": {"file_path": "prepared/environment.json", "content": "{}"}},
    {"tool_name": "apply_patch", "tool_input": {"command": "*** Begin Patch\n*** Update File: prepared/baseline/spec_orders.json\n*** End Patch"}},
], ids=["claude-code", "codex"])
def test_protect_prepared_denies_a_write_under_prepared_on_both_hosts(event, tmp_path):
    output = run_hook("protect-prepared", "protect_prepared.py", event, tmp_path)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny" and "prepared/" in output["permissionDecisionReason"]
    assert run_hook("protect-prepared", "protect_prepared.py", {"tool_input": {"file_path": "models/orders.sql"}}, tmp_path) == {}


@pytest.mark.parametrize("event", [
    {"tool_input": {"file_path": "models/staging/stg_orders.sql"}},
    {"tool_input": {"command": "*** Begin Patch\n*** Update File: models/staging/stg_orders.sql\n*** End Patch"}},
], ids=["claude-code", "codex"])
def test_the_shared_layer_notice_names_a_staging_model(event, tmp_path):
    context = run_hook(*NOTICE, event, tmp_path)["hookSpecificOutput"]["additionalContext"]
    assert "models/staging/stg_orders.sql" in context and "downstream consumers" in context


def test_the_shared_layer_notice_reads_only_the_directories_inside_the_workspace(tmp_path):
    # Regression: the host gives an absolute path, and a checkout that sits under a directory named `tests` or
    # `staging` got the notice on every write of every session. The harness's own records are not a layer.
    workspace = tmp_path / "tests" / "staging" / "checkout"
    workspace.mkdir(parents=True)
    for path in ("models/marts/orders.sql", "results/staging/notes.json", "prepared/tests/spec.json"):
        assert run_hook(*NOTICE, {"tool_input": {"file_path": str(workspace / path)}}, workspace) == {}
    noted = run_hook(*NOTICE, {"tool_input": {"file_path": str(workspace / "tests/orders_unique.sql")}}, workspace)
    assert "a test" in noted["hookSpecificOutput"]["additionalContext"]


def test_the_session_brief_is_silent_before_setup_and_never_prints_a_target_it_was_not_given(tmp_path):
    assert run_hook("session-brief", "session_brief.py", {}, tmp_path) == {}
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "environment.json").write_text(json.dumps({"reset_command": "make reset"}))
    context = run_hook("session-brief", "session_brief.py", {}, tmp_path)["hookSpecificOutput"]["additionalContext"]
    # Regression: every key of the file is optional, and an absent target was printed as `None`.
    assert "None" not in context and "no target reads production" in context and "the profile's default target" in context


def test_the_stop_hook_asks_once_for_the_proof_of_a_single_change(tmp_path):
    decision = run_hook("require-proof", "require_proof.py", {}, tmp_path)
    assert decision["decision"] == "block" and "results/proof.json does not exist" in decision["reason"]
    assert "ledger" not in decision["reason"]                    # a single change has no backlog to keep
    assert run_hook("require-proof", "require_proof.py", {"stop_hook_active": True}, tmp_path) == {}


def test_the_stop_hook_does_not_ask_a_proof_for_one_model(tmp_path):
    # Regression: `model` was required although nothing reads it, so a change to a macro, a snapshot or
    # several datasets could not write a proof the hook accepted.
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "proof.json").write_text(json.dumps({
        "change_kind": "refactor", "label": "unsure", "validated": {"commands": [], "skipped": []}, "changed_files": ["macros/cents.sql"]}))
    reason = run_hook("require-proof", "require_proof.py", {}, tmp_path)["reason"]
    assert "`model` is missing" not in reason and "was never measured" in reason and "validated.skipped" in reason


def test_the_manifest_keeps_the_prepared_guard_off_the_session_that_writes_prepared():
    modules = yaml.safe_load((HARNESS / "harness.yaml").read_text(encoding="utf-8"))["modules"]
    assert "protect-prepared" not in modules["setup"]["hooks"]
    assert all("protect-prepared" in modules[name]["hooks"] for name in modules if name != "setup")
    assert modules["implementation"]["hooks"] == ["session-brief", "protect-prepared", "shared-layer-notice", "require-proof"]


def environment(tmp_path, settings, tickets=()):
    (tmp_path / "prepared").mkdir(exist_ok=True)
    (tmp_path / "prepared" / "environment.json").write_text(json.dumps(settings))
    if tickets:
        (tmp_path / "prepared" / "backlog").mkdir(exist_ok=True)
        (tmp_path / "prepared" / "backlog" / "inventory.json").write_text(json.dumps([{"ticket": ticket} for ticket in tickets]))
    return run_script(EVIDENCE, ["environment", "--ticket", "change"], tmp_path)


@pytest.mark.parametrize("status, verdict", [(0, "pass"), (3, "fail")])
def test_an_acceptance_check_without_patterns_is_read_by_its_exit_status(status, verdict, tmp_path):
    # Regression: the command had to take a ticket's key and print a line a pattern reads, so a repository
    # whose check is a test command that exits non-zero on failure could not pin it at all.
    done = environment(tmp_path, {"acceptance_command": f"{sys.executable} -c 'import sys; sys.exit({status})'"})
    assert done.returncode == 0 and f"read as {verdict} (exit {status}" in done.stdout


def test_one_command_cannot_check_a_backlog_without_the_tickets_key(tmp_path):
    done = environment(tmp_path, {"acceptance_command": f"{sys.executable} -c 'pass'"}, tickets=("late-orders", "refund-share"))
    assert done.returncode == 1 and "holds 2 tickets" in done.stdout
    assert environment(tmp_path, {"acceptance_fail_pattern": "FAIL"}).returncode == 1          # a pattern needs its command


def test_a_ticket_key_that_cannot_name_a_file_and_a_branch_is_refused(tmp_path):
    done = run_script(EVIDENCE, ["label", "--ticket", "DATA/123"], tmp_path)
    assert done.returncode == 2 and "letters, digits" in done.stderr


def test_a_script_says_that_dbt_is_missing(tmp_path):
    # Regression: on a machine or a stack without dbt the scripts ended in a FileNotFoundError traceback.
    done = run_script(LINEAGE, ["orders"], tmp_path, env={"PATH": str(tmp_path)})
    assert done.returncode != 0 and "dbt is not on PATH" in done.stderr and "Traceback" not in done.stderr


def test_a_numeric_column_is_told_by_its_types_leading_word():
    spec = importlib.util.spec_from_file_location("fingerprint", FINGERPRINT)
    fingerprint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fingerprint)
    numeric = [name for name in ("bigint", "int64", "number(38,0)", "decimal(18,3)", "double precision", "float8", "hugeint",
                                 "interval", "interval day to second", "point", "varchar", "timestamp")
               if fingerprint.NUMERIC_TYPE.match(name)]
    # Regression: a substring test read `interval` and `point` as integers and summed them.
    assert numeric == ["bigint", "int64", "number(38,0)", "decimal(18,3)", "double precision", "float8", "hugeint"]
