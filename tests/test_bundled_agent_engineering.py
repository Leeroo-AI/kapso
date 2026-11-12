"""The agent-engineering harness's triage script against the results convention its skill states
(skills/done-check-and-eval): the adapter's own vocabulary is read as written and never guessed at, so a file
from another runner is classed by what it says or refused by name, not silently filed under the nearest class."""

import json
import subprocess
import sys

from kapso.harness import BUNDLED_ROOT

TRIAGE = BUNDLED_ROOT / "agent-engineering" / "skills" / "episode-triage" / "scripts" / "triage.py"


def step(index, name, kind, **fields):
    return {"index": index, "requestor": "assistant", "name": name, "kind": kind, "matched": False, "attempted": False, "diff_keys": [], **fields}


def call(turn, name, kind, **fields):
    return {"turn": turn, "requestor": "assistant", "name": name, "kind": kind, "arg_keys": [], **fields}


def triage(tmp_path, rows, *arguments):
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps({"id": f"t{number}", "trial": 0, "passed": False, **row}) + "\n" for number, row in enumerate(rows)))
    done = subprocess.run([sys.executable, str(TRIAGE), str(path), "--json", *arguments], capture_output=True, text=True)
    return done, (json.loads(done.stdout)["triage"] if done.returncode == 0 else None)


def test_an_error_is_the_adapters_category_and_never_the_runners_text(tmp_path):
    # Regression: the script guessed a category from substrings of one runner's messages, so another
    # adapter's `sandbox_crash` came back as an endpoint error and "empty patch" as an empty reply.
    done, report = triage(tmp_path, [{"error": "sandbox_crash"}, {"error": "overflow"}])
    assert report["classes"] == {"sandbox_crash": 1, "overflow": 1}
    done, report = triage(tmp_path, [{"error": "Connection timed out after 30 seconds"}])
    assert report is None and "must be a category the adapter chose" in done.stderr


def test_a_requestor_outside_the_convention_is_refused_by_name(tmp_path):
    # Regression: the extra-write class compares the requestor with `assistant`; a file that says `agent`
    # lost the class without a word.
    done, report = triage(tmp_path, [{"calls": [call(1, "close_ticket", "write", requestor="agent")]}])
    assert report is None and "`requestor` must be one of assistant, user, tool" in done.stderr


def test_an_untaken_step_of_a_kind_the_adapter_added_is_not_called_a_write(tmp_path):
    _, report = triage(tmp_path, [{"expected_actions": [step(0, "run_tests", "execute")], "calls": []}])
    assert "step_never_made" in report["classes"] and "write_never_made" not in report["classes"]


def test_the_runners_own_terminations_are_named_or_reported_unread(tmp_path):
    # Regression: the words for "the episode simply ended" were a fixed guess, so on a runner that says
    # `user_stop` the ended-early class never fired and nothing said why.
    rows = [{"termination": "user_stop", "expected_actions": [step(0, "refund", "write")], "calls": []}]
    _, report = triage(tmp_path, rows)
    assert report["terminations"] == {"user_stop": {"episodes": 1, "read_as": "unread"}} and "ended_early" not in report["classes"]
    _, report = triage(tmp_path, rows, "--endings", "user_stop")
    assert report["terminations"]["user_stop"]["read_as"] == "ending" and report["classes"]["ended_early"] == 1
    _, report = triage(tmp_path, [{"termination": "hit_token_ceiling"}], "--limit-terminations", "ceiling")
    assert report["classes"] == {"timeout": 1}


def test_siblings_are_the_adapters_families_before_the_name_heuristic(tmp_path):
    # Regression: siblings were told by a verb prefix and a numeric suffix only, one runner's naming;
    # `createIssue` and `updateIssue` were never siblings.
    def row(**family):
        return {"expected_actions": [step(0, "createIssue", "write", **family)], "calls": [call(1, "updateIssue", "write", **family)]}

    _, named = triage(tmp_path, [row(family="issue")])
    _, unnamed = triage(tmp_path, [row()])
    assert "wrong_sibling_tool" in named["classes"] and "(name heuristic)" not in named["symptoms"][0]["detail"]
    assert "write_never_made" in unnamed["classes"] and "wrong_sibling_tool" not in unnamed["classes"]
