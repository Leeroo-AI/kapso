"""S4's driver (docs/plans/redesign/wp9-semantic-suite.md, change 7): the
fact sheet is extracted from a real trace (the second machine's churn
campaign) and the branch's history, renders the six questions, and says
"nothing voided" and "not established" where the record does; the judge's
block grades each reader by the questions right in both passes, the run
by the worst reader."""

import importlib.util
import subprocess
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "redesign"))
spec = importlib.util.spec_from_file_location("s4_reading", ROOT / "scripts" / "redesign" / "s4_reading.py")
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)
TRACE = json.loads((ROOT / "tests" / "fixtures" / "semantic_transcripts" / "churn-trace.json").read_text())


def test_the_fact_sheet_is_extracted_from_the_trace_and_the_history():
    facts = s4.fact_sheet(TRACE, labels=True)
    assert facts["what won"]["branch"] == "generic_exp_0" and facts["what won"]["score"] == 0.847 and facts["what won"]["commit"].startswith("4223437")
    assert facts["what was voided and on what evidence"] == "nothing voided"
    assert facts["what was rejected and why"].startswith("nothing rejected")
    # The two Codex repo-memory sessions carry no cost: the sheet says so by
    # host, so a reader's "not established for the whole" reads as right.
    assert facts["what it cost"] == {"usd": 4.0038255, "coverage": "claude_code", "unpriced_runs": 0, "sessions_without_a_cost": {"codex": 2}}
    merge = facts["safe to merge"]
    assert merge["labels_in_history"].startswith("the hold-out labels are in the delivered commit's own history")
    assert "kapso_evaluation/result.json" in merge["files_changed"] and merge["integrity"] == "no integrity error on the delivered attempt"
    assert s4.fact_sheet(TRACE, labels=None)["safe to merge"]["labels_in_history"].startswith("not established")
    sheet = s4.render_sheet(facts)
    assert all(f"## {key}" in sheet for key, _ in s4.QUESTIONS) and sheet.startswith("# The fact sheet")


def test_a_scored_runner_up_is_not_a_rejection_but_a_void_is():
    # A scored non-winner is a runner-up, not a rejection; a rejection is a
    # voided entry, split from an integrity void by its substance: the engine
    # words an integrity failure "rejected: Provided evaluation integrity
    # check failed (changed: evaluate.py)" (s4-claude-1's campaign), and it
    # is the void the design means.
    trace = json.loads(json.dumps(TRACE))
    trace["attempts"].append({"branch": "generic_exp_1", "evaluation": {"score": 0.84, "valid": True, "integrity_error": None}, "cards": {"read": [], "listed": 0, "load_bearing": []}, "judge": None, "idea": "x", "verification": None})
    trace["voided"] = [{"node": 3, "branch": "generic_exp_2", "reason": "rejected: judged not valid", "evidence": ["the judge said so"]},
                       {"node": 4, "branch": "generic_exp_3", "reason": "void: integrity error on kapso_evaluation/evaluate.py", "evidence": ["hash mismatch"]}]
    facts = s4.fact_sheet(trace, labels=False)
    assert facts["scored_not_delivered"] == [{"branch": "generic_exp_1", "score": 0.84}]
    assert facts["what was rejected and why"] == [{"node": 3, "branch": "generic_exp_2", "reason": "rejected: judged not valid", "evidence": ["the judge said so"]}]
    assert facts["what was voided and on what evidence"] == [{"node": 4, "branch": "generic_exp_3", "reason": "void: integrity error on kapso_evaluation/evaluate.py", "evidence": ["hash mismatch"]}]
    assert facts["campaign_holds"] == {"void": True, "rejection": True}
    # The engine's own semantics decide: the entry's attempt carries an
    # integrity error when the integrity check voided it, whatever the
    # reason's first word; the same wording on an attempt without one is
    # the judge's rejection.
    engine_wording = {"node": 0, "branch": "generic_exp_0", "reason": "rejected: Provided evaluation integrity check failed (changed: evaluate.py)", "evidence": ["$ python3 kapso_evaluation/evaluate.py"]}
    node_zero = next(attempt for attempt in trace["attempts"] if attempt.get("node") == 0)
    node_zero["evaluation"]["integrity_error"] = "Provided evaluation integrity check failed (changed: evaluate.py)"
    trace["voided"] = [engine_wording]
    facts = s4.fact_sheet(trace, labels=False)
    assert facts["what was voided and on what evidence"] == [engine_wording], "the integrity failure is the void, whatever the engine's first word"
    assert facts["what was rejected and why"].startswith("nothing rejected") and facts["campaign_holds"] == {"void": True, "rejection": False}
    node_zero["evaluation"]["integrity_error"] = None
    facts = s4.fact_sheet(trace, labels=False)
    assert facts["what was rejected and why"] == [engine_wording] and facts["campaign_holds"] == {"void": False, "rejection": True}


def test_labels_in_history_reads_the_delivered_commits_own_ancestry(tmp_path):
    repo = tmp_path / "campaign"
    (repo / "kapso_evaluation" / "private").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    run("config", "user.email", "x@example.com")
    run("config", "user.name", "x")
    (repo / "predict.py").write_text("x = 1\n")
    run("add", "-A")
    run("commit", "-q", "-m", "clean, no labels")
    clean = run("rev-parse", "HEAD").stdout.strip()
    assert s4.labels_in_history(repo, clean) is False, "the clean commit carries no labels"
    (repo / "kapso_evaluation" / "private" / "holdout_labels.csv").write_text("id,churned\n")
    run("add", "-A")
    run("commit", "-q", "-m", "leak the labels")
    leaked = run("rev-parse", "HEAD").stdout.strip()
    assert s4.labels_in_history(repo, leaked) is True and s4.labels_in_history(repo, clean) is False
    assert s4.labels_in_history(tmp_path / "no-repo", leaked) is None and s4.labels_in_history(repo, None) is None


def test_the_steps_and_the_questions_match_the_suite():
    assert s4.STEP_NAMES == s4.SUITE["tests"]["S4"]["required_steps"]["claude_code"]
    assert [key for key, _ in s4.QUESTIONS] == s4.SUITE["tests"]["S4"]["judge"]["dimensions"]
    assert all(text in s4.PROMPT for _, text in s4.QUESTIONS) and "not established" in s4.PROMPT


def test_the_campaign_step_builds_at_a_budget_two_attempts_fit_and_requires_the_void_alone():
    # The campaign S4 builds holds the void the integrity check guarantees
    # (attempt one, the evaluator edited in main) and a delivered attempt two,
    # at S5's budget: a hundred minutes let a first attempt run 86 of them on
    # the gate's Codex machine, and no second attempt ran. A rejection is the
    # verifier's or the judge's word on an attempt's own work, nothing a driver
    # arranges: it is recorded, never a bar.
    assert s4.SUITE["builds"]["s4"] == s4.SUITE["builds"]["s5"], "the budget two attempts fit in, as S5's campaigns show"
    assert set(s4.SUITE["tests"]["S4"]["bars"]) == {"readers_correct_min", "campaign_has_void"}
    assert not hasattr(s4, "LEAK")


def _item(scores):
    return {key: {"score": score, "evidence": []} for (key, _), score in zip(s4.QUESTIONS, scores)}


def test_the_judges_block_grades_each_reader_by_the_questions_right_in_both_passes():
    record = {"steps": [
        {"name": "fact-sheet", "kind": "shell"},
        {"name": "reading-claude_code", "kind": "reading", "metrics": {"correct": None}},
        {"name": "reading-codex", "kind": "reading", "metrics": {"correct": None}},
    ], "metrics": {}, "judge": {"failure": None, "passes": [
        {"items": {"reading-claude_code": _item([1, 1, 1, 1, 1, 1]), "reading-codex": _item([1, 1, 0, 1, 1, 1])}},
        {"items": {"reading-claude_code": _item([1, 1, 1, 1, 1, 0]), "reading-codex": _item([1, 1, 1, 1, 1, 1])}},
    ]}}
    graded = s4.grade(record)
    assert graded["steps"][1]["metrics"]["correct"] == 5 and graded["steps"][2]["metrics"]["correct"] == 5
    assert graded["metrics"]["readers_correct_min"] == 5
    failed = {"steps": [{"name": "reading-codex", "kind": "reading", "metrics": {"correct": None}}], "metrics": {}, "judge": {"failure": "x", "passes": []}}
    assert s4.grade(failed)["metrics"] == {}


def test_an_adjudicated_question_is_counted_as_the_adjudication_decided_not_the_bare_minimum():
    # Claude's "why" (question 2) straddles 0/1 across passes and is
    # adjudicated correct: it must count toward the reader's score, not stay
    # wrong on the two-pass minimum.
    why = [key for key, _ in s4.QUESTIONS][1]
    record = {"steps": [{"name": "reading-claude_code", "kind": "reading", "metrics": {"correct": None}}], "metrics": {}, "judge": {
        "failure": None,
        "passes": [{"items": {"reading-claude_code": _item([1, 0, 1, 1, 1, 1])}}, {"items": {"reading-claude_code": _item([1, 1, 1, 1, 1, 1])}}],
        "adjudication": {f"{why}|reading-claude_code": {"score": 1, "evidence": []}},
    }}
    assert s4.grade(record)["steps"][0]["metrics"]["correct"] == 6, "the adjudication resolved question 2 to correct"
