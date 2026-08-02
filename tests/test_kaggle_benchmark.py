"""Contract tests for the Kaggle code-competition benchmark.

What must hold: the handler context is the statement plus the minimal kapso
contract (and stays free of the removed protocol/economics sermons), the
runner's leaderboard parsing survives CLI pagination noise and windows
submissions to the run, and the preflight parses a competition slug from its
URL (fail-loud on a malformed one).
"""

import json
import os
import time

import pytest
import yaml

from benchmarks.kaggle import kernel_slots
from benchmarks.kaggle.handler import KaggleNotebookHandler
from benchmarks.kaggle.preflight import slug_from_url
from benchmarks.kaggle.runner import (
    RULES_PATH,
    audit_kernel,
    best_public_score,
    discover_run_kernels,
    parse_submissions_json,
    submission_matches_template,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "kaggle", "config.yaml"
)

SESSION_CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}
KAGGLE = {"competition": "ioai-2026-ai-models-track-practice-task-1"}


def make_handler(tmp_path, **overrides):
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "RULES.md").write_text("rules body")
    kwargs = dict(
        task_dir=str(tmp_path / "task"),
        statement="statement body",
        deadline_ts=time.time() + 7200,
        session_caps=SESSION_CAPS,
        kaggle=KAGGLE,
    )
    kwargs.update(overrides)
    return KaggleNotebookHandler(**kwargs)


def test_handler_context_is_statement_plus_minimal_contract(tmp_path):
    handler = make_handler(tmp_path)
    context = handler.get_problem_context()
    assert context.startswith("statement body")
    assert KAGGLE["competition"] in context
    assert "operator approval" not in context
    # Competition framing, not a safety floor: highest expected score wins.
    assert "highest expected final score" in context
    assert "at least one" not in context
    # End-to-end clock: submission round trips are inside the budget.
    assert "END-TO-END" in context and "round trip" in context
    # Prior attempts are read off Kaggle, not off a file the lanes maintain.
    assert "kaggle competitions submissions" in context
    assert "kernels pull" in context
    assert "best_score" not in context
    assert "<score>" in context
    # The protocol/economics sermons must stay gone.
    for banned in ("SUBMISSION BUDGET", "INSURANCE", "flock",
                   "Reward & time economics", "push TWICE"):
        assert banned not in context
    # Budget the contract we author, not the environment it renders in: the
    # statement is unbounded and the task dir is 37 chars in production but
    # ~140 under pytest, which used to make this guard measure tmp_path.
    contract = context.split("# Kapso operational context", 1)[1]
    assert len(contract.replace(handler.task_dir, "/task")) < 2200


def test_handler_rejects_missing_kaggle_slug(tmp_path):
    with pytest.raises(ValueError, match="kaggle"):
        make_handler(tmp_path, kaggle={})


def test_handler_requires_the_staged_rules_and_points_the_agent_at_them(tmp_path):
    # A run without the organizers' rules could ship a solution that breaks one
    # (two GPUs, an external checkpoint) and be voided — so fail at construction.
    context = make_handler(tmp_path).get_problem_context()
    assert os.path.join(str(tmp_path / "task"), "RULES.md") in context
    # The rules themselves stay in RULES.md: the handler is the modality-agnostic
    # contract, and not every task submits a kernel — some only upload a
    # prediction file. Reading past attempts off Kaggle is fine; authoring or
    # pushing a kernel here is what misleads those tasks.
    body = context.split("# Kapso operational context", 1)[1].lower()
    for kernel_mechanic in ("cuda", "kernels push", "machine_shape",
                            "enable_gpu", "kernel-metadata"):
        assert kernel_mechanic not in body
    os.remove(tmp_path / "task" / "RULES.md")
    with pytest.raises(FileNotFoundError, match="RULES.md"):
        KaggleNotebookHandler(
            task_dir=str(tmp_path / "task"), statement="s",
            deadline_ts=time.time() + 60, session_caps=SESSION_CAPS,
            kaggle=KAGGLE,
        )


def test_staged_rules_carry_the_binding_kernel_constraints():
    # The handler deliberately states no kernel mechanics, so RULES.md is the
    # only place the agent learns them: a trim that drops the one-GPU pin or the
    # never-P100 pin would silently void submissions that still score.
    rules = open(RULES_PATH, encoding="utf-8").read()
    assert "cuda:0" in rules
    assert "NvidiaTeslaT4" in rules and "P100" in rules
    assert "50 submissions per task" in rules


def test_parse_submissions_json_tolerates_pagination_noise():
    raw = (
        "Next Page Token = CfDJ8ABC\n"
        '[{"date": "2026-07-22 15:30:00", "status": "complete", '
        '"publicScore": "0.71", "description": "baseline"}]'
    )
    subs = parse_submissions_json(raw)
    assert subs[0]["publicScore"] == "0.71"
    with pytest.raises(ValueError, match="JSON payload"):
        parse_submissions_json("no brackets here")


def test_best_public_score_windows_to_run_start():
    submissions = [
        {"date": "2026-07-22 10:00:00", "status": "complete",
         "publicScore": "0.99", "description": "yesterday's run"},
        {"date": "2026-07-22 15:30:00", "status": "complete",
         "publicScore": "0.41", "description": "insurance"},
        {"date": "2026-07-22 16:10:00", "status": "complete",
         "publicScore": "0.78", "description": "distill"},
        {"date": "2026-07-22 16:40:00", "status": "pending",
         "publicScore": "", "description": "still scoring"},
    ]
    report = best_public_score(submissions, "2026-07-22T15:00:00+00:00")
    assert report["best"]["score"] == 0.78
    assert len(report["submissions"]) == 3  # pre-run entry excluded


def test_audit_kernel_flags_external_pulls(tmp_path):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "script.py").write_text(
        "from transformers import ASTForAudioClassification\n"
        "m = ASTForAudioClassification.from_pretrained('MIT/ast-finetuned')\n"
        "ok = ASTForAudioClassification.from_pretrained('/kaggle/input/x/model')\n"
    )
    findings = audit_kernel(str(kernel))
    assert len(findings) == 1 and "MIT/ast-finetuned" in findings[0]


def test_handler_keeps_the_full_round_trip_reserve(tmp_path):
    # The reserve covers one submission round trip and is never shrunk: the
    # signal that would license shrinking it (a banked public score) lived in
    # the shared board, and the board is gone — Kaggle is the only authority
    # on what scored, and the budget path does not query it.
    assert make_handler(tmp_path).deliverable_ready_reserve_seconds() is None


def test_discover_run_kernels_finds_namespaced_lane_dirs(tmp_path):
    # K-way lanes namespace their submission dirs, so there is no single
    # canonical kernel path — discovery must walk them and dedupe.
    submission = tmp_path / "task" / "submission"
    for lane, ref in (("lane0_exp_0", "u/kernel-a"), ("lane2_exp_2", "u/kernel-b"),
                      ("lane5_exp_5", "u/kernel-a")):
        (submission / lane).mkdir(parents=True)
        (submission / lane / "kernel-metadata.json").write_text(
            json.dumps({"id": ref, "code_file": "script.py"}))
    assert discover_run_kernels(str(tmp_path / "task")) == ["u/kernel-a", "u/kernel-b"]
    (submission / "broken").mkdir()
    (submission / "broken" / "kernel-metadata.json").write_text('{"title": "x"}')
    with pytest.raises(ValueError, match="'id'"):
        discover_run_kernels(str(tmp_path / "task"))


def test_submission_matches_template_gates_on_ids_not_just_size(tmp_path):
    template = tmp_path / "template.csv"
    template.write_text("path,target\naudio/a.wav,0\naudio/b.wav,0\n")
    good = tmp_path / "good.csv"
    good.write_text("path,target\naudio/a.wav,17\naudio/b.wav,3\n")
    assert submission_matches_template(str(good), str(template))
    # right row count, wrong ids — the failure a length check would miss
    reordered = tmp_path / "reordered.csv"
    reordered.write_text("path,target\naudio/b.wav,3\naudio/a.wav,17\n")
    assert not submission_matches_template(str(reordered), str(template))
    short = tmp_path / "short.csv"
    short.write_text("path,target\naudio/a.wav,17\n")
    assert not submission_matches_template(str(short), str(template))


def test_kernel_slots_never_overcommits_and_reclaims_dead_lanes(tmp_path):
    # The whole point: Kaggle's 2 concurrent GPU sessions are per ACCOUNT, so
    # parallel lanes must queue rather than race. Over-issuing a ticket would
    # send a lane into a push that Kaggle rejects.
    task = tmp_path / "task"
    task.mkdir()
    (task / ".kernel_slots_config.json").write_text(
        json.dumps({"gpu": 2, "cpu": 5, "ttl_seconds": 1500}))
    tickets = [kernel_slots.try_acquire(str(task), "gpu", f"lane{i}")
               for i in range(5)]
    assert sum(t is not None for t in tickets) == 2, "issued more than the limit"
    # The CPU pool is independent — a full GPU pool must not block it.
    assert kernel_slots.try_acquire(str(task), "cpu", "lane9") is not None

    held = [t for t in tickets if t][0]
    assert kernel_slots.release(str(task), held) is True
    assert kernel_slots.try_acquire(str(task), "gpu", "lane6") is not None

    # A lane that dies mid-kernel must not park a slot forever.
    ledger = json.loads((task / ".kernel_slots.json").read_text())
    for entry in ledger["gpu"]:
        entry["acquired"] = time.time() - 9999
    (task / ".kernel_slots.json").write_text(json.dumps(ledger))
    assert kernel_slots.status(str(task))["gpu"]["in_use"] == 0


def test_kernel_slots_fails_loud_without_its_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="kernel_slots_config"):
        kernel_slots.try_acquire(str(tmp_path), "gpu", "lane0")


def test_slug_from_url_parses_competition_forms():
    cases = {
        "https://www.kaggle.com/competitions/some-comp/overview": "some-comp",
        "https://www.kaggle.com/competitions/some-comp": "some-comp",
        "https://www.kaggle.com/c/another_comp/data": "another_comp",
        "kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1/rules":
            "ioai-2026-ai-models-track-practice-task-1",
    }
    for url, slug in cases.items():
        assert slug_from_url(url) == slug
    with pytest.raises(ValueError, match="slug"):
        slug_from_url("https://example.com/not-a-competition")


def test_kaggle_mode_config_minimal_knobs():
    with open(CONFIG_PATH) as f:
        mode = yaml.safe_load(f)["modes"]["KAGGLE"]
    assert "kaggle" not in mode  # slug comes from the run root, not config
    assert "contest_economics" not in mode
    assert mode["budget"] == {"min_iteration_seconds": 900}
