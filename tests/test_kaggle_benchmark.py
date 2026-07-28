"""Contract tests for the Kaggle code-competition benchmark.

What must hold: the handler context is the statement plus the minimal kapso
contract (and stays free of the removed protocol/economics sermons), the
runner's leaderboard parsing survives CLI pagination noise and windows
submissions to the run, and the preflight parses a competition slug from its
URL (fail-loud on a malformed one).
"""

import os
import time

import pytest
import yaml

from benchmarks.kaggle.handler import KaggleNotebookHandler
from benchmarks.kaggle.preflight import slug_from_url
from benchmarks.kaggle.runner import (
    audit_kernel,
    best_public_score,
    parse_submissions_json,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "kaggle", "config.yaml"
)

SESSION_CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}
KAGGLE = {"competition": "ioai-2026-ai-models-track-practice-task-1"}


def make_handler(tmp_path, **overrides):
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
    context = make_handler(tmp_path).get_problem_context()
    assert context.startswith("statement body")
    assert KAGGLE["competition"] in context
    assert "operator approval" not in context
    # Competition framing, not a safety floor: highest expected score wins.
    assert "highest expected final score" in context
    assert "at least one" not in context
    # End-to-end clock: submission round trips are inside the budget.
    assert "END-TO-END" in context and "round trip" in context
    assert "best_score.log" in context and "public scores only" in context
    assert "<score>" in context
    # The protocol/economics sermons must stay gone.
    for banned in ("SUBMISSION BUDGET", "INSURANCE", "flock",
                   "Reward & time economics", "push TWICE"):
        assert banned not in context
    assert len(context) < 2500


def test_handler_rejects_missing_kaggle_slug(tmp_path):
    with pytest.raises(ValueError, match="kaggle"):
        make_handler(tmp_path, kaggle={})


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
