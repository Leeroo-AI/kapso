"""Contract tests for the Kaggle code-competition benchmark.

What must hold: the handler renders the metered-submission economics from
config values (never literals), the insured predicate reads only >0 banked
public scores and fails loud on corruption, the runner's leaderboard parsing
survives CLI pagination noise and windows submissions to the run, and the
preparer builds the exact layout the runner requires.
"""

import json
import os
import time

import pytest
import yaml

from benchmarks.kaggle.data.prepare_task1 import prepare
from benchmarks.kaggle.handler import KaggleNotebookHandler
from benchmarks.kaggle.runner import (
    audit_kernel,
    best_public_score,
    parse_submissions_json,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "kaggle", "config.yaml"
)

SESSION_CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}
ECONOMICS = {
    "insurance_minutes": 40,
    "confirm_gain_ratio": 2.0,
    "insured_freeze_minutes": 10,
}
KAGGLE = {
    "competition": "ioai-2026-ai-models-track-practice-task-1",
    "daily_submission_cap": 5,
    "run_submission_cap": 3,
}


def make_handler(tmp_path, **overrides):
    kwargs = dict(
        task_dir=str(tmp_path / "task"),
        statement="statement body",
        deadline_ts=time.time() + 7200,
        session_caps=SESSION_CAPS,
        contest_economics=ECONOMICS,
        kaggle=KAGGLE,
    )
    kwargs.update(overrides)
    return KaggleNotebookHandler(**kwargs)


def test_handler_renders_submission_economics_from_config_values(tmp_path):
    context = make_handler(tmp_path).get_problem_context()
    assert KAGGLE["competition"] in context
    assert "run's own cap: 3" in context
    assert "daily cap: 5" in context
    assert "~40 minutes" in context
    assert "flock 9" in context
    assert "PUBLIC scores only" in context
    assert "train inside the kernel" in context.lower() or (
        "TRAIN from the provided checkpoint" in context
    )


def test_handler_rejects_missing_kaggle_keys(tmp_path):
    with pytest.raises(ValueError, match="kaggle"):
        make_handler(tmp_path, kaggle={"competition": "x"})
    with pytest.raises(ValueError, match="kaggle"):
        make_handler(
            tmp_path,
            kaggle={"daily_submission_cap": 5, "run_submission_cap": 3},
        )


def test_insured_predicate_needs_kernel_and_positive_public_score(tmp_path):
    handler = make_handler(tmp_path)
    assert handler.deliverable_ready_reserve_seconds() is None

    kernel_dir = tmp_path / "task" / "submission" / "kernel"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "script.py").write_text("print('hi')\n")
    assert handler.deliverable_ready_reserve_seconds() is None

    log = tmp_path / "task" / "best_score.log"
    log.write_text("0.0 2026-07-22T15:00:00 insurance-placeholder\n")
    assert handler.deliverable_ready_reserve_seconds() is None

    log.write_text(
        "0.0 2026-07-22T15:00:00 placeholder\n"
        "0.41 2026-07-22T15:30:00 kernel=u/s:v2 baseline\n"
    )
    assert handler.deliverable_ready_reserve_seconds() == 600.0


def test_insured_predicate_raises_on_corrupt_bank_line(tmp_path):
    handler = make_handler(tmp_path)
    kernel_dir = tmp_path / "task" / "submission" / "kernel"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "script.py").write_text("print('hi')\n")
    (tmp_path / "task" / "best_score.log").write_text("not-a-score today\n")
    with pytest.raises(ValueError):
        handler.deliverable_ready_reserve_seconds()


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


def test_prepare_builds_runner_layout(tmp_path):
    source = tmp_path / "src"
    (source / "audio").mkdir(parents=True)
    (source / "audio" / "a.wav").write_bytes(b"RIFF")
    (source / "model").mkdir()
    for name in ("config.json", "model.safetensors",
                 "preprocessor_config.json"):
        (source / "model" / name).write_text("{}")
    (source / "train.csv").write_text(
        "path,split,target,category\naudio/a.wav,train,0,Dog\n")
    (source / "fine_tune.csv").write_text(
        "path,split,target,category\naudio/a.wav,train,16,Axe\n")
    (source / "submission.csv").write_text("path,target\naudio/a.wav,0\n")

    root = prepare(str(tmp_path / "root"), str(source), "some-competition")

    dataset = os.path.join(root, "task", "dataset")
    for entry in ("audio/a.wav", "model/config.json", "train.csv",
                  "fine_tune.csv", "submission.csv", "statement.md"):
        assert os.path.exists(os.path.join(dataset, entry)), entry
    with open(os.path.join(root, "task", "kaggle.json")) as f:
        assert json.load(f) == {"competition": "some-competition"}


def test_prepare_rejects_incomplete_source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="audio"):
        prepare(str(tmp_path / "root"), str(source), "c")


def test_kaggle_mode_config_carries_submission_caps():
    with open(CONFIG_PATH) as f:
        mode = yaml.safe_load(f)["modes"]["KAGGLE"]
    assert mode["kaggle"] == {"daily_submission_cap": 5,
                              "run_submission_cap": 3}
    assert {"insurance_minutes", "confirm_gain_ratio",
            "insured_freeze_minutes"} <= mode["contest_economics"].keys()
    assert mode["budget"]["min_iteration_seconds_insured"] == 300
