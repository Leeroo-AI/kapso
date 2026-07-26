"""Posttrain knowledge seeding: the optional prior-run-learnings offer.

The runner pre-seeds knowledge/<benchmark_id>.md into the campaign shared
cache so sessions get the standing OPTIONAL offer. The contract worth
pinning: the registry it writes must validate against the strategy-side
loader (a malformed entry would crash every session's brief render), the
seed must be idempotent across resumes, and a missing doc is the documented
"nothing seeded" default.
"""

import json

import pytest

from benchmarks.posttrain import runner as posttrain_runner
from kapso.execution.search_strategies.generic.shared_cache import (
    load_artifact_registry,
    render_artifacts_brief,
)


@pytest.fixture()
def knowledge_dir(tmp_path, monkeypatch):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "arenahardwriting.md").write_text(
        "# lessons\n- ship rp 1.05\n", encoding="utf-8"
    )
    monkeypatch.setattr(posttrain_runner, "KNOWLEDGE_DIR", str(kdir))
    return kdir


def test_seed_copies_doc_and_registers_valid_offer(tmp_path, knowledge_dir):
    workspace = tmp_path / "kapso_campaign"

    assert posttrain_runner.seed_benchmark_knowledge(
        str(workspace), "arenahardwriting"
    ) is True

    cache_dir = workspace / ".kapso" / "shared_cache"
    seeded = cache_dir / posttrain_runner.KNOWLEDGE_ARTIFACT_FILENAME
    assert "ship rp 1.05" in seeded.read_text(encoding="utf-8")

    # The strategy-side loader must accept what the runner wrote, and the
    # rendered brief must carry the optional-offer framing + the doc.
    entries = load_artifact_registry(cache_dir)
    assert [e["name"] for e in entries] == [
        "prior-run-learnings-arenahardwriting"
    ]
    brief = render_artifacts_brief(cache_dir, entries)
    assert "OFFER, not an instruction" in brief
    assert "prior-run-learnings-arenahardwriting" in brief
    assert "MISSING" not in brief


def test_seed_is_idempotent_and_preserves_other_artifacts(
    tmp_path, knowledge_dir
):
    workspace = tmp_path / "kapso_campaign"
    cache_dir = workspace / ".kapso" / "shared_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "teacher.jsonl").write_text("{}\n", encoding="utf-8")
    (cache_dir / "artifacts.json").write_text(
        json.dumps([
            {
                "name": "teacher-traces",
                "path": "teacher.jsonl",
                "description": "distill traces from a prior experiment",
            }
        ]),
        encoding="utf-8",
    )

    posttrain_runner.seed_benchmark_knowledge(str(workspace), "arenahardwriting")
    posttrain_runner.seed_benchmark_knowledge(str(workspace), "arenahardwriting")

    entries = load_artifact_registry(cache_dir)
    assert sorted(e["name"] for e in entries) == [
        "prior-run-learnings-arenahardwriting",
        "teacher-traces",
    ]


def test_no_knowledge_doc_seeds_nothing(tmp_path, knowledge_dir):
    workspace = tmp_path / "kapso_campaign"

    assert posttrain_runner.seed_benchmark_knowledge(
        str(workspace), "gpqamain"
    ) is False
    assert not (workspace / ".kapso" / "shared_cache" / "artifacts.json").exists()
