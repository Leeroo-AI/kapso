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
from benchmarks.kaggle.preflight import SPEC_PATH, slug_from_url
from benchmarks.kaggle.runner import (
    RULES_PATH,
    audit_kernel,
    banked_kernel_refs,
    best_public_score,
    classify_submit_output,
    discover_run_kernels,
    parse_submissions_json,
    rank_harvest_candidates,
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
    (task_dir / "KAGGLE_CLI.md").write_text("playbook body")
    kwargs = dict(
        task_dir=str(tmp_path / "task"),
        statement="statement body",
        deadline_ts=time.time() + 7200,
        session_caps=SESSION_CAPS,
        kaggle=KAGGLE,
        insured_reserve_seconds=300.0,
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
    # Ideas and code come off Kaggle; best_score.log carries only what is
    # actually banked, which is what releases the finalization reserve.
    assert "kaggle competitions submissions" in context
    assert "kernels pull" in context
    assert "KAGGLE_CLI.md" in context
    assert "best_score.log" in context and "public scores only" in context
    assert "<score>" in context
    # The protocol/economics sermons must stay gone.
    for banned in ("SUBMISSION BUDGET", "INSURANCE", "flock",
                   "Reward & time economics", "push TWICE"):
        assert banned not in context
    # Budget the contract we author, not the environment it renders in: the
    # statement is unbounded and the task dir is 37 chars in production but
    # ~140 under pytest, which used to make this guard measure tmp_path. The
    # ceiling moves only when a channel is deliberately added (the banked-score
    # log alongside the Kaggle history commands) — never to fit new prose.
    contract = context.split("# Kapso operational context", 1)[1]
    # 2350 + the TEST-ONLY web-search paragraph (see handler.get_problem_context);
    # restore to 2350 when that paragraph is removed for the real competition.
    assert len(contract.replace(handler.task_dir, "/task")) < 2500


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
    os.remove(tmp_path / "task" / "KAGGLE_CLI.md")
    with pytest.raises(FileNotFoundError, match="KAGGLE_CLI.md"):
        KaggleNotebookHandler(
            task_dir=str(tmp_path / "task"), statement="s",
            deadline_ts=time.time() + 60, session_caps=SESSION_CAPS,
            kaggle=KAGGLE, insured_reserve_seconds=300.0,
        )
    os.remove(tmp_path / "task" / "RULES.md")
    with pytest.raises(FileNotFoundError, match="RULES.md"):
        KaggleNotebookHandler(
            task_dir=str(tmp_path / "task"), statement="s",
            deadline_ts=time.time() + 60, session_caps=SESSION_CAPS,
            kaggle=KAGGLE, insured_reserve_seconds=300.0,
        )


def test_staged_rules_carry_the_binding_kernel_constraints():
    # The handler deliberately states no kernel mechanics, so RULES.md is the
    # only place the agent learns them: a trim that drops the one-GPU pin or the
    # never-P100 pin would silently void submissions that still score.
    rules = open(RULES_PATH, encoding="utf-8").read()
    assert "cuda:0" in rules
    assert "NvidiaTeslaT4" in rules and "P100" in rules
    assert "50 submissions per task" in rules


def test_preflight_spec_authors_no_submit_mechanics():
    # S2 (run 5): the spec used to template the file-upload submit command,
    # the preflight copied it into a kernel task's statement, and six lanes
    # burned their first submission on a 400. The statement carries modality
    # and file shape only; CLI mechanics never come from preflight authorship.
    spec = open(SPEC_PATH, encoding="utf-8").read()
    assert "modality" in spec
    assert "competitions submit" not in spec
    assert "kernels push" not in spec


def test_runner_stages_the_real_cli_playbook():
    # The statement authors no CLI mechanics and the handler hard-requires
    # KAGGLE_CLI.md, so the runner's staging source must exist and must be the
    # actual playbook — a moved/renamed skill would otherwise strand every
    # lane with no submit mechanics at all.
    from benchmarks.kaggle.runner import SKILL_PATH
    assert os.path.isfile(SKILL_PATH)
    skill = open(SKILL_PATH, encoding="utf-8").read()
    assert "kernels push" in skill and "competitions submit" in skill


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


def test_reserve_is_insured_only_once_a_public_score_is_banked(tmp_path):
    # The full reserve covers one submission round trip; it is released only
    # when a score is actually on the leaderboard, never merely attempted.
    handler = make_handler(tmp_path)
    assert handler.deliverable_ready_reserve_seconds() is None
    log = os.path.join(handler.task_dir, "best_score.log")
    with open(log, "w") as f:
        f.write("0.0 2026-07-29T16:00:00Z placeholder\n")
    assert handler.deliverable_ready_reserve_seconds() is None
    with open(log, "a") as f:
        f.write("0.83626 2026-07-29T16:20:00Z lane0\n")
    assert handler.deliverable_ready_reserve_seconds() == 300.0


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


def make_slots_task(tmp_path, gpu=2, cpu=5, ttl=600):
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / ".kernel_slots_config.json").write_text(json.dumps(
        {"gpu": gpu, "cpu": cpu, "ttl_seconds": ttl,
         "reap_interval_seconds": 60, "verify_timeout_seconds": 30}))
    return str(task)


def grab(task, pool="gpu", kind="push", lane="lane", priority=None,
         waiter_id=None, now=None):
    """One poll_once step — the atomic unit acquire_blocking loops on."""
    return kernel_slots.poll_once(
        task, pool, kind, "owner/kernel-a", lane,
        priority or kernel_slots.DEFAULT_PRIORITY[kind],
        waiter_id or f"w-{lane}", now if now is not None else time.time())


def test_kernel_slots_pushes_and_scores_share_one_pool(tmp_path):
    # The account limit counts SESSIONS: a scoring re-run occupies a slot
    # exactly like a push (run 5's harvest filled the pool with its own two
    # submissions and misread 72 rejections as dead kernels).
    task = make_slots_task(tmp_path)
    assert grab(task, kind="push", lane="a")["ticket"]
    assert grab(task, kind="score", lane="b")["ticket"]
    assert grab(task, kind="push", lane="c")["ticket"] is None
    # One queue PER POOL: a full GPU pool must not block the CPU pool.
    assert grab(task, pool="cpu", lane="d")["ticket"]


def test_kernel_slots_priority_queue_orders_grants(tmp_path):
    # ship > first-score > run > reroll, FIFO within a tier — a freed slot
    # goes to the highest tier, not to whoever happens to poll first.
    task = make_slots_task(tmp_path, gpu=1)
    held = grab(task, lane="holder")["ticket"]
    t0 = time.time()
    assert grab(task, lane="早", priority="run", waiter_id="w-run",
                now=t0)["ticket"] is None
    assert grab(task, kind="score", lane="ship", priority="ship",
                waiter_id="w-ship", now=t0 + 1)["ticket"] is None
    kernel_slots.release(task, held)
    # The earlier-arrived run waiter polls first but must NOT jump the queue.
    assert grab(task, lane="早", priority="run", waiter_id="w-run",
                now=t0 + 2)["ticket"] is None
    assert grab(task, kind="score", lane="ship", priority="ship",
                waiter_id="w-ship", now=t0 + 3)["ticket"]
    # With the slot retaken, the run waiter keeps queueing.
    assert grab(task, lane="早", priority="run", waiter_id="w-run",
                now=t0 + 4)["ticket"] is None


def test_kernel_slots_grants_top_k_and_prunes_dead_waiters(tmp_path):
    task = make_slots_task(tmp_path)
    t0 = time.time()
    held = [grab(task, lane=f"h{i}", waiter_id=f"w-h{i}")["ticket"]
            for i in range(2)]
    # Three lanes queue on the full pool; the first then dies mid-wait (its
    # heartbeat goes stale) and must not hold a place in line.
    assert grab(task, lane="dead", waiter_id="w-dead", now=t0 - 120)["ticket"] is None
    assert grab(task, lane="b", waiter_id="w-b", now=t0)["ticket"] is None
    assert grab(task, lane="c", waiter_id="w-c", now=t0 + 1)["ticket"] is None
    for ticket in held:
        kernel_slots.release(task, ticket)
    # Two slots free -> BOTH fresh waiters grant (rank < free), regardless of
    # poll order — a slow-polling head cannot strand the second slot — and
    # the dead waiter's stale entry blocks neither.
    assert grab(task, lane="c", waiter_id="w-c", now=t0 + 2)["ticket"]
    assert grab(task, lane="b", waiter_id="w-b", now=t0 + 3)["ticket"]


def test_kernel_slots_reap_releases_only_kaggle_confirmed_dead(tmp_path):
    # TTL alone must NOT reclaim: a dead lane's kernel keeps RUNNING on
    # Kaggle, and blind reclaim over-granted the pool (run 5's ticket-holding
    # lanes still hit the session cap). Truth comes from Kaggle.
    task = make_slots_task(tmp_path)
    running = grab(task, kind="push", lane="zombie")["ticket"]
    scoring = grab(task, kind="score", lane="scored")["ticket"]
    ledger_path = os.path.join(task, ".kernel_slots.json")
    ledger = json.loads(open(ledger_path).read())
    for entry in ledger["tickets"]["gpu"]:
        entry["acquired"] = time.time() - 9999
    open(ledger_path, "w").write(json.dumps(ledger))

    released = kernel_slots.maybe_reap(
        task, time.time(),
        kernel_status_fn=lambda ref, timeout: "running",
        pending_fn=lambda task_dir, timeout: 0)
    assert released == [scoring]          # nothing pending -> score slot free
    assert running not in released        # kernel still running -> keep it

    # Rate limit: an immediate second reap must not hammer Kaggle.
    assert kernel_slots.maybe_reap(
        task, time.time(),
        kernel_status_fn=lambda ref, timeout: "complete",
        pending_fn=lambda task_dir, timeout: 0) == []
    # Past the interval, the terminal kernel's ticket is released too.
    assert kernel_slots.maybe_reap(
        task, time.time() + 61,
        kernel_status_fn=lambda ref, timeout: "complete",
        pending_fn=lambda task_dir, timeout: 0) == [running]
    assert kernel_slots.status(task)["gpu"]["in_use"] == 0


def test_kernel_slots_fails_loud_without_its_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="kernel_slots_config"):
        grab(str(tmp_path))


def test_classify_submit_output_reads_text_not_exit_codes():
    # Verified live 2026-08-02: the kaggle CLI exits 0 even on a rejected
    # submission, so the output text is the only signal.
    assert classify_submit_output("", "403 Client Error: Forbidden for url: "
                                  ".../CreateCodeSubmission") == "rejected-403"
    assert classify_submit_output("400 Client Error: Bad Request", "") == "rejected-400"
    assert classify_submit_output("", "") == "accepted"
    assert classify_submit_output("Successfully submitted", "") == "accepted"


def test_harvest_ranks_never_scored_kernels_first(tmp_path):
    # Run 5 submitted two already-scored kernels (alphabetical order) while
    # two unscored ones bounced off the full pool and were lost.
    task = tmp_path / "task"
    task.mkdir()
    (task / "best_score.log").write_text(
        "0.85952 2026-08-02T15:22:42Z owner/scored-a lane0 storyboard\n"
        "0.84514 2026-08-02T15:27:33Z - file-upload entry, no kernel\n"
        "bad-line-without-ref\n"
    )
    banked = banked_kernel_refs(str(task))
    assert banked == {"owner/scored-a"}
    refs = ["owner/scored-a", "owner/unscored-b", "owner/unscored-c"]
    assert rank_harvest_candidates(refs, banked) == [
        "owner/unscored-b", "owner/unscored-c", "owner/scored-a"]
    assert banked_kernel_refs(str(tmp_path / "nowhere")) == set()


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
