# Kapso facade — learn() / memory / serving-hook contracts
# (learn-api-design.md §§1-8; Rule 9: each test names its regression).

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kapso.core.preflight import PreflightError
from kapso.execution.solution import SolutionResult
from kapso.kapso import Kapso
from kapso.learning.bank_remote import connect_bank
from kapso.learning.lesson_result import LessonResult, MemoryStatus
from kapso.learning.update_frame import init_bank
from tests.test_bank_retriever import card_text
from tests.test_update_frame import seed_bank_home


def facade_config(tmp_path) -> str:
    """A minimal real config file exercising the actual load path."""
    config = {
        "default_mode": "GENERIC",
        "defaults": {"models": {"embedding": "e"},
                     "retry": {"request_timeout_seconds": 600}},
        "modes": {"GENERIC": {"knowledge_search": {
            "type": "kg_graph_search", "enabled": True, "params": {}}}},
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"),
                                 "remote": None},
            "import_report_dir": str(tmp_path / "imports"),
            "status_dir": str(tmp_path / "status"),
            "harvest": {"enabled": True},
            "serving": {"enabled": False},
            "bank": {"local_path": str(tmp_path / "bank-home.git")},
            "retriever": {"probe_budget": 1},
            "graders": {"run_root": str(tmp_path / "graders")},
            "update_crew": {"default_version": "crew_test",
                            "run_root": str(tmp_path / "update")},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def make_campaign_dir(tmp_path) -> Path:
    """A bare evolve-workspace-shaped campaign dir."""
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    (campaign / ".kapso" / "experiment_history.json").write_text(
        json.dumps([{"branch": "exp-1", "score": 0.5,
                     "summary": "baseline experiment"}])
    )
    (campaign / "train.py").write_text("print('model')\n")
    return campaign


def stub_chain(monkeypatch, bank_home, lesson_commit_message=None):
    """Stub the heavy crew frames; the store/bank/git layers stay REAL.

    The update stub commits one insight card to the bank home when
    lesson_commit_message is set — so admitted/diff bookkeeping runs
    against real git, not a mock.
    """
    calls = SimpleNamespace(mined=[], exams=[], updates=[])

    class FakeMining:
        @classmethod
        def from_config(cls, config):
            fake = cls()
            fake._config = config
            return fake

        def mine(self, trajectory_id, force=False):
            calls.mined.append(trajectory_id)
            # Stamp the manifest as the real miner does, so learn()'s
            # mined-once idempotency is exercised for real.
            from kapso.learning.trajectory_store import TrajectoryStore
            store = TrajectoryStore.from_config(self._config)
            bundle = store.local / trajectory_id
            (bundle / "mined").mkdir(exist_ok=True)
            manifest = yaml.safe_load((bundle / "trajectory.yaml").read_text())
            manifest["derived"] = {"mined": {"generated": "test"}}
            (bundle / "trajectory.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False)
            )
            return f"mined/{trajectory_id}"

    class FakeGrading:
        def __init__(self, store, config):
            pass

        def grade_exam(self, trajectory_id, bank_dir, bank_head,
                       run_root, learn_set_ids):
            calls.exams.append(
                {"trajectory": trajectory_id, "bank_head": bank_head}
            )
            report = Path(run_root) / "exam" / "report.md"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("---\nserving: null\n---\n")
            return report

    class FakeUpdate:
        def __init__(self, store, config):
            pass

        def run_update(self, batch, run_root, learner_version):
            calls.updates.append(
                {"batch": batch, "version": learner_version}
            )
            if lesson_commit_message:
                work = Path(run_root) / "bank-work"
                if work.exists():
                    subprocess.run(["rm", "-rf", str(work)], check=True)
                subprocess.run(
                    ["git", "clone", "--quiet", str(bank_home), str(work)],
                    check=True,
                )
                card = work / "insights" / "facade-test-card.md"
                card.parent.mkdir(exist_ok=True)
                card.write_text(card_text("facade-test-card"))
                subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
                subprocess.run(
                    ["git", "-C", str(work), "-c", "user.email=t@t",
                     "-c", "user.name=t", "commit", "-q", "-m",
                     lesson_commit_message],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(work), "push", "-q", "origin", "main"],
                    check=True,
                )
            run_dir = Path(run_root) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "report.md").write_text("lesson\n")
            return run_dir

    monkeypatch.setattr("kapso.kapso.MiningFrame", FakeMining)
    monkeypatch.setattr("kapso.kapso.GradingFrame", FakeGrading)
    monkeypatch.setattr("kapso.kapso.UpdateFrame", FakeUpdate)
    return calls


def test_learn_dir_dispatch_exam_pin_and_admitted(tmp_path, monkeypatch):
    # Regression: the full learn() chain — a bare campaign dir is harvested
    # under the historical contract (facade-synthesized meta/report/log),
    # the exam grades the PRE-lesson head (the pin), and admitted derives
    # from real git heads with the created card named.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    kapso = Kapso(config_path=config_path)
    calls = stub_chain(monkeypatch, kapso._bank_home,
                       lesson_commit_message="lesson: facade test card")
    head_before = kapso._bank_head()

    lesson = kapso.learn(str(make_campaign_dir(tmp_path)),
                         trajectory_id="facade-task/20260824T000000_t1")

    assert isinstance(lesson, LessonResult)
    assert calls.mined == ["facade-task/20260824T000000_t1"]
    # exam pin: graded head == the head BEFORE the lesson commit
    assert calls.exams[0]["bank_head"] == head_before
    assert lesson.bank_head_before == head_before
    assert lesson.admitted and lesson.bank_head_after != head_before
    assert lesson.cards_created == ["facade-test-card"]
    assert lesson.metadata["learner_version"] == "crew_test"
    assert lesson.metadata["pushed"] is False  # no remote configured
    assert "facade-test-card" in lesson.explain()
    # the synthesized historical-contract files landed in the bundle
    store_bundle = tmp_path / "store" / "facade-task" / "20260824T000000_t1"
    assert (store_bundle / "campaign_meta.json").is_file()
    assert (store_bundle / "final_report.json").is_file()


def test_learn_store_id_skips_import_and_no_admission_is_honest(
    tmp_path, monkeypatch
):
    # Regression: a store id dispatches WITHOUT re-import; a lesson that
    # moves nothing reports admitted=False with empty card lists.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    kapso = Kapso(config_path=config_path)
    calls = stub_chain(monkeypatch, kapso._bank_home)  # update commits nothing

    first = kapso.learn(str(make_campaign_dir(tmp_path)),
                        trajectory_id="facade-task/20260824T000000_t1")
    assert first.admitted is False and first.cards_created == []

    again = kapso.learn("facade-task/20260824T000000_t1")
    assert again.trajectory_id == "facade-task/20260824T000000_t1"
    # mined once only (the view exists), no second harvest happened
    assert calls.mined == ["facade-task/20260824T000000_t1"]


def test_learn_auto_inits_missing_bank_then_fails_on_bad_source(tmp_path):
    # Regression (onboarding E2E finding #2): a missing bank home is not a
    # setup error — learn() founds it automatically, so the README's
    # evolve→learn loop needs zero bank ceremony. The bogus source still
    # fails loud, but only AFTER the bank exists.
    config_path = facade_config(tmp_path)
    kapso = Kapso(config_path=config_path)
    assert not kapso._bank_home.exists()
    with pytest.raises(FileNotFoundError, match="neither a store"):
        kapso.learn(str(tmp_path / "nowhere"))
    assert (kapso._bank_home / "HEAD").is_file()  # bare repo founded
    assert kapso._bank_head()  # skeleton commit landed


def test_learn_push_preflight_fires_before_any_pipeline_work(
    tmp_path, monkeypatch
):
    # Regression (onboarding E2E finding #1): an unreachable push
    # destination must kill learn() in seconds at the START — never after
    # hours of crew work at the final push. push=False opts out and runs
    # local-only.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    kapso = Kapso(config_path=config_path)
    calls = stub_chain(monkeypatch, kapso._bank_home)
    subprocess.run(
        ["git", "--git-dir", str(kapso._bank_home), "remote", "add",
         "origin", str(tmp_path / "no-such-remote.git")],
        check=True,
    )
    campaign = make_campaign_dir(tmp_path)
    with pytest.raises(PreflightError) as excinfo:
        kapso.learn(str(campaign))
    assert calls.mined == []  # preflight fired before harvest/mine
    # The report IS the message, and it has to be actionable: git's own
    # reason, plus how to fix access or detach the remote.
    report = str(excinfo.value)
    assert "bank remote reachable" in report
    assert "does not appear to be a git repository" in report
    assert "remote remove origin" in report

    lesson = kapso.learn(str(campaign), push=False,
                         trajectory_id="facade-task/20260824T000000_t9")
    assert lesson.metadata["pushed"] is False


def test_learn_pushes_to_connected_origin(tmp_path, monkeypatch):
    # Regression: with an origin attached (kapso bank connect), learn()
    # pushes main+tags there by default and the share remote ends at the
    # post-lesson head — the multi-machine share contract.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    kapso = Kapso(config_path=config_path)
    stub_chain(monkeypatch, kapso._bank_home,
               lesson_commit_message="lesson: pushed card")
    share = tmp_path / "share.git"
    subprocess.run(["git", "init", "--bare", "-q", str(share)], check=True)
    connect_bank(kapso._bank_home, str(share))

    lesson = kapso.learn(str(make_campaign_dir(tmp_path)),
                         trajectory_id="facade-task/20260824T000000_t2")

    assert lesson.metadata["pushed"] is True
    remote_head = subprocess.run(
        ["git", "--git-dir", str(share), "rev-parse", "main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert remote_head == lesson.bank_head_after


def test_close_releases_backend_and_registers_atexit(tmp_path, monkeypatch):
    # Regression (onboarding E2E finding #8): the facade owns network
    # clients through knowledge_search — close() must release them, and
    # construction must register close() atexit so a plain script exits
    # without unclosed-socket ResourceWarnings.
    import kapso.kapso as facade_module

    registered = []
    monkeypatch.setattr(facade_module.atexit, "register",
                        lambda fn: registered.append(fn))
    kapso = Kapso(config_path=facade_config(tmp_path))
    assert kapso.close in registered
    closed = []
    kapso.knowledge_search = SimpleNamespace(close=lambda: closed.append(True))
    kapso.close()
    assert closed == [True]


def test_bank_constructor_override_beats_config(tmp_path):
    config_path = facade_config(tmp_path)
    other = tmp_path / "other-bank.git"
    kapso = Kapso(config_path=config_path, bank=str(other))
    assert kapso._bank_home == other


def test_memory_status_shapes(tmp_path):
    # Regression: memory reports both stores truthfully — before a bank
    # exists (uninitialized) and after (head + card count).
    config_path = facade_config(tmp_path)
    kapso = Kapso(config_path=config_path)
    status = kapso.memory
    assert isinstance(status, MemoryStatus)
    assert status.knowledge_enabled is False
    assert status.bank_head is None
    assert "not initialized" in status.explain()

    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    status = Kapso(config_path=config_path).memory
    assert status.bank_head is not None
    assert status.bank_active_cards == 1
    assert "1 active cards" in status.explain()


def test_evolve_serving_staging_and_disabled_byte_identity(
    tmp_path, monkeypatch
):
    # Regression (design §4): serving on + bank present -> the intro rides
    # the problem context, bank_serving reaches the orchestrator, and the
    # provenance stamp lands; serving off -> context byte-identical to the
    # pre-serving facade and no override is passed.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})

    captured = {}

    class FakeOrchestrator:
        # What gates the strategy resolved — the real one appends "bank"
        # when bank_serving is in its params; tests flip this to prove the
        # facade refuses to inject an intro over an unmounted gate.
        gates = ["research", "repo_memory", "bank"]

        def __init__(self, handler, **kwargs):
            captured["handler"] = handler
            captured["kwargs"] = kwargs
            self.search_strategy = SimpleNamespace(
                workspace=SimpleNamespace(workspace_dir=str(tmp_path / "ws")),
                checkout_to_best_experiment_branch=lambda: "main",
                get_deliverable_score=lambda: None,
                get_experiment_history=lambda best_last=True: [],
                iteration_evaluations=[],
                ideation_gates=list(FakeOrchestrator.gates),
            )
            self.operation_status = SimpleNamespace(
                path=str(tmp_path / "ws" / ".kapso" / "status.json")
            )

        def solve(self, **kwargs):
            return SimpleNamespace(
                final_feedback=None, iterations_run=0,
                cumulative_iterations=0, total_cost=0.0,
                stopped_reason="max_iterations", stop_detail=None,
                best_experiment=None,
            )

    monkeypatch.setattr("kapso.kapso.OrchestratorAgent", FakeOrchestrator)
    monkeypatch.setattr(
        Kapso, "_extract_experiment_logs", lambda self, orch: [], raising=True
    )
    monkeypatch.chdir(tmp_path)

    # serving disabled: no intro, no override
    kapso = Kapso(config_path=config_path)
    kapso.evolve(goal="test goal", output_path=str(tmp_path / "out1"))
    off_context = captured["handler"].additional_context
    assert "Knowledge bank" not in off_context
    assert captured["kwargs"]["strategy_params_overrides"] is None

    # serving enabled: intro + override + stamp
    config = yaml.safe_load(Path(config_path).read_text())
    config["learning"]["serving"]["enabled"] = True
    Path(config_path).write_text(yaml.safe_dump(config))
    kapso = Kapso(config_path=config_path)
    solution = kapso.evolve(goal="test goal",
                            output_path=str(tmp_path / "out2"))
    on_context = captured["handler"].additional_context
    assert "## Knowledge bank" in on_context
    assert "bank_index()" in on_context
    override = captured["kwargs"]["strategy_params_overrides"]
    assert "KAPSO_BANK_DIR" in override["bank_serving"]
    assert solution.metadata["bank_head_served"] is not None
    # the serving record landed inside the campaign workspace
    record = tmp_path / "out2" / ".kapso" / "serving" / "serving-record.yaml"
    assert record.is_file()


def test_learn_knowledge_flattens_research_output(tmp_path, monkeypatch):
    # Regression (E2E 2026-08-24): research(mode="idea") returns a LIST of
    # typed sources; passing it directly (the advertised contract) must
    # reach the pipeline flattened — never as one 'list'-typed source.
    config_path = facade_config(tmp_path)
    kapso = Kapso(config_path=config_path)
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, *sources, skip_merge=False, status=None):
            captured["sources"] = sources
            return SimpleNamespace(
                sources_processed=len(sources), total_pages_extracted=1,
                created=1, edited=0, errors=[],
            )

    monkeypatch.setattr("kapso.kapso.KnowledgePipeline", FakePipeline)
    idea_a, idea_b = object(), object()
    kapso.learn_knowledge([idea_a, idea_b], skip_merge=True)
    assert captured["sources"] == (idea_a, idea_b)
    with pytest.raises(ValueError, match="empty source lists"):
        kapso.learn_knowledge([], skip_merge=True)


def test_evolve_refuses_to_advertise_unmounted_bank_tools(tmp_path, monkeypatch):
    # Regression (E2E review 2026-08-24, blocker 1): the injected intro
    # instructs sessions to call bank_index / bank_get_card /
    # bank_get_card_with_evidence. If the gate providing them did not
    # resolve, the intro lies and every pull log stays empty — the facade
    # must fail loud instead.
    config_path = facade_config(tmp_path)
    seed_bank_home(tmp_path, {"seed-card": card_text("seed-card")})
    config = yaml.safe_load(Path(config_path).read_text())
    config["learning"]["serving"]["enabled"] = True
    Path(config_path).write_text(yaml.safe_dump(config))

    class GatelessOrchestrator:
        def __init__(self, handler, **kwargs):
            self.search_strategy = SimpleNamespace(
                workspace=SimpleNamespace(workspace_dir=str(tmp_path / "ws")),
                checkout_to_best_experiment_branch=lambda: "main",
                get_deliverable_score=lambda: None,
                get_experiment_history=lambda best_last=True: [],
                iteration_evaluations=[],
                ideation_gates=["research", "repo_memory"],  # no "bank"
            )

        def solve(self, **kwargs):  # never reached
            raise AssertionError("solve must not run with an unmounted gate")

    monkeypatch.setattr("kapso.kapso.OrchestratorAgent", GatelessOrchestrator)
    monkeypatch.chdir(tmp_path)
    kapso = Kapso(config_path=config_path)
    with pytest.raises(RuntimeError, match="'bank' gate is not mounted"):
        kapso.evolve(goal="test goal", output_path=str(tmp_path / "out3"))


def test_learn_knowledge_records_the_index_it_wrote(tmp_path, monkeypatch):
    # Regression (E2E review 2026-08-24): kg_index provenance was None on
    # every learn_knowledge -> evolve sequence because the facade never
    # recorded the index the merge wrote.
    config_path = facade_config(tmp_path)
    kapso = Kapso(config_path=config_path)
    wiki_dir = tmp_path / "wikis"
    wiki_dir.mkdir()
    (wiki_dir / ".index").write_text('{"page_count": 3}')

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, *sources, skip_merge=False, status=None):
            return SimpleNamespace(
                sources_processed=1, total_pages_extracted=3,
                created=3, edited=0, errors=[],
            )

    monkeypatch.setattr("kapso.kapso.KnowledgePipeline", FakePipeline)
    monkeypatch.setattr(
        "kapso.kapso.KnowledgeSearchFactory",
        SimpleNamespace(create=lambda **kw: SimpleNamespace(
            is_enabled=lambda: True), create_null=lambda: None),
    )
    kapso.learn_knowledge(object(), wiki_dir=str(wiki_dir))
    assert kapso._kg_index_path == str(wiki_dir / ".index")
    assert kapso.memory.knowledge_index == str(wiki_dir / ".index")
