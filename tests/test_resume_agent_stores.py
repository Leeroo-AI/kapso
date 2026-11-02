"""After a resume, what agents read must match what the checkpoint says happened.

A round that dies before its checkpoint is redone under the same node id. These
tests run the real orchestrator over the real GenericSearch, the real history
store and the real evaluation maintainer, with only the paid sessions faked,
and kill the round between the moment it wrote to a store and its checkpoint.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_support import fake_config

import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
import kapso.execution.memories.experiment_memory.store as store_module
import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.evaluation_maintainer.registry import EvaluationRegistry
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.inbox import (
    file_requests,
    load_requests,
    new_requests_for_session,
    record_reply,
    withdraw_orphaned_requests,
)
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from kapso.gated_mcp.gates.experiment_history_gate import ExperimentHistoryGate

from tests.test_campaign_spend_durability import (
    Campaign,
    ENTRY,
    FEEDBACK_USD,
    IDEATION_USD,
    SimulatedKill,
)

CHANGE_REQUEST = "<evaluation_change_request>timeout bug</evaluation_change_request>"
GHOST_SCORE = 0.99
REDO_SCORE = 0.2


def raise_kill(evaluation: Path) -> None:
    raise SimulatedKill()


class MaintainerSession:
    """The paid maintainer session. Setup writes the entrypoint; a change
    request runs ``during_change_request`` and answers with ``verdict``."""

    def __init__(self, owner: "AgentStoreCampaign"):
        self.owner = owner

    def initialize(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    def generate_code(self, prompt: str):
        evaluation = self.workspace / "kapso_evaluation"
        if (evaluation / "kapso_eval.py").exists():
            self.owner.during_change_request(evaluation)
            return SimpleNamespace(success=True, output=self.owner.verdict, error=None)
        evaluation.mkdir(exist_ok=True)
        (evaluation / "kapso_eval.py").write_text("ENTRYPOINT = True\n")
        return SimpleNamespace(success=True, output="done", error=None)

    def get_cumulative_cost(self) -> float:
        return 0.0

    def cleanup(self) -> None:
        pass


class AgentStoreCampaign(Campaign):
    """Node 0 is honest. Node 1 is a different idea each time it is built, so
    the first attempt and its redo cannot be mistaken for each other."""

    def __init__(self, tmp_path, monkeypatch):
        super().__init__(tmp_path, monkeypatch)
        self.attempts = {}
        self.change_request_on = None
        self.during_change_request = lambda evaluation: None
        self.verdict = "<change_verdict>reject</change_verdict><reason>no</reason>"
        self.bridge_score = 0.37

    def with_maintainer(self, monkeypatch) -> "AgentStoreCampaign":
        monkeypatch.setattr(
            orchestrator_module,
            "load_config",
            lambda *args, **kwargs: fake_config(
                search_strategy={"type": "generic", "params": {}},
                evaluation_maintainer={"enabled": True, "max_change_requests": 5},
            ),
        )
        monkeypatch.setattr(
            maintainer_module.CodingAgentFactory,
            "create",
            classmethod(lambda cls, config: MaintainerSession(self)),
        )
        manifest = {
            "fidelity": "fast",
            "fraction": 0.03,
            "seed": 1337,
            "items": 5,
            "total_items": 100,
            "score": 0.4,
        }
        monkeypatch.setattr(
            maintainer_module,
            "subprocess",
            SimpleNamespace(
                run=lambda command, **kwargs: SimpleNamespace(
                    returncode=0,
                    stdout=f"{maintainer_module.MANIFEST_MARKER} {json.dumps(manifest)}\n",
                    stderr="",
                )
            ),
        )
        return self

    def _strategy(self):
        strategy = super()._strategy()
        strategy.run_bridge_evaluation = self._bridge
        return strategy

    def _bridge(self, node, **kwargs) -> bool:
        strategy = self.orchestrator.search_strategy
        node.evaluation_attempts.append(
            EvaluationAttempt(
                commit_sha="sha",
                evaluator_id=strategy.registered_evaluator_id,
                fidelity=kwargs["fidelity"],
                fraction=kwargs["fraction"],
                seed=strategy.registered_subsample_seed,
                score=self.bridge_score,
                duration_seconds=1.0,
            )
        )
        return True

    def _generate_solution(self, problem, parent_branch):
        node_id = len(self.orchestrator.search_strategy.node_history)
        attempt = self.attempts[node_id] = self.attempts.get(node_id, 0) + 1
        return [f"node {node_id} attempt {attempt}"], [], self.bill.charge(IDEATION_USD)

    def _implement(self, **kwargs):
        output, *rest = super()._implement(**kwargs)
        if kwargs["node_id"] == self.change_request_on:
            output = CHANGE_REQUEST
        return (output, *rest)

    def _generate_feedback(self, node):
        node.phase_telemetry["feedback"] = self.bill.charge(FEEDBACK_USD)
        node.score = (
            GHOST_SCORE
            if self.attempts.get(node.node_id) == 1 and node.node_id == 1
            else REDO_SCORE
        )
        node.feedback = f"judged {node.solution}"
        self.during_feedback(node.node_id)
        return node

    @property
    def history_path(self) -> Path:
        return self.workspace / ".kapso" / "experiment_history.json"

    def stored(self):
        return [
            (record["node_id"], record["solution"], record["score"])
            for record in json.loads(self.history_path.read_text())
        ]

    def checkpointed(self):
        return [
            (node["node_id"], node["solution"], node["score"])
            for node in self.checkpoint().strategy_state["node_history"]
            if not node["suspended"]
        ]

    def agents_read(self, tool: str = "get_top_experiments") -> str:
        """What the gate hands a session, opened the way its own process does."""
        gate = ExperimentHistoryGate()
        gate._store = store_module.open_campaign_history(
            str(self.history_path), goal=None, llm=None
        )
        return asyncio.run(gate.handle_call(tool, {"k": 10}))[0].text

    def kill_during_triage_of_node_one(
        self, evaluation_edit=lambda evaluation: None
    ) -> None:
        def die(evaluation: Path) -> None:
            evaluation_edit(evaluation)
            raise SimulatedKill()

        self.change_request_on = 1
        self.during_change_request = die
        with pytest.raises(SimulatedKill):
            self.open().solve(experiment_max_iter=5)
        self.change_request_on = None
        self.during_change_request = lambda evaluation: None


@pytest.fixture
def campaign(tmp_path, monkeypatch) -> AgentStoreCampaign:
    return AgentStoreCampaign(tmp_path, monkeypatch)


@pytest.fixture
def maintained(campaign, monkeypatch) -> AgentStoreCampaign:
    return campaign.with_maintainer(monkeypatch)


# =========================================================================
# The experiment history
# =========================================================================


def test_a_round_killed_before_its_checkpoint_never_reaches_the_agents(maintained):
    maintained.kill_during_triage_of_node_one()

    # The checkpoint has one node, and so does everything agents can read.
    assert [node_id for node_id, _, _ in maintained.checkpointed()] == [0]
    assert maintained.stored() == maintained.checkpointed()

    maintained.open(resume=True).solve(experiment_max_iter=1)

    # The redo replaced the branch of node 1. A score for what that branch
    # used to hold is a score for code nobody can get back.
    assert maintained.checkpointed() == [
        (0, "node 0 attempt 1", REDO_SCORE),
        (1, "node 1 attempt 2", REDO_SCORE),
    ]
    assert maintained.stored() == maintained.checkpointed()
    for tool in ("get_top_experiments", "get_recent_experiments"):
        read = maintained.agents_read(tool)
        assert "node 1 attempt 1" not in read
        assert str(GHOST_SCORE) not in read
        assert read.count("## Experiment 1 ") == 1


def test_a_multi_lane_round_killed_before_its_checkpoint_leaves_no_lane_behind(
    maintained,
):
    maintained.solutions = ["one", "two"]
    maintained._generate_solution = lambda problem, parent: (
        list(maintained.solutions),
        [],
        {},
    )
    maintained.kill_during_triage_of_node_one()

    assert maintained.checkpointed() == []
    assert not maintained.history_path.exists()


def test_a_continuation_killed_before_its_checkpoint_is_recorded_once(maintained):
    maintained.inbox_enabled = True
    maintained.ask_on_node = 0
    assert maintained.open().solve(experiment_max_iter=1).stopped_reason == (
        "waiting_for_user"
    )
    assert not maintained.history_path.exists()
    record_reply(maintained.inbox, 1, "added the key")
    maintained.change_request_on = 0
    maintained.during_change_request = raise_kill
    with pytest.raises(SimulatedKill):
        maintained.open(resume=True).solve(experiment_max_iter=1)

    # The checkpoint still says the node waits, so nothing was recorded for it.
    assert not maintained.history_path.exists()

    maintained.change_request_on = None
    maintained.open(resume=True).solve(experiment_max_iter=1)

    assert [record[0] for record in maintained.stored()] == [0]


def test_a_kill_right_after_the_checkpoint_catches_up_without_a_redo(
    campaign, monkeypatch
):
    real = ExperimentHistoryStore.reconcile

    def die_on_the_second_node(store, nodes):
        if len(nodes) == 2:
            raise SimulatedKill()
        real(store, nodes)

    monkeypatch.setattr(ExperimentHistoryStore, "reconcile", die_on_the_second_node)
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)
    monkeypatch.setattr(ExperimentHistoryStore, "reconcile", real)
    assert len(campaign.checkpointed()) == 2
    assert len(campaign.stored()) == 1
    paid = campaign.bill.total

    campaign.open(resume=True).solve(experiment_max_iter=0)

    assert campaign.stored() == campaign.checkpointed()
    assert campaign.bill.total == paid


def test_a_kill_while_the_history_is_rewritten_leaves_the_last_whole_file(
    campaign, monkeypatch
):
    real_dump = json.dump

    def die_halfway(payload, handle, **kwargs):
        if isinstance(payload, list) and len(payload) == 2:
            handle.write(json.dumps(payload, **kwargs)[:100])
            handle.flush()
            raise SimulatedKill()
        real_dump(payload, handle, **kwargs)

    monkeypatch.setattr(store_module.json, "dump", die_halfway)
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)
    monkeypatch.setattr(store_module.json, "dump", real_dump)

    # A session reading now gets the whole previous file, and the campaign,
    # whose checkpoint already has both nodes, resumes and catches up.
    assert len(campaign.stored()) == 1
    campaign.open(resume=True).solve(experiment_max_iter=0)
    assert len(campaign.stored()) == 2


def test_agents_read_the_scores_the_evaluator_transition_left(maintained):
    def edit(evaluation: Path) -> None:
        (evaluation / "kapso_eval.py").write_text("ENTRYPOINT = True\nFIXED = True\n")

    maintained.change_request_on = 1
    maintained.during_change_request = edit
    maintained.verdict = (
        "<change_verdict>accept</change_verdict><reason>real bug</reason>"
    )

    maintained.open().solve(experiment_max_iter=2)

    # Node 0 was never measured under the new ruler and projects None; node 1
    # is bridged first because it asked. Selection sees exactly this, and until
    # now agents were still shown the scores of the evaluator just replaced.
    assert maintained.checkpointed() == [
        (0, "node 0 attempt 1", None),
        (1, "node 1 attempt 1", maintained.bridge_score),
    ]
    assert maintained.stored() == maintained.checkpointed()
    read = maintained.agents_read()
    assert str(GHOST_SCORE) not in read
    assert str(maintained.bridge_score) in read


def test_reconcile_drops_strangers_and_pays_for_nothing_it_already_has(tmp_path):
    class Embedder:
        calls = []

        def create_embedding(self, text):
            self.calls.append(text)
            return [1.0, 0.0]

    def node(node_id, solution, score):
        return SimpleNamespace(
            node_id=node_id,
            solution=solution,
            score=score,
            feedback="",
            branch_name=f"generic_exp_{node_id}",
            had_error=False,
            error_message="",
        )

    store = ExperimentHistoryStore(str(tmp_path / "history.json"), llm=Embedder())
    store.reconcile([node(0, "kept", 0.5), node(1, "ghost", 0.99)])
    stamped = store.experiments[0].timestamp
    written = Path(store.json_path).stat().st_mtime_ns

    reopened = ExperimentHistoryStore(store.json_path, llm=Embedder())
    reopened.reconcile([node(0, "kept", 0.5), node(1, "ghost", 0.99)])
    assert Path(store.json_path).stat().st_mtime_ns == written

    reopened.reconcile([node(0, "kept", 0.7), node(1, "redo", 0.2)])

    assert Embedder.calls == ["kept", "ghost", "redo"]  # never the same one twice
    assert [(r.node_id, r.solution, r.score) for r in reopened.experiments] == [
        (0, "kept", 0.7),
        (1, "redo", 0.2),
    ]
    assert reopened.experiments[0].timestamp == stamped


# =========================================================================
# The inbox
# =========================================================================


def _kill_after_node_one_asks(campaign: AgentStoreCampaign) -> None:
    """The session on node 1 files a request, and the process dies before the
    pause reaches the checkpoint."""
    campaign.inbox_enabled = True

    def ask_and_die(node_id: int) -> None:
        if node_id == 1:
            file_requests(campaign.inbox, node=1, session="dead", entries=[ENTRY])
            raise SimulatedKill()

    campaign.during_implementation = ask_and_die
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)
    campaign.during_implementation = lambda node_id: None


def test_a_redo_can_ask_again_what_a_dead_round_asked(campaign):
    _kill_after_node_one_asks(campaign)
    assert [request.id for request in load_requests(campaign.inbox).values()] == [1]

    campaign.ask_on_node = 1
    result = campaign.open(resume=True).solve(experiment_max_iter=1)

    # The dead round's request is gone, and the redo's own question reached the
    # adapter's tail: a join on an open key writes no event, so the session
    # would have been told it was stopping and then left running.
    assert result.stopped_reason == "waiting_for_user"
    assert [request.id for request in load_requests(campaign.inbox).values()] == [2]
    assert [request["id"] for request in result.requests] == [2]

    record_reply(campaign.inbox, 2, "added the key")
    result = campaign.open(resume=True).solve(experiment_max_iter=1)
    assert result.stopped_reason == "max_iterations"
    assert [record[0] for record in campaign.stored()] == [0, 1]


def test_a_withdrawn_request_cannot_be_answered_and_its_id_is_never_reused(tmp_path):
    path = tmp_path / "inbox.jsonl"
    file_requests(path, node=1, session="dead", entries=[ENTRY])

    assert withdraw_orphaned_requests(path, waiting_nodes=set()) == [1]

    assert load_requests(path) == {}
    with pytest.raises(ValueError, match="withdrawn"):
        record_reply(path, 1, "too late")
    ((request_id, previous_reply),) = file_requests(
        path, node=2, session="live", entries=[ENTRY]
    )
    assert (request_id, previous_reply) == (2, None)
    assert new_requests_for_session(path, "live", 0)[1] == [2]


def test_only_what_nothing_waits_on_is_withdrawn(tmp_path):
    path = tmp_path / "inbox.jsonl"
    file_requests(
        path,
        node=0,
        session="s0",
        entries=[{**ENTRY, "key": "env:ANSWERED"}, {**ENTRY, "key": "env:WAITING"}],
    )
    record_reply(path, 1, "here")
    file_requests(path, node=3, session="s3", entries=[{**ENTRY, "key": "env:DEAD"}])

    assert withdraw_orphaned_requests(path, waiting_nodes={0}) == [3]

    # The reply holds for the whole campaign; node 0 is still suspended.
    requests = load_requests(path)
    assert sorted(requests) == [1, 2]
    assert requests[1].reply == "here"
    assert requests[2].open


def test_a_request_the_checkpoint_waits_on_survives_the_resume(campaign):
    campaign.inbox_enabled = True
    campaign.ask_on_node = 0
    campaign.open().solve(experiment_max_iter=1)

    result = campaign.open(resume=True).solve(experiment_max_iter=1)

    assert result.stopped_reason == "waiting_for_user"
    assert load_requests(campaign.inbox)[1].open


# =========================================================================
# The registered evaluation
# =========================================================================


def half_edit(evaluation: Path) -> None:
    (evaluation / "kapso_eval.py").write_text("ENTRYPOINT = True\nHALF_DONE = True\n")
    (evaluation / "scratch.py").write_text("LEFTOVER = True\n")


def test_a_kill_during_triage_does_not_lock_the_campaign_out(maintained):
    maintained.kill_during_triage_of_node_one(half_edit)
    head = EvaluationRegistry(str(maintained.workspace)).head()

    # The refusal this replaces: "does not match the registered evaluator head".
    result = maintained.open(resume=True).solve(experiment_max_iter=1)

    assert result.stopped_reason == "max_iterations"
    evaluation = maintained.workspace / "kapso_evaluation"
    assert (evaluation / "kapso_eval.py").read_text() == "ENTRYPOINT = True\n"
    assert not (evaluation / "scratch.py").exists()
    assert (
        EvaluationRegistry(str(maintained.workspace)).head().version
        == head.version
        == 1
    )
    assert not maintained.repo.is_dirty(path="kapso_evaluation")


def test_edits_the_killed_agent_committed_are_undone_too(maintained):
    def half_edit_and_commit(evaluation: Path) -> None:
        half_edit(evaluation)
        maintained.repo.git.add("-f", "kapso_evaluation")
        maintained.repo.git.commit("-m", "wip")

    maintained.kill_during_triage_of_node_one(half_edit_and_commit)

    result = maintained.open(resume=True).solve(experiment_max_iter=1)

    assert result.stopped_reason == "max_iterations"
    evaluation = maintained.workspace / "kapso_evaluation"
    assert (evaluation / "kapso_eval.py").read_text() == "ENTRYPOINT = True\n"
    assert not (evaluation / "scratch.py").exists()


def test_a_gitignored_leftover_of_a_killed_triage_is_not_committed_with_the_restore(maintained):
    def half_edit_and_log(evaluation: Path) -> None:
        half_edit(evaluation)
        (maintained.workspace / ".gitignore").write_text("*.log\n")
        (evaluation / "triage_run.log").write_text("the evaluator's output\n")

    maintained.kill_during_triage_of_node_one(half_edit_and_log)

    result = maintained.open(resume=True).solve(experiment_max_iter=1)

    assert result.stopped_reason == "max_iterations"
    assert not (maintained.workspace / "kapso_evaluation" / "triage_run.log").exists()


def test_a_kill_between_registering_and_committing_keeps_the_verdict(
    maintained, monkeypatch
):
    def fixed(evaluation: Path) -> None:
        (evaluation / "kapso_eval.py").write_text("ENTRYPOINT = True\nFIXED = True\n")

    maintained.change_request_on = 1
    maintained.during_change_request = fixed
    maintained.verdict = "<change_verdict>accept</change_verdict><reason>bug</reason>"
    real_commit = maintainer_module.EvaluationMaintainer._commit_evaluation

    def die_on_the_second_version(agent, message):
        if "v2" in message:
            raise SimulatedKill()
        real_commit(agent, message)

    monkeypatch.setattr(
        maintainer_module.EvaluationMaintainer,
        "_commit_evaluation",
        die_on_the_second_version,
    )
    with pytest.raises(SimulatedKill):
        maintained.open().solve(experiment_max_iter=5)
    monkeypatch.setattr(
        maintainer_module.EvaluationMaintainer,
        "_commit_evaluation",
        real_commit,
    )
    maintained.change_request_on = None

    maintained.open(resume=True).solve(experiment_max_iter=1)

    # v2 is committed, so a later failed triage restores v2 and not v1, and the
    # scores moved onto it.
    assert EvaluationRegistry(str(maintained.workspace)).head().version == 2
    assert not maintained.repo.is_dirty(path="kapso_evaluation")
    assert "FIXED" in maintained.repo.git.show("HEAD:kapso_evaluation/kapso_eval.py")
    strategy = maintained.orchestrator.search_strategy
    assert strategy.scores_evaluator_id == strategy.registered_evaluator_id


def test_a_finished_change_request_leaves_nothing_to_recover(maintained):
    maintained.change_request_on = 0

    maintained.open().solve(experiment_max_iter=1)

    assert not (
        maintained.workspace / ".kapso" / "evaluation_change_open.json"
    ).exists()


# =========================================================================
# The session transcript
# =========================================================================


def test_a_redo_reports_difficulties_from_its_own_transcript(campaign):
    stream = (
        campaign.workspace / ".kapso" / "sessions" / "generic_exp_1" / "stream.jsonl"
    )
    seen = {}

    def write_and_die(node_id: int) -> None:
        if node_id == 1:
            stream.parent.mkdir(parents=True, exist_ok=True)
            stream.write_text('{"dead": "attempt"}\n')
            raise SimulatedKill()

    campaign.during_implementation = write_and_die
    with pytest.raises(SimulatedKill):
        campaign.open().solve(experiment_max_iter=5)

    campaign.during_implementation = lambda node_id: seen.update(
        existed=stream.exists()
    )
    campaign.open(resume=True).solve(experiment_max_iter=1)

    assert seen == {"existed": False}
