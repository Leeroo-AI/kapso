"""Support for the harness tests that drive the real orchestrator: a fake
strategy that scores one candidate per run on a real branch and stages a
harness exactly as the real one does, a fake inference backend, the
patches that keep the engine's sessions out of a test, and the real
orchestrator over the real `GenericSearch` for the tests of what
construction writes (wp6-observability.md, change 13)."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import git

import kapso.execution.orchestrator as orchestrator_module
import kapso.kapso as kapso_module
from kapso.core.config import deep_merge, load_config
from kapso.core.roles import resolve_role
from kapso.execution.coding_agents.base import CodingResult
from kapso.execution.observability import BuildStatus
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch



def fake_config(**blocks: Any) -> Dict[str, Any]:
    """The packaged config with a test's own blocks over it."""
    return deep_merge(load_config(), blocks)


class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def get_cumulative_cost(self) -> float:
        return 0.0

    def create_embedding(self, text, model=None):
        return []


class FakeWorkspace:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def get_cumulative_cost(self) -> float:
        return 0.0


class ScoringFakeStrategy:
    """One scored candidate per run, on a real branch of the workspace;
    stages the harness the way the real strategy does."""

    env_strip: List[str] = []
    env_defaults: Dict[str, str] = {}
    gate_failure_policy = "warn"
    inbox_settings: Optional[Dict[str, Any]] = None
    budget_snapshot = None

    def __init__(self, workspace_dir: str, harness: Any):
        self.workspace_dir = workspace_dir
        self.workspace = FakeWorkspace(workspace_dir)
        self.last_round_seconds = None
        self.finishing_cut = None
        self.harness = harness
        config = load_config()
        self.implementation_role = resolve_role(config["hosts"], config["search_strategy"]["params"], "implementation")
        self.iteration_count = 0
        self.node_history: List[SearchNode] = []
        self.scores_evaluator_id = ""
        self.evaluator_transition = None
        self.registered_evaluator_id = ""
        self.operation_status = BuildStatus(
            Path(workspace_dir) / ".kapso" / "status.json", phase="setup",
            harness=harness.ref_dict() if harness is not None else None,
        )
        self.phases: List[str] = []

    def _status_phase(self, phase: str) -> None:
        self.phases.append(phase)
        self.operation_status.phase(phase)

    def _clamped_timeout(self, seconds: float) -> float:
        return seconds

    def _room(self, *, finishing: bool = False, obligations_seconds: float = 0.0):
        return None

    def stage_harness_for(self, module: str, session_dir: str, label: str, agent_type: str):
        # The engine's own method, on this fake: the records tests pin the
        # contract itself, not a copy of it.
        return GenericSearch.stage_harness_for(self, module, session_dir, label, agent_type)

    def continues_an_answered_node(self) -> bool:
        return False   # the fake has no inbox continuation; the orchestrator's admission asks every strategy

    def run(self, context: str, budget_progress: float = 0.0) -> SearchNode:
        node_id = len(self.node_history)
        branch = f"generic_exp_{node_id}"
        repo = git.Repo(self.workspace_dir)
        if branch not in {head.name for head in repo.heads}:
            repo.create_head(branch)
        node = SearchNode(node_id=node_id, branch_name=branch, solution="# idea\n", score=0.5 + node_id / 10, feedback="fine")
        self.node_history.append(node)
        self.iteration_count += 1
        return node

    def waiting_requests(self):
        return []

    def observe_budget(self, snapshot: Any) -> None:
        pass

    def observe_fidelity(self, decision: Any) -> None:
        pass

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        return list(self.node_history)

    def get_best_experiment(self) -> Optional[SearchNode]:
        return max(self.node_history, key=lambda n: n.score or 0.0) if self.node_history else None

    def get_deliverable_experiment(self) -> Optional[SearchNode]:
        return self.get_best_experiment()

    def get_deliverable_score(self) -> Optional[float]:
        best = self.get_best_experiment()
        return best.score if best else None

    def checkout_to_best_experiment_branch(self) -> Optional[str]:
        best = self.get_best_experiment()
        if best is None:
            return None
        git.Repo(self.workspace_dir).git.checkout(best.branch_name)
        return best.branch_name

    def dump_state(self) -> Dict[str, Any]:
        return {"node_history": [node.to_dict() for node in self.node_history]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.node_history = [SearchNode.from_dict(item) for item in state.get("node_history", [])]


class NoopAgent:
    """A coding-agent session that does nothing and succeeds: what the
    orchestrator's own sessions (the harness setup) run as in tests that
    are not about them."""

    def __init__(self, config):
        self.config = config

    def initialize(self, workspace: str) -> None:
        pass

    def generate_code(self, prompt: str, debug_mode: bool = False, timeout_seconds=None) -> CodingResult:
        return CodingResult(success=True, output="")

    def get_cumulative_cost(self) -> float:
        return 0.0

    def cleanup(self) -> None:
        pass


class FakeProblemHandler:
    """What the real strategy asks of a problem handler at construction
    and in the summary: the scoring direction and the context."""

    maximize_scoring = True

    def get_problem_context(self) -> str:
        return "Improve the candidate"


# The orchestrator's own strategy creation, kept before any test stubs it.
REAL_CREATE_STRATEGY = OrchestratorAgent._create_search_strategy


def patch_engine(
    monkeypatch, *, inbox: Optional[Dict[str, Any]] = None, real_strategy: bool = False,
) -> List[Any]:
    """The real orchestrator with its sessions stubbed, the facade's inbox
    registry off, no preflight; returns the strategies it creates. A
    strategy-level inbox block (path, stop_grace_seconds) makes the
    strategy report the inbox on, without the facade's registry. The
    orchestrator's own session (harness setup) runs as a NoopAgent; a test
    about it patches the factory again after this. With `real_strategy`
    the orchestrator creates the real `GenericSearch` through the real
    factory (the mode config names no embedding model, so the history
    store is recency-only; the repo memory bootstrap has no inference to
    run on and the strategy keeps the deterministic map), so construction
    walks the orchestrator-to-factory seam and writes the real status
    file."""
    monkeypatch.setattr(orchestrator_module.CodingAgentFactory, "create", staticmethod(lambda config: NoopAgent(config)))
    monkeypatch.setattr(orchestrator_module, "CliInference", FakeLLM)
    # Every launch registers the campaign (wp6-observability.md): the
    # registry is a file of the test's own, never the machine's.
    config = fake_config(
        inbox={"enabled": False, "registry": str(Path(tempfile.mkdtemp(prefix="kapso-registry-")) / "campaigns.jsonl")},
    )
    monkeypatch.setattr(orchestrator_module, "load_config", lambda *a, **k: dict(config))
    monkeypatch.setattr(kapso_module, "load_config", lambda *a, **k: dict(config))
    monkeypatch.setattr(kapso_module, "run_preflight", lambda *a, **k: None)
    monkeypatch.setattr(OrchestratorAgent, "_create_feedback_generator", lambda self: object())
    strategies: List[Any] = []

    def create_strategy(self, workspace_dir, start_from_checkpoint):
        if real_strategy:
            strategy = REAL_CREATE_STRATEGY(self, workspace_dir, start_from_checkpoint)
        else:
            strategy = ScoringFakeStrategy(workspace_dir, self.harness)
        if inbox is not None:
            strategy.inbox_settings = dict(inbox)
        strategies.append(strategy)
        return strategy

    monkeypatch.setattr(OrchestratorAgent, "_create_search_strategy", create_strategy)
    return strategies


def real_orchestrator(
    monkeypatch, workspace: Path, *, goal: str = "Improve support", harness: Any = None,
    experience: Optional[List[Dict[str, Any]]] = None, resume: bool = False,
) -> OrchestratorAgent:
    """The real `OrchestratorAgent` over the real `GenericSearch` on a
    seeded workspace (`git_workspace`), its sessions stubbed as
    `patch_engine` does: construction alone, so a test reads what it
    wrote (the status file with the harness and the sources) or what it
    published when its own setup raised. Nothing runs unless the test
    calls `solve()`."""
    patch_engine(monkeypatch, real_strategy=True)
    return OrchestratorAgent(
        FakeProblemHandler(), workspace_dir=str(workspace), goal=goal, harness=harness,
        experience=experience, resume=resume,
    )


def stub_solve(monkeypatch, result: Any) -> None:
    """`solve()` replaced for a facade test over the real construction: it
    hands `result` to the delivery and returns it, running no session."""

    def solve(self, *args, finish=None, **kwargs):
        if finish is not None:
            finish(result)
        return result

    monkeypatch.setattr(OrchestratorAgent, "solve", solve)
