"""Pin the platform-default MCP gate lists of the generic strategy.

Design decision #4 (platform unification): `research` STAYS in the
platform default ideation gates — benchmarks not wanting the proxy trim
it in their own mode config, never by shrinking the platform default.
"""

from contextlib import contextmanager
from types import SimpleNamespace

from kapso.execution.search_strategies.generic.strategy import GenericSearch


@contextmanager
def _patched_super_init(workspace_dir):
    from kapso.execution.search_strategies.base import SearchStrategy

    original = SearchStrategy.__init__

    def fake_init(self, config, wd=None, import_from_checkpoint=False):
        self.params = config.params or {}
        self.workspace_dir = workspace_dir
        self.feedback_generator = None

    SearchStrategy.__init__ = fake_init
    yield
    SearchStrategy.__init__ = original


def test_default_gates_include_research(tmp_path):
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(SimpleNamespace(params={}), str(tmp_path))
    assert strategy.ideation_gates == [
        "research", "experiment_history", "repo_memory", "leeroopedia",
    ]
    assert strategy.implementation_gates == [
        "research", "repo_memory", "leeroopedia",
    ]


def test_staged_memory_stores_mount_their_own_gates(tmp_path):
    # Regression (E2E review 2026-08-24, blockers 1+2): serving injects an
    # intro naming bank_index / bank_get_card / bank_get_card_with_evidence
    # and learn_knowledge writes pages the wiki-search gates read. Both are
    # reachable only if the providing gate is mounted, so the gate list
    # FOLLOWS the staged store instead of being an independent config
    # choice — the live run advertised bank tools no session had.
    params = {
        "bank_serving": {"KAPSO_BANK_DIR": "/tmp/bank"},
        "kg_index_path": "/tmp/wikis/.index",
    }
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(SimpleNamespace(params=params), str(tmp_path))
    for gates in (strategy.ideation_gates, strategy.implementation_gates):
        assert "bank" in gates
        assert {"idea", "code"} <= set(gates)


def test_gates_unchanged_without_staged_stores(tmp_path):
    # The mount is conditional: no staged store, no extra gates.
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(SimpleNamespace(params={}), str(tmp_path))
    for gates in (strategy.ideation_gates, strategy.implementation_gates):
        assert "bank" not in gates
        assert "idea" not in gates and "code" not in gates
