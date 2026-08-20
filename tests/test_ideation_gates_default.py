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
