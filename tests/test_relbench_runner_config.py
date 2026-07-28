"""RelBench runtime configuration joins the canonical repository registry."""

from __future__ import annotations

from benchmarks.relbench.runner import build_runtime_config
from kapso.core.config import load_effective_config


def test_runtime_config_carries_the_typed_relbench_binding(tmp_path):
    path = build_runtime_config(str(tmp_path))

    effective = load_effective_config(path, "RELBENCH_GENERIC")

    assert effective.cross_run_binding.to_dict() == {
        "scope_id": "ml_ai",
        "task_family_id": "relational_tabular_prediction",
        "task_adapter_id": "relbench",
    }
    assert effective.cross_run.scopes.resolve("ml_ai").to_dict() == {
        "scope_id": "ml_ai",
        "expert_repository": "Leeroo-AI/kapso-expert",
        "knowledge_repository": "Leeroo-AI/kapso-knowledge",
        "security_repository": "Leeroo-AI/kapso-security",
    }
