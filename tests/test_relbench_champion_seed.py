"""Champion seeding contract (benchmarks/relbench/handler.py).

Pins the two things that must never regress:
- the staged bundle is SANITIZED — validation-side fields only, never
  test metrics or audit output;
- a task without a claim (or with a claim missing its code snapshot)
  seeds nothing and returns None, the documented default.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import benchmarks.relbench.handler as handler_module
from benchmarks.relbench.handler import RelBenchHandler


def _stub(tmp_path: Path, problem_id: str) -> SimpleNamespace:
    shared = tmp_path / "shared_cache"
    shared.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        problem_id=problem_id,
        shared_cache_dir=shared,
        spec=SimpleNamespace(primary_metric="roc_auc"),
    )


def _write_claim(claims_root: Path, problem_id: str, with_code: bool = True) -> None:
    claim = claims_root / problem_id
    claim.mkdir(parents=True)
    (claim / "final_report.json").write_text(json.dumps({
        "run": "run_0007",
        "primary_metric": "roc_auc",
        "val_metrics": {"roc_auc": 0.8379, "f1": 0.41},
        "test_metrics": {"roc_auc": 0.8889},
        "audit": {"flags": []},
    }))
    if with_code:
        (claim / "code").mkdir()
        (claim / "code" / "main.py").write_text("print('champion')\n")
    (claim / "solution.md").write_text("# design\n")


def test_seed_champion_stages_sanitized_bundle(tmp_path, monkeypatch):
    claims_root = tmp_path / "claims"
    _write_claim(claims_root, "rel-x--task-a")
    monkeypatch.setattr(handler_module, "CLAIMS_DIR", claims_root)

    stub = _stub(tmp_path, "rel-x--task-a")
    result = RelBenchHandler._seed_champion(stub)

    assert result == {"run": "run_0007", "val": 0.8379}
    dest = stub.shared_cache_dir / "champion"
    assert (dest / "code" / "main.py").read_text() == "print('champion')\n"
    assert (dest / "solution.md").exists()
    staged = json.loads((dest / "champion_report.json").read_text())
    assert staged == {
        "run": "run_0007",
        "primary_metric": "roc_auc",
        "val_metrics": {"roc_auc": 0.8379, "f1": 0.41},
    }
    assert "test_metrics" not in staged and "audit" not in staged


def test_seed_champion_absent_claim_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(handler_module, "CLAIMS_DIR", tmp_path / "claims")
    stub = _stub(tmp_path, "rel-x--task-a")
    assert RelBenchHandler._seed_champion(stub) is None
    assert not (stub.shared_cache_dir / "champion").exists()


def test_seed_champion_claim_without_code_returns_none(tmp_path, monkeypatch):
    claims_root = tmp_path / "claims"
    _write_claim(claims_root, "rel-x--task-a", with_code=False)
    monkeypatch.setattr(handler_module, "CLAIMS_DIR", claims_root)
    stub = _stub(tmp_path, "rel-x--task-a")
    assert RelBenchHandler._seed_champion(stub) is None
    assert not (stub.shared_cache_dir / "champion").exists()
