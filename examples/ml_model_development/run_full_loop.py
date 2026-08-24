"""The complete Kapso production loop, end to end, on the ML example.

research -> learn_knowledge -> evolve -> learn -> evolve
(learn-facade-integration.md Phase 2; the reviewer gate judges the logs.)

Every stage's artifacts land in an isolated sandbox under learning/
(gitignored): its own bank, trajectory store, run roots, and wiki dir —
production stores untouched. The KG BACKENDS (local Weaviate/Neo4j
containers) are the shared dev ones; the wiki pages written there carry
this run's stamp in their sandbox wiki_dir copy.

Run:  PYTHONPATH=src:. python examples/ml_model_development/run_full_loop.py
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import yaml

from kapso.kapso import DEFAULT_CONFIG_PATH, Kapso
from kapso.learning.update_frame import init_bank
from examples.ml_model_development.e2e.generate_data import generate

EVOLVE_BUDGET_MINUTES = 20
EVOLVE_MAX_ITERATIONS = 4

GOAL = """
Optimize the ML model in `train.py` to improve accuracy on the Spaceship
Titanic dataset.

## Target File
The file `train.py` contains:
- `train_model()`: Function that trains and returns a model
- `predict_with_model()`: Function that makes predictions with the trained model

IMPORTANT: Do NOT change function names or signatures. Only modify internal
implementation.

## Data
The data/ directory contains:
- train.csv: Training data with features and the Transported target
- test.csv: Test data for final predictions (no target column)

## Evaluation
Run: python evaluate.py --data-dir ./data --seed 0

The evaluation:
- Splits training data into train/validation (90/10)

## Success Criteria
- Accuracy: Higher is better (baseline ~0.50 with DummyClassifier)
- Target: 0.78+ accuracy through improved modeling
"""

RESEARCH_QUESTION = (
    "Best-practice feature engineering and model selection for small "
    "tabular binary classification datasets (~2000 rows) with mixed "
    "numeric/categorical features and missing values, Spaceship-Titanic "
    "style: which imputation, encoding, interaction-feature, and "
    "gradient-boosting/ensemble choices reliably beat a plain baseline, "
    "and which common choices are traps at this sample size?"
)


def stage(status_path: Path, name: str, state: str, **extra) -> None:
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status[name] = {"state": state,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **extra}
    status_path.write_text(json.dumps(status, indent=1, default=str))
    print(f"\n=== STAGE {name}: {state} {extra if extra else ''}\n", flush=True)


def main() -> None:
    example_dir = Path(__file__).parent
    repo_root = example_dir.parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sandbox = repo_root / "learning" / "e2e-facade" / stamp
    sandbox.mkdir(parents=True)
    status_path = sandbox / "stage_status.json"
    print(f"E2E sandbox: {sandbox}", flush=True)

    # --- isolated config: packaged config with sandboxed learning homes ---
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    config["learning"]["bank"] = {"local_path": str(sandbox / "bank-home.git"),
                                  "remote": None}
    config["learning"]["trajectory_store"] = {"local": str(sandbox / "store"),
                                              "remote": None}
    config["learning"]["import_report_dir"] = str(sandbox / "imports")
    config["learning"]["serving"]["enabled"] = True
    config["learning"]["graders"]["run_root"] = str(sandbox / "graders")
    config["learning"]["update_crew"]["run_root"] = str(sandbox / "update")
    config["learning"]["mining"]["timeout_minutes"] = 60
    config_path = sandbox / "config.e2e.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    init_bank(str(sandbox / "bank-home.git"))

    # --- stage the task material ---
    workdir = sandbox / "task"
    shutil.copytree(example_dir / "initial_repo", workdir)
    train_df, test_df = generate(rows=2000, test_rows=500, seed=1337)
    (workdir / "data").mkdir()
    train_df.to_csv(workdir / "data" / "train.csv", index=False)
    test_df.to_csv(workdir / "data" / "test.csv", index=False)

    kapso = Kapso(config_path=str(config_path))
    (sandbox / "memory-0-start.txt").write_text(kapso.memory.explain())

    # --- 1. research ---
    stage(status_path, "research", "running")
    findings = kapso.research(RESEARCH_QUESTION, mode="idea", depth="light")
    (sandbox / "research_findings.md").write_text(str(findings))
    if len(str(findings)) < 500:
        raise RuntimeError(
            f"research produced only {len(str(findings))} chars — not a "
            "usable findings set"
        )
    stage(status_path, "research", "done",
          chars=len(str(findings)))

    # --- 2. learn_knowledge ---
    stage(status_path, "learn_knowledge", "running")
    kg_result = kapso.learn_knowledge(
        findings, wiki_dir=str(sandbox / "wikis"),
    )
    (sandbox / "memory-1-after-knowledge.txt").write_text(kapso.memory.explain())
    if not kapso.memory.knowledge_enabled:
        raise RuntimeError(
            "S1 CONTRACT VIOLATION: knowledge_search is not live after a "
            "merged learn_knowledge run"
        )
    if kg_result.errors or kg_result.total_pages_extracted == 0:
        raise RuntimeError(
            f"learn_knowledge ingested nothing usable: pages="
            f"{kg_result.total_pages_extracted}, errors={kg_result.errors}"
        )
    stage(status_path, "learn_knowledge", "done",
          pages=kg_result.total_pages_extracted,
          created=kg_result.created, edited=kg_result.edited,
          errors=len(kg_result.errors))

    # --- 3. evolve #1 (KG + founding bank) ---
    stage(status_path, "evolve1", "running")
    sol1 = kapso.evolve(
        goal=GOAL,
        initial_repo=str(workdir),
        output_path=str(sandbox / "campaign1"),
        max_iterations=EVOLVE_MAX_ITERATIONS,
        time_budget_minutes=EVOLVE_BUDGET_MINUTES,
    )
    (sandbox / "solution1.txt").write_text(sol1.explain())
    stage(status_path, "evolve1", "done",
          score=sol1.final_score, succeeded=sol1.succeeded,
          bank_head_served=sol1.metadata.get("bank_head_served"),
          kg_index=sol1.metadata.get("kg_index"))

    # --- 4. learn (the experience loop closes) ---
    stage(status_path, "learn", "running")
    lesson = kapso.learn(sol1)
    (sandbox / "lesson.txt").write_text(lesson.explain())
    (sandbox / "memory-2-after-lesson.txt").write_text(kapso.memory.explain())
    stage(status_path, "learn", "done",
          trajectory=lesson.trajectory_id, admitted=lesson.admitted,
          head_before=lesson.bank_head_before,
          head_after=lesson.bank_head_after,
          cards_created=lesson.cards_created,
          cards_updated=lesson.cards_updated)

    # --- 5. evolve #2 (KG + updated bank) ---
    stage(status_path, "evolve2", "running")
    sol2 = kapso.evolve(
        goal=GOAL,
        initial_repo=str(workdir),
        output_path=str(sandbox / "campaign2"),
        max_iterations=EVOLVE_MAX_ITERATIONS,
        time_budget_minutes=EVOLVE_BUDGET_MINUTES,
    )
    (sandbox / "solution2.txt").write_text(sol2.explain())
    (sandbox / "memory-3-final.txt").write_text(kapso.memory.explain())
    stage(status_path, "evolve2", "done",
          score=sol2.final_score, succeeded=sol2.succeeded,
          bank_head_served=sol2.metadata.get("bank_head_served"))

    # --- loop-closure check: sol2 was served the POST-lesson head ---
    served2 = sol2.metadata.get("bank_head_served")
    if lesson.admitted and served2 != lesson.bank_head_after:
        raise RuntimeError(
            f"LOOP NOT CLOSED: evolve2 served {served2}, lesson produced "
            f"{lesson.bank_head_after}"
        )
    stage(status_path, "loop", "closed",
          served_head_evolve2=served2, admitted=lesson.admitted)
    print(f"\nFULL_LOOP_COMPLETE sandbox={sandbox}", flush=True)


if __name__ == "__main__":
    main()
