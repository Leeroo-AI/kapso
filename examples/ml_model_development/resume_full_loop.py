"""Resume the full-loop E2E from a sandbox whose research+learn_knowledge
stages completed (run_full_loop.py, learn-facade-integration.md Phase 2).

Usage:
  PYTHONPATH=src:. python examples/ml_model_development/resume_full_loop.py \
      learning/e2e-facade/<stamp> [--from evolve1|learn]

`--from learn` reuses a completed evolve1 by learning from its campaign
DIRECTORY (learn()'s path dispatch) — no SolutionResult needed, so a
crashed later stage never costs a good campaign.

Validates the completed stages' artifacts, reconnects the knowledge
search (the pages already live in the KG backends — no re-learning),
wipes any partial campaign, then runs evolve1 -> learn -> evolve2 and
the loop-closure check into the SAME sandbox.
"""

import json
import shutil
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from kapso.kapso import Kapso
from kapso.knowledge_base.search import KnowledgeSearchFactory

from examples.ml_model_development.run_full_loop import (
    EVOLVE_BUDGET_MINUTES,
    EVOLVE_MAX_ITERATIONS,
    GOAL,
    stage,
)


def main() -> None:
    sandbox = Path(sys.argv[1]).resolve()
    start_at = "evolve1"
    if "--from" in sys.argv:
        start_at = sys.argv[sys.argv.index("--from") + 1]
    if start_at not in ("evolve1", "learn"):
        raise ValueError("--from must be evolve1 or learn")
    status_path = sandbox / "stage_status.json"
    status = json.loads(status_path.read_text())
    if status["research"]["state"] != "done" or (
        status["learn_knowledge"]["state"] != "done"
    ):
        raise RuntimeError(
            "resume requires research + learn_knowledge done; run "
            "run_full_loop.py instead"
        )
    if len((sandbox / "research_findings.md").read_text()) < 500:
        raise RuntimeError("research_findings.md missing or thin")
    print(f"Resuming E2E in sandbox: {sandbox}", flush=True)

    kapso = Kapso(config_path=str(sandbox / "config.e2e.yaml"))

    # Reconnect knowledge search: the 54 pages are already merged into the
    # KG backends; a fresh process just needs the connection (the same
    # config-preset construction learn_knowledge's post-merge refresh and
    # index_kg use).
    mode = kapso._config.get("default_mode", "GENERIC")
    mode_config = kapso._config.get("modes", {}).get(mode, {})
    search_config = mode_config.get("knowledge_search", {})
    params = search_config.get("params", {}).copy()
    params.setdefault("models", mode_config.get("models"))
    params.setdefault("retry", mode_config.get("retry"))
    kapso.knowledge_search = KnowledgeSearchFactory.create(
        search_type=search_config.get("type", "kg_graph_search"),
        params=params,
    )
    if not kapso.memory.knowledge_enabled:
        raise RuntimeError("knowledge search failed to reconnect on resume")
    (sandbox / "memory-1b-resumed.txt").write_text(kapso.memory.explain())

    # A partial campaign from the interrupted run cannot be resumed at this
    # layer (the driver owns whole stages) — wipe and redo cleanly.
    keep = {"campaign1"} if start_at == "learn" else set()
    for partial in ("campaign1", "campaign2"):
        if partial in keep:
            continue
        if (sandbox / partial).exists() and status.get(
            f"evolve{partial[-1]}", {}
        ).get("state") != "done":
            shutil.rmtree(sandbox / partial)
            print(f"  wiped partial {partial}", flush=True)

    workdir = sandbox / "task"

    # --- 3. evolve #1 (KG + founding bank) ---
    sol1 = None
    if start_at == "learn":
        if status.get("evolve1", {}).get("state") != "done":
            raise RuntimeError("--from learn requires a completed evolve1")
        print("  reusing completed evolve1 (campaign1)", flush=True)
    elif status.get("evolve1", {}).get("state") != "done":
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
    else:
        # Redo: a completed-but-unusable stage (e.g. a budget-starved
        # campaign) is re-run from scratch into a fresh campaign dir.
        # The durable stages (research, learn_knowledge) are never redone.
        print("  evolve1 was done — redoing it (fresh campaign)", flush=True)
        if (sandbox / "campaign1").exists():
            shutil.rmtree(sandbox / "campaign1")
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

    # --- 4. learn ---
    stage(status_path, "learn", "running")
    lesson = kapso.learn(
        sol1 if sol1 is not None else str(sandbox / "campaign1")
    )
    (sandbox / "lesson.txt").write_text(lesson.explain())
    (sandbox / "memory-2-after-lesson.txt").write_text(kapso.memory.explain())
    stage(status_path, "learn", "done",
          trajectory=lesson.trajectory_id, admitted=lesson.admitted,
          head_before=lesson.bank_head_before,
          head_after=lesson.bank_head_after,
          cards_created=lesson.cards_created,
          cards_updated=lesson.cards_updated)

    # --- 5. evolve #2 ---
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
