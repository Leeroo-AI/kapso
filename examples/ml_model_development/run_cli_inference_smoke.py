"""Live smoke of the CLI-only inference conversion (design §7.7).

Scaled-down production loop — research (CLI web-search session) ->
learn_knowledge (small ingest; S1 refresh) -> one short evolve (repo
memory, commit messages, judge, embeddings on the converted stack) —
with per-stage wall-clock recorded so the timings can be compared
against the API-path baseline in learning/e2e-facade/20260825T223634
(research 76s; evolve1 15m26s for the full 45-min-budget run).

Run:  PYTHONPATH=src:. python examples/ml_model_development/run_cli_inference_smoke.py
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml

from kapso.kapso import DEFAULT_CONFIG_PATH, Kapso
from kapso.learning.update_frame import init_bank
from examples.ml_model_development.e2e.generate_data import generate
from examples.ml_model_development.run_full_loop import GOAL

# Smoke budgets: the point is exercising every converted seam through at
# least one full iteration (implementation -> evaluation -> judge ->
# commit), not reaching the example's target score. The example's
# reference E2E showed 20 minutes starves the FIRST implementation, so
# 25 gives one full iteration headroom without paying for a real run.
EVOLVE_BUDGET_MINUTES = 25
EVOLVE_MAX_ITERATIONS = 2

RESEARCH_QUESTION = (
    "For small tabular binary classification (~2000 rows, mixed "
    "numeric/categorical, missing values): the two highest-leverage "
    "modeling moves over a DummyClassifier baseline, with concrete "
    "sklearn-compatible choices."
)


def main() -> None:
    example_dir = _REPO_ROOT / "examples/ml_model_development"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sandbox = _REPO_ROOT / "learning" / "e2e-facade" / f"{stamp}-cli-smoke"
    sandbox.mkdir(parents=True)
    build = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO_ROOT),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    timings = {"build": build}
    timings_path = sandbox / "stage_timings.json"
    print(f"CLI smoke sandbox: {sandbox} (build {build})", flush=True)

    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    config["learning"]["bank"] = {"local_path": str(sandbox / "bank-home.git"),
                                  "remote": None}
    config["learning"]["trajectory_store"] = {"local": str(sandbox / "store"),
                                              "remote": None}
    config["learning"]["import_report_dir"] = str(sandbox / "imports")
    config["learning"]["serving"]["enabled"] = True
    config["learning"]["graders"]["run_root"] = str(sandbox / "graders")
    config["learning"]["update_crew"]["run_root"] = str(sandbox / "update")
    config_path = sandbox / "config.smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    init_bank(str(sandbox / "bank-home.git"))

    workdir = sandbox / "task"
    shutil.copytree(example_dir / "initial_repo", workdir)
    train_df, test_df = generate(rows=2000, test_rows=500, seed=1337)
    (workdir / "data").mkdir()
    train_df.to_csv(workdir / "data" / "train.csv", index=False)
    test_df.to_csv(workdir / "data" / "test.csv", index=False)

    kapso = Kapso(config_path=str(config_path))

    def timed(name, fn):
        print(f"=== SMOKE STAGE {name}: running", flush=True)
        t0 = time.time()
        result = fn()
        timings[name] = round(time.time() - t0, 1)
        timings_path.write_text(json.dumps(timings, indent=1))
        print(f"=== SMOKE STAGE {name}: done in {timings[name]}s", flush=True)
        return result

    # --- 1. research: the facade's Researcher() now runs codex --search ---
    findings = timed("research", lambda: kapso.research(
        RESEARCH_QUESTION, mode="idea", depth="light",
    ))
    (sandbox / "research_findings.md").write_text(str(findings))
    if len(str(findings)) < 500:
        raise RuntimeError(
            f"research produced only {len(str(findings))} chars"
        )

    # --- 2. learn_knowledge: small ingest; proves the KG path + S1 wiring ---
    kg_result = timed("learn_knowledge", lambda: kapso.learn_knowledge(
        findings, wiki_dir=str(sandbox / "wikis"),
    ))
    if kg_result.errors or kg_result.total_pages_extracted == 0:
        raise RuntimeError(
            f"learn_knowledge ingested nothing usable: pages="
            f"{kg_result.total_pages_extracted}, errors={kg_result.errors}"
        )
    if not kapso.memory.knowledge_enabled:
        raise RuntimeError("S1 refresh failed: knowledge_search not live")
    timings["learn_knowledge_pages"] = kg_result.total_pages_extracted
    timings_path.write_text(json.dumps(timings, indent=1))

    # --- 3. short evolve: repo memory + commit messages + judge + embeddings ---
    sol = timed("evolve", lambda: kapso.evolve(
        goal=GOAL,
        initial_repo=str(workdir),
        output_path=str(sandbox / "campaign"),
        max_iterations=EVOLVE_MAX_ITERATIONS,
        time_budget_minutes=EVOLVE_BUDGET_MINUTES,
    ))
    (sandbox / "solution.txt").write_text(sol.explain())
    timings["evolve_score"] = sol.final_score
    timings["evolve_succeeded"] = sol.succeeded
    timings_path.write_text(json.dumps(timings, indent=1))
    print(f"\nCLI_SMOKE_COMPLETE sandbox={sandbox} timings={timings}",
          flush=True)


if __name__ == "__main__":
    main()
