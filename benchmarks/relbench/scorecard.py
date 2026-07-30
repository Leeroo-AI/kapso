"""Category scorecard: assemble the classification / regression / recommendation
tables with Kapso's results inserted into the verified baseline field.

Reads tmp/relbench/<dataset>--<task>/final_report.json files (produced by
final_evaluate), converts to leaderboard conventions (AUROC %, NMAE, MAP %),
ranks Kapso inside the full board field, and evaluates the category-level
claim gates against RelAgent, KumoRFM (fine-tuned), and the board #1.

Usage:
    PYTHONPATH=src:. python -m benchmarks.relbench.scorecard [--work-root tmp/relbench]
        [--out tmp/relbench/scorecard.md]

Conventions (see BASELINES.md):
- Classification: test AUROC (%), higher better. 12 v1 tasks.
- Regression: test NMAE = MAE / std(train targets), lower better. 9 v1 tasks.
  Divisors from data/baselines.json (verified against the board and raw data).
- Recommendation: test MAP (%), higher better. Two aggregate sets:
  * v1-9: the 9 v1 tasks — the KumoRFM-ft comparison set (report Table 4);
  * board-10: the board's 10 columns (adds rel-f1/driver-circuit-compete).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"

V1_REC_TASKS = [
    "rel-amazon/user-item-purchase", "rel-amazon/user-item-rate",
    "rel-amazon/user-item-review", "rel-avito/user-ad-visit",
    "rel-hm/user-item-purchase", "rel-stack/user-post-comment",
    "rel-stack/post-post-related", "rel-trial/condition-sponsor-run",
    "rel-trial/site-sponsor-run",
]

# Tasks whose test windows extend years past test_timestamp, so the two
# published evaluation regimes (frozen-db vs per-row seed-time) diverge.
# These are evaluated through the rolling harness (per-tick snapshot cascade,
# EVALUATION_PROTOCOL.md) so their regime matches the published bars.
PROTOCOL_SENSITIVE_TASKS = {
    "rel-f1/driver-position",
    "rel-f1/driver-dnf",
    "rel-f1/driver-top3",
}

# Rolling harness verified per task (campaign gate opens only for these):
# driver-position — full acceptance 2026-07-29: reference B-rolling through the
#   real grader+cascade scored test MAE 2.6516 (band 2.653±0.015; bar 2.731);
# driver-dnf / driver-top3 — cascade invariants verified same day: every test
#   snapshot leak-clean, snapshot-relabeled train == official labels exactly.
ROLLING_VERIFIED = set(PROTOCOL_SENSITIVE_TASKS)


def load_kapso_results(work_root: Path) -> Dict[str, dict]:
    results = {}
    for report_file in sorted(work_root.glob("*--*/final_report.json")):
        try:
            report = json.loads(report_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "test_metrics" not in report or not report["test_metrics"]:
            continue
        results[f"{report['dataset']}/{report['task']}"] = report
    return results


def kapso_value(report: dict, section: str, divisors: Dict[str, float]) -> Optional[float]:
    tm = report["test_metrics"]
    task_id = f"{report['dataset']}/{report['task']}"
    if section == "classification" and "roc_auc" in tm:
        return round(tm["roc_auc"] * 100, 2)
    if section == "regression" and "mae" in tm:
        div = divisors.get(task_id)
        return round(tm["mae"] / div, 4) if div else None
    if section == "recommendation" and "link_prediction_map" in tm:
        return round(tm["link_prediction_map"] * 100, 2)
    return None


def rank_in_field(value: float, field: List[float], higher_better: bool) -> int:
    better = sum(1 for v in field if (v > value if higher_better else v < value))
    return better + 1


def fmt(v, nd=2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def build_section(
    section: str,
    board: dict,
    baselines: dict,
    kapso: Dict[str, dict],
    rec_set: str = "board-10",
) -> Tuple[str, dict]:
    meta = board["_meta"]["metric"][section]
    higher = meta["higher_is_better"]
    nd = 4 if section == "regression" else 2
    divisors = baselines["_meta"]["train_std_divisors_nmae"]

    tasks = list(board[section]["tasks"])
    methods = dict(board[section]["methods"])

    # KumoRFM rows are absent from the board's recommendation table — inject
    # the tech-report numbers (best known) so ranks reflect the true field.
    if section == "recommendation":
        for key, label in (("kumorfm_fine_tuned", "KumoRFM (fine-tuned) [report]"),
                           ("kumorfm_in_context", "KumoRFM (in-context) [report]")):
            cells = {t: baselines[key]["v1_recommendation_map_pct"].get(t) for t in tasks}
            methods[label] = {"regime": "task-specific" if "fine" in key else "zero-shot",
                              "mean": None, "cells": cells}
        if rec_set == "v1-9":
            tasks = [t for t in tasks if t in V1_REC_TASKS]

    # Kapso's row.
    kapso_cells = {}
    for t in tasks:
        report = kapso.get(t)
        if report:
            kapso_cells[t] = kapso_value(report, section, divisors)

    done = [t for t in tasks if kapso_cells.get(t) is not None]
    missing = [t for t in tasks if t not in done]

    # Per-task ranks over the full field + Kapso (methods with a value there).
    ranks = {}
    for t in done:
        field = [m["cells"].get(t) for m in methods.values()]
        field = [v for v in field if v is not None]
        ranks[t] = rank_in_field(kapso_cells[t], field, higher)

    # Key comparison rows.
    key_rows = {}
    if section == "classification":
        key_rows["RelAgent"] = {t: baselines["relagent"]["v1_classification_auroc_pct"].get(t, {}).get("test") for t in tasks}
        key_rows["KumoRFM (fine-tuned)"] = {t: baselines["kumorfm_fine_tuned"]["v1_classification_auroc_pct"].get(t) for t in tasks}
    elif section == "regression":
        key_rows["RelAgent"] = {t: baselines["relagent"]["v1_regression_mae"].get(t, {}).get("board_nmae") for t in tasks}
        key_rows["KumoRFM (fine-tuned)"] = {t: baselines["kumorfm_fine_tuned"]["v1_regression_mae"].get(t, {}).get("board_nmae") for t in tasks}
        key_rows["RT (pretrained + fine-tuned)"] = dict(methods.get("RT (pretrained + fine-tuned)", {}).get("cells", {}))
    else:
        key_rows["KumoRFM (fine-tuned)"] = {t: baselines["kumorfm_fine_tuned"]["v1_recommendation_map_pct"].get(t) for t in tasks}
        key_rows["ID-GNN (4 layers)"] = dict(methods.get("ID-GNN (4 layers)", {}).get("cells", {}))

    def mean_over(cells: Dict[str, Optional[float]], subset: List[str]) -> Optional[float]:
        vals = [cells.get(t) for t in subset if cells.get(t) is not None]
        return round(sum(vals) / len(vals), nd) if len(vals) == len(subset) and subset else None

    # Wins vs each key baseline over completed tasks with both values.
    gates = {"completed": len(done), "total": len(tasks), "missing": missing}
    for name, cells in key_rows.items():
        both = [t for t in done if cells.get(t) is not None]
        wins = sum(
            1 for t in both
            if (kapso_cells[t] > cells[t] if higher else kapso_cells[t] < cells[t])
        )
        full_mean_baseline = mean_over(cells, tasks)
        full_mean_kapso = mean_over(kapso_cells, tasks)
        gates[name] = {
            "wins": wins,
            "comparable": len(both),
            "baseline_mean": full_mean_baseline,
            "kapso_mean": full_mean_kapso,
            "beats_mean": (
                None if full_mean_kapso is None or full_mean_baseline is None
                else (full_mean_kapso > full_mean_baseline if higher else full_mean_kapso < full_mean_baseline)
            ),
        }
    gates["mean_rank_completed"] = (
        round(sum(ranks.values()) / len(ranks), 2) if ranks else None
    )

    # Markdown table.
    title = section.capitalize()
    if section == "recommendation":
        title += f" ({rec_set}; MAP %)"
    elif section == "regression":
        title += " (NMAE = MAE ÷ std(train); lower better)"
    else:
        title += " (AUROC %)"
    lines = [f"### {title}", "", "| Task | " + " | ".join(list(key_rows) + ["Kapso", "rank"]) + " |",
             "|" + "---|" * (len(key_rows) + 3)]
    for t in tasks:
        row = [t] + [fmt(key_rows[n].get(t), nd) for n in key_rows]
        row.append(fmt(kapso_cells.get(t), nd))
        row.append(str(ranks[t]) if t in ranks else "—")
        lines.append("| " + " | ".join(row) + " |")
    mean_row = ["**Mean**"] + [fmt(mean_over(key_rows[n], tasks), nd) for n in key_rows]
    mean_row.append(fmt(mean_over(kapso_cells, tasks), nd))
    mean_row.append(fmt(gates["mean_rank_completed"]) if gates["mean_rank_completed"] else "—")
    lines.append("| " + " | ".join(mean_row) + " |")
    lines.append("")
    lines.append(
        f"Coverage: {len(done)}/{len(tasks)} tasks"
        + (f" — missing: {', '.join(missing)}" if missing else " — complete ✓")
    )
    for name, g in gates.items():
        if isinstance(g, dict) and "wins" in g:
            verdict = {True: "BEATS mean", False: "below mean", None: "mean n/a (incomplete)"}[g["beats_mean"]]
            lines.append(
                f"- vs **{name}**: wins {g['wins']}/{g['comparable']} tasks; "
                f"means {fmt(g['kapso_mean'], nd)} vs {fmt(g['baseline_mean'], nd)} → {verdict}"
            )
    return "\n".join(lines), gates


def _campaign_orders() -> Tuple[List[str], set]:
    """ROI order + cpu-safe set — sourced from the campaign pipeline (one home)."""
    from benchmarks.relbench.campaign import CPU_LOCAL_QUEUE, ROI_QUEUE

    return list(ROI_QUEUE), set(CPU_LOCAL_QUEUE)


def _kapso_primary(report: dict, spec) -> Optional[float]:
    tm = report.get("test_metrics") or {}
    v = tm.get(spec.primary_metric)
    return None if v is None else float(v)


def _to_sota_units(value: float, metric: str, divisors: Dict[str, float], task_id: str) -> Optional[float]:
    """Convert a raw primary-metric value into the units used by sota.json."""
    if metric == "auroc_pct" or metric == "accuracy_pct" or metric == "map_pct":
        return value * 100
    if metric == "nmae":
        div = divisors.get(task_id)
        return value / div if div else None
    return value  # r2, raw mae


def _family_and_metric(ds: str, task_name: str) -> Tuple[str, str]:
    """Family + primary metric from registry metadata only (no data loading)."""
    from relbench.base import AutoCompleteTask, RecommendationTask, TaskType

    from benchmarks.relbench import task_specs as ts
    from relbench.tasks import task_registry

    cls, args, kwargs = task_registry[ds][task_name]
    if cls is AutoCompleteTask or (isinstance(cls, type) and issubclass(cls, AutoCompleteTask)):
        task_type = kwargs["task_type"]
        family = {
            TaskType.BINARY_CLASSIFICATION: ts.AUTOCOMPLETE_BINARY,
            TaskType.MULTICLASS_CLASSIFICATION: ts.AUTOCOMPLETE_MULTICLASS,
            TaskType.REGRESSION: ts.AUTOCOMPLETE_REGRESSION,
        }[task_type]
    elif issubclass(cls, RecommendationTask):
        family = ts.RECOMMENDATION
    else:
        family = {
            TaskType.BINARY_CLASSIFICATION: ts.ENTITY_BINARY,
            TaskType.MULTICLASS_CLASSIFICATION: ts.ENTITY_MULTICLASS,
            TaskType.REGRESSION: ts.ENTITY_REGRESSION,
        }[cls.task_type]
    metric, _ = ts.PRIMARY_METRIC_OVERRIDES.get(f"{ds}/{task_name}", ts.PRIMARY_METRIC[family])
    return family, metric


CLAIMS_DIR = Path(__file__).parent / "claims"


def _archive_claim(task_id: str, report: dict, work_root: Path) -> Optional[dict]:
    """Copy a winning run's evidence into the durable, committed claims/ dir.

    tmp/ archives are box-local and gitignored; organizer handoffs need the
    winning code snapshot, both prediction files, the solution spec, and the
    final report to survive. Idempotent per (task, run): re-copies only when
    the selected run changes.
    """
    import hashlib
    import shutil

    run_name = report.get("run")
    src_run = work_root / task_id.replace("/", "--") / "runs" / str(run_name)
    if not run_name or not src_run.exists():
        return None

    dest = CLAIMS_DIR / task_id.replace("/", "--")
    marker = dest / "CLAIMED_RUN"
    if not (marker.exists() and marker.read_text().strip() == run_name):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name in ("val_predictions.npy", "test_predictions.npy", "solution.md"):
            if (src_run / name).exists():
                shutil.copy2(src_run / name, dest / name)
        if (src_run / "code").exists():
            shutil.copytree(src_run / "code", dest / "code")
        (dest / "final_report.json").write_text(json.dumps(report, indent=2, default=str))
        marker.write_text(run_name + "\n")

    def sha(name: str) -> str:
        p = dest / name
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "—"

    rel = dest.relative_to(Path(__file__).parents[2])
    return {
        "run": run_name,
        "dir": str(rel),
        "val_sha": sha("val_predictions.npy"),
        "test_sha": sha("test_predictions.npy"),
    }


def build_reference(work_root: Path) -> str:
    from relbench.tasks import get_task_names

    from benchmarks.relbench.task_specs import NATIVE_DATASETS, SIZE_TIER

    baselines = json.loads((DATA_DIR / "baselines.json").read_text())
    sota = json.loads((DATA_DIR / "sota.json").read_text())
    divisors = baselines["_meta"]["train_std_divisors_nmae"]
    kapso = load_kapso_results(work_root)
    roi_order, cpu_safe = _campaign_orders()

    # RelBench v1 task set = the 30 tasks in the original benchmark (all
    # evaluated by KumoRFM-ft). Everything else in the native 66 is a v2
    # addition (the autocomplete family + the 4 v2-only databases + extra
    # entity/rec tasks). The relbench-hf submission leaderboard ranks 31
    # tasks — the 30 v1 tasks split into Classification (12) / Regression
    # (9) / Recommendation (9), plus the one v2 rec task the board carries
    # (rel-f1/driver-circuit-compete) → 10 in the Recommendation category.
    v1_tasks = set()
    for _sect in ("v1_classification_auroc_pct", "v1_regression_mae",
                  "v1_recommendation_map_pct"):
        v1_tasks |= {k for k in baselines["kumorfm_fine_tuned"][_sect] if not k.startswith("_")}
    board_tasks = v1_tasks | {"rel-f1/driver-circuit-compete"}

    ra_clf = baselines["relagent"]["v1_classification_auroc_pct"]
    ra_reg = baselines["relagent"]["v1_regression_mae"]
    ra_v2 = baselines["relagent"]["v2_subset"]
    ku_clf = baselines["kumorfm_fine_tuned"]["v1_classification_auroc_pct"]
    ku_reg = baselines["kumorfm_fine_tuned"]["v1_regression_mae"]
    ku_rec = baselines["kumorfm_fine_tuned"]["v1_recommendation_map_pct"]
    ic_v1 = baselines["kumorfm_in_context"]
    ic_v2 = baselines["kumorfm_2_in_context"]

    def in_context_cell(block: dict, task_id: str) -> Optional[float]:
        """Zero-shot KumoRFM cell in the row's units (AUROC %, board NMAE, MAP %)."""
        if task_id in block["v1_classification_auroc_pct"]:
            return block["v1_classification_auroc_pct"][task_id]
        if task_id in block["v1_regression_mae"]:
            div = divisors.get(task_id)
            return round(block["v1_regression_mae"][task_id] / div, 4) if div else None
        return block.get("v1_recommendation_map_pct", {}).get(task_id)

    rows = []
    n_done = n_beats_sota = 0
    for ds in NATIVE_DATASETS:
        for task_name in get_task_names(ds):
            task_id = f"{ds}/{task_name}"
            if ds == "rel-mimic":
                rows.append({"task": task_id, "family": "clf", "metric": "auroc_pct",
                             "sota": sota.get(task_id, {}), "hw": "blocked",
                             "ver": "v2", "board": False,
                             "status": "⛔ credentialed data", "roi": None})
                continue
            family, primary_metric = _family_and_metric(ds, task_name)
            entry = sota.get(task_id, {})
            fam_map = {"entity_binary_classification": "clf", "entity_multiclass_classification": "mc",
                       "entity_regression": "reg", "recommendation": "rec"}
            fam = fam_map.get(family) or "AC-" + family.split("_")[1][:3]

            # Baseline cells in the same units as the best-known column: for
            # v1 regression (nmae rows) use the verified board NMAE; RelAgent's
            # v2-subset regression values are raw MAE while the row unit is R²
            # — annotate rather than mix silently.
            relagent = kumo = None
            if task_id in ra_clf:
                relagent = ra_clf[task_id]["test"]
            elif task_id in ra_reg:
                relagent = ra_reg[task_id]["board_nmae"]
            elif task_id in ra_v2:
                v = ra_v2[task_id].get("test_selected")
                relagent = v if ra_v2[task_id].get("metric") != "mae" else f"{v:g} (MAE)"
            if task_id in ku_clf:
                kumo = ku_clf[task_id]
            elif task_id in ku_reg:
                kumo = ku_reg[task_id]["board_nmae"]
            elif task_id in ku_rec:
                kumo = ku_rec[task_id]

            report = kapso.get(task_id)
            kapso_val = beats = None
            if report:
                n_done += 1
                raw = (report.get("test_metrics") or {}).get(primary_metric)
                if raw is not None:
                    kapso_val = _to_sota_units(float(raw), entry.get("metric", ""), divisors, task_id)
                    if kapso_val is not None and entry.get("value") is not None:
                        higher = entry["metric"] != "nmae"
                        ok = kapso_val > entry["value"] if higher else kapso_val < entry["value"]
                        beats = "✅ beats best-known" if ok else "below best-known"
                        n_beats_sota += ok

            # Campaign runs record what actually ran (hardware, wall-clock cap)
            # and where the full trace archive lives; those override the
            # requirement-derived defaults for this row.
            meta_path = work_root / f"{ds}--{task_name}" / "campaign_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            archive_path = work_root / f"{ds}--{task_name}" / "artifact_archive.json"
            archive = json.loads(archive_path.read_text()) if archive_path.exists() else {}

            rows.append({
                "task": task_id, "family": fam, "metric": entry.get("metric", primary_metric),
                "ver": "v1" if task_id in v1_tasks else "v2",
                "board": task_id in board_tasks,
                "sota": entry, "relagent": relagent, "kumo": kumo,
                "kumo_v1_ic": in_context_cell(ic_v1, task_id),
                "kumo_v2_ic": in_context_cell(ic_v2, task_id),
                "kapso": kapso_val, "beats": beats,
                "hw": meta.get("hardware") or ("CPU-ok" if task_id in cpu_safe else "GPU box"),
                "tier": (f"{meta['cap_hours']:g}h" if meta.get("cap_hours") else
                         {"small": "2h", "medium": "4h", "large": "8h"}.get(SIZE_TIER.get(ds, "medium"), "4h")),
                "status": ("✅ done" if report else
                           ("▶ running" if meta and not meta.get("planned") else "· pending")),
                "roi": roi_order.index(task_id) + 1 if task_id in roi_order else None,
                "archive": archive.get("uri"),
            })

    # Order: v1 tasks first, then v2; within each version by ROI rank
    # (unranked tasks last).
    rows.sort(key=lambda r: (r.get("ver") != "v1", r["roi"] is None, r["roi"] or 999))

    # Durable evidence for every winning cell (organizer handoff).
    claims = []
    for r in rows:
        if r.get("beats") == "✅ beats best-known":
            claim = _archive_claim(r["task"], kapso[r["task"]], work_root)
            if claim:
                claims.append((r["task"], claim))

    def f(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    lines = [
        "# RelBench campaign reference — agent results, baselines, hardware, status",
        "",
        "**Auto-generated — do not edit by hand.** Regenerate with:",
        "`PYTHONPATH=src:. python -m benchmarks.relbench.scorecard --reference`",
        "",
        f"Status: **{n_done}/66 tasks run**, {n_beats_sota} beating the best published number. "
        "Category-level gates: run the scorecard (same module, no flags).",
        "",
        "## Benchmark version & leaderboard submission",
        "",
        "Each task is tagged **v1** or **v2** in the `Ver` column. The public "
        "**relbench-hf submission leaderboard ranks 31 tasks** (marked ★): the 30 "
        "original v1 tasks — Classification (12) / Regression (9) / Recommendation (9) "
        "— plus the one v2 recommendation task the board carries "
        "(`rel-f1/driver-circuit-compete`), making 10 in Recommendation. The other 35 "
        "tasks (the 23-task autocomplete family + the v2-only databases rel-arxiv / "
        "rel-salt / rel-ratebeer / rel-mimic + extra v2 entity/rec tasks) are **not on "
        "the submission board** — they are tracked here for completeness against the "
        "v2 paper. **Submit v1**: the three categories are independent but each requires "
        "predictions for *all* its tasks (per `relbench.leaderboard.LEADERBOARD_TASKS`; "
        "package with `python -m relbench.leaderboard preds/ --package`, then open a "
        "submission issue on stanford-star/relbench@relbench-hf).",
        "",
        "## Hardware requirements",
        "",
        "- **CPU-ok** (39 tasks): rel-f1 (1 MB), rel-salt (34 MB), rel-event (100 MB), "
        "rel-arxiv (145 MB), rel-avito (347 MB), rel-trial (548 MB db.zip). "
        "Runs on an 8-core / 32 GB box; the handler steers agents to duckdb/GBDT when no GPU is present.",
        "- **GPU box** (26 tasks): rel-stack (840 MB, text-heavy), rel-hm (31M-row transactions), "
        "rel-ratebeer (2.2 GB), rel-amazon (6.1 GB). CUDA GPU + 64 GB RAM recommended; 8h full-run caps.",
        "- **Blocked** (1 task): rel-mimic needs credentialed PhysioNet + BigQuery access.",
        "- The 'Cap' column is a **harness setting, not a benchmark rule** — RelBench imposes "
        "no time/compute limits (baselines range from RelAgent's ~1h/task to RelGT-AC's 22h runs). "
        "Our caps (2h/4h/8h full, 15/20/30 min debug, by DB tier) bound a single candidate run so "
        "the search always proceeds; override with RELBENCH_FULL_TIMEOUT / RELBENCH_DEBUG_TIMEOUT.",
        "",
        "## Baselines",
        "",
        "Verified primary-source numbers (see BASELINES.md for protocols and citations): "
        "**RelAgent** (arXiv:2605.07840, val-selected test of 5 searches; v1 entity + 6-task v2 subset, "
        "no recommendation), **KumoRFM fine-tuned** (Kumo tech report Tables 2-4, single values, all 30 v1 tasks), "
        "**KumoRFM-v1/v2 in-context** (zero-shot; v1: tech report, all 30 tasks; v2: arXiv:2604.12596, "
        "21 entity tasks, no recommendation), "
        "full board field in `data/leaderboard_snapshot.json`, per-task best-known in `data/sota.json`.",
        "",
        "## Evaluation protocol",
        "",
        "Per-task temporal regimes are documented in `EVALUATION_PROTOCOL.md` — the "
        "single authority on what data a solution may see at test time. Where RelBench's "
        "library default (freeze the database at `test_timestamp`) and the bar-setters' "
        "released evaluation (KumoRFM: full database, per-row seed-time anchoring) "
        "disagree, this campaign adopts the KumoRFM regime — its numbers are the "
        "comparison target. ⚠ marks the tasks where the regimes diverge (multi-tick "
        "rel-f1 windowed tasks): these are evaluated through the **rolling harness** "
        "(per-tick snapshot cascade; verified 2026-07-29 — the reference model "
        "reproduced its hand-run seed-time score, test MAE 2.6516 vs the 2.731 bar). "
        "Kapso cells recorded before that date were frozen-regime numbers and "
        "undersell the seed-time score.",
        "",
        "## Per-task table (v1 then v2; ROI order within each)",
        "",
        "Values in the best-known number's units (AUROC/acc/MAP in %, NMAE, R², raw MAE). "
        "'Best known' = strongest published result anywhere (board ∪ papers).",
        "",
        "'Ver' = benchmark version; ★ = one of the 31 tasks on the relbench-hf "
        "submission leaderboard; ⚠ = evaluation-regime-sensitive (see `EVALUATION_PROTOCOL.md`).",
        "",
        "| ROI# | Task | Fam | Ver | Best known (method) | RelAgent | KumoRFM-ft | KumoRFM-v1 (ic) | KumoRFM-v2 (ic) | Kapso | vs best | HW | Cap | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["sota"]
        best = f"{f(e.get('value'))} ({e.get('method', '?').split(' (')[0][:24]})" if e else "—"
        ver_cell = r.get("ver", "—") + ("★" if r.get("board") else "")
        task_cell = r["task"] + (" ⚠" if r["task"] in PROTOCOL_SENSITIVE_TASKS else "")
        lines.append(
            f"| {f(r['roi'], 0)} | {task_cell} | {r['family']} | {ver_cell} | {best} "
            f"| {f(r.get('relagent'))} | {f(r.get('kumo'))} "
            f"| {f(r.get('kumo_v1_ic'))} | {f(r.get('kumo_v2_ic'))} | {f(r.get('kapso'))} "
            f"| {r.get('beats') or '—'} | {r['hw']} | {r.get('tier', '—')} | {r['status']} |"
        )
    lines += [
        "",
        "Notes: baseline columns show per-task values in the same units where the method "
        "published one (— where it did not evaluate). KumoRFM-ft is the fine-tuned regime "
        "(Kumo tech report Tables 2-4); KumoRFM-v1/v2 (ic) are the zero-shot in-context "
        "regimes (v1: tech report; v2: arXiv:2604.12596, cross-checked against RelAgent's "
        "KumoRFM-v2 rows — no published v1 recommendation numbers exist for v2). Current "
        "'done' rows from harness-validation runs are baseline-quality placeholders until "
        "the campaign proper replaces them.",
    ]
    archived = [(r["task"], r["archive"]) for r in rows if r.get("archive")]
    if archived:
        lines += [
            "",
            "## Run artifacts (full traces)",
            "",
            "Complete campaign state per run — session transcripts, lens/ideation "
            "history, every candidate run, workspace and logs — archived to durable "
            "storage at task completion:",
            "",
            "| Task | Archive |",
            "|---|---|",
        ]
        for t, uri in archived:
            lines.append(f"| {t} | `{uri}` |")
    if claims:
        lines += [
            "",
            "## Winning artifacts (durable, committed — for organizer handoff)",
            "",
            "Each claimed cell's evidence is copied from the box-local run archive into "
            "`benchmarks/relbench/claims/<task>/`: winning code snapshot, both prediction "
            "files, solution spec, and final report (val+test metrics, audit). "
            "SHA-256 prefixes pin the exact prediction files the metrics were computed from.",
            "",
            "| Task | Run | Evidence dir | val preds sha256 | test preds sha256 |",
            "|---|---|---|---|---|",
        ]
        for task_id, c in claims:
            lines.append(
                f"| {task_id} | {c['run']} | `{c['dir']}/` | `{c['val_sha']}` | `{c['test_sha']}` |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="tmp/relbench")
    parser.add_argument("--out", default=None)
    parser.add_argument("--reference", action="store_true",
                        help="regenerate benchmarks/relbench/RESULTS.md and exit")
    args = parser.parse_args()

    if args.reference:
        doc = build_reference(Path(args.work_root))
        out = Path(__file__).parent / "RESULTS.md"
        out.write_text(doc)
        print(f"[scorecard] wrote {out}")
        return

    board = json.loads((DATA_DIR / "leaderboard_snapshot.json").read_text())
    baselines = json.loads((DATA_DIR / "baselines.json").read_text())
    kapso = load_kapso_results(Path(args.work_root))

    print(f"[scorecard] found {len(kapso)} Kapso task reports under {args.work_root}\n")
    sections = []
    summary = {}
    for section in ("classification", "regression"):
        md, gates = build_section(section, board, baselines, kapso)
        sections.append(md)
        summary[section] = gates
    for rec_set in ("v1-9", "board-10"):
        md, gates = build_section("recommendation", board, baselines, kapso, rec_set=rec_set)
        sections.append(md)
        summary[f"recommendation_{rec_set}"] = gates

    doc = "# RelBench category scorecard\n\n" + "\n\n".join(sections) + "\n"
    print(doc)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(doc)
        Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
        print(f"[scorecard] wrote {args.out} (+ .json)")


if __name__ == "__main__":
    main()
