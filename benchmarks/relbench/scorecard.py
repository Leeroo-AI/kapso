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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="tmp/relbench")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

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
