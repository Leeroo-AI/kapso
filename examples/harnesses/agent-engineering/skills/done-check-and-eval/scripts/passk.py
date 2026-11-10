"""pass^1, pass^k, the mean score when present, the per-task table, the cost
line, and a paired comparison between two runs on the same tasks and trials,
from results/episodes.jsonl (the scaffold-evaluation convention). Standard
library only.

Usage:
  passk.py <episodes.jsonl> --k 3 [--against <other.jsonl>] [--write-summary <path>] [--json]

`--k` is the task's trial count. The script refuses a duplicate episode, a
non-boolean `passed`, tasks with different trial identifiers, and a comparison
across different tasks, trials or k. The comparison reports the per-task
paired differences with their mean and standard error across tasks; that
standard error describes the spread across tasks, not the run-to-run noise on
this fixed set, which the two baseline runs show as flips.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from math import comb, sqrt

REQUIRED = ("id", "trial", "passed")


def read(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in REQUIRED:
                if key not in row:
                    raise SystemExit(f"{path}:{number}: missing `{key}`")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no episodes")
    validate_rows(rows)
    return rows


def validate_rows(rows):
    """Every episode is one (id, trial) with a boolean outcome; every task has the same trials."""
    seen = set()
    trial_sets = defaultdict(set)
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise SystemExit("every episode needs a non-empty string `id`")
        trial = row.get("trial")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
            raise SystemExit(f"{row['id']}: `trial` must be a non-negative integer")
        if not isinstance(row.get("passed"), bool):
            raise SystemExit(f"{row['id']} t{trial}: `passed` must be a JSON boolean")
        key = (row["id"], trial)
        if key in seen:
            raise SystemExit(f"duplicate episode {row['id']} t{trial}")
        seen.add(key)
        trial_sets[row["id"]].add(trial)
    if len({frozenset(v) for v in trial_sets.values()}) != 1:
        raise SystemExit("tasks have different trial identifiers; the run is incomplete")
    return seen


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(rows, k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SystemExit("k must be a positive integer")
    by_task = defaultdict(list)
    trial_ids = set()
    for row in rows:
        by_task[row["id"]].append(row["passed"])
        trial_ids.add(row["trial"])
    n_trials = len(trial_ids)
    tasks = sorted(by_task)
    passes = {task: sum(by_task[task]) for task in tasks}
    rates = [passes[t] / n_trials for t in tasks]
    pass_1 = statistics.mean(rates)
    # The spread of the per-task rates across tasks, over sqrt(tasks): a
    # description of the task population, not of run-to-run noise.
    pass_1_se = statistics.stdev(rates) / sqrt(len(tasks)) if len(tasks) > 1 else 0.0
    pass_k = None
    if n_trials >= k:
        pass_k = statistics.mean(comb(passes[t], k) / comb(n_trials, k) for t in tasks)
    scores = [row["score"] for row in rows if isinstance(row.get("score"), (int, float)) and not isinstance(row.get("score"), bool)]
    errors = sum(1 for row in rows if row.get("error"))

    def line(key):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if not values:
            return None
        return {"mean": round(statistics.mean(values), 1), "median": percentile(values, 0.5), "p90": percentile(values, 0.9)}

    return {
        "tasks": len(tasks),
        "trials": n_trials,
        "trial_ids": sorted(trial_ids),
        "k": k,
        "pass_1": round(pass_1, 4),
        "pass_k": round(pass_k, 4) if pass_k is not None else None,
        "pass_1_task_se": round(pass_1_se, 4),
        "mean_score": round(statistics.mean(scores), 4) if scores else None,
        "passes_total": sum(passes.values()),
        "episodes": len(rows),
        "errors": errors,
        "per_task": passes,
        "cost": {
            "input_tokens": line("input_tokens"),
            "output_tokens": line("output_tokens"),
            "tool_calls": line("tool_calls"),
            "steps": line("steps"),
            "seconds": line("seconds"),
        },
    }


def compare(current, other):
    tasks = sorted(current["per_task"])
    if set(tasks) != set(other["per_task"]):
        raise SystemExit("the two runs cover different tasks; no comparison")
    if current["trials"] != other["trials"] or current["trial_ids"] != other["trial_ids"]:
        raise SystemExit(f"the two runs have different trials ({current['trial_ids']} vs {other['trial_ids']}); no comparison")
    if current["k"] != other["k"]:
        raise SystemExit("the two runs use different k; no comparison")
    n, k = current["trials"], current["k"]
    differences = [(current["per_task"][t] - other["per_task"][t]) / n for t in tasks]

    def se(values):
        return round(statistics.stdev(values) / sqrt(len(values)), 4) if len(values) > 1 else None

    k_differences = None
    if n >= k:
        k_differences = [(comb(current["per_task"][t], k) - comb(other["per_task"][t], k)) / comb(n, k) for t in tasks]
    return {
        "delta_pass_1": round(statistics.mean(differences), 4),
        "delta_pass_1_task_se": se(differences),
        "delta_pass_k": round(statistics.mean(k_differences), 4) if k_differences is not None else None,
        "delta_pass_k_task_se": se(k_differences) if k_differences is not None else None,
        "delta_passes": current["passes_total"] - other["passes_total"],
        "tasks_up": [t for t, d in zip(tasks, differences) if d > 0],
        "tasks_down": [t for t, d in zip(tasks, differences) if d < 0],
        "noise_note": (
            "the standard errors describe the spread of the paired per-task differences across tasks, not "
            "run-to-run noise on this fixed set; read the flips against the tasks that flipped between the two "
            "baseline runs"
        ),
    }


def render(summary, comparison):
    lines = [
        f"tasks {summary['tasks']}  trials {summary['trials']}  episodes {summary['episodes']}  errors {summary['errors']}",
        f"pass^1 {summary['pass_1']:.3f} (task se {summary['pass_1_task_se']:.3f})   pass^{summary['k']} "
        + (f"{summary['pass_k']:.3f}" if summary["pass_k"] is not None else "n/a (fewer trials than k)")
        + (f"   mean score {summary['mean_score']}" if summary["mean_score"] is not None else ""),
        "cost per episode (mean / median / p90):",
    ]
    for key, row in summary["cost"].items():
        if row:
            lines.append(f"  {key:<14} {row['mean']} / {row['median']} / {row['p90']}")
    lines.append("per task: " + ", ".join(f"{t}={p}/{summary['trials']}" for t, p in sorted(summary["per_task"].items())))
    if comparison:
        lines.append(
            f"against: delta pass^1 {comparison['delta_pass_1']:+.3f} (paired task se {comparison['delta_pass_1_task_se']}), "
            f"delta pass^k {comparison['delta_pass_k']} (paired task se {comparison['delta_pass_k_task_se']}), "
            f"delta passes {comparison['delta_passes']:+d}, up {comparison['tasks_up'] or '-'}, "
            f"down {comparison['tasks_down'] or '-'}; {comparison['noise_note']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--against", default=None)
    parser.add_argument("--write-summary", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize(read(args.path), args.k)
    comparison = compare(summary, summarize(read(args.against), args.k)) if args.against else None
    if args.write_summary:
        with open(args.write_summary, "w", encoding="utf-8") as handle:
            json.dump({"source": args.path, "against": args.against, "summary": summary, "comparison": comparison}, handle, indent=2)
            handle.write("\n")
    if args.json:
        json.dump({"summary": summary, "comparison": comparison}, sys.stdout, indent=2)
        print()
    else:
        print(render(summary, comparison))


if __name__ == "__main__":
    main()
