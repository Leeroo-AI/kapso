"""Symptom triage over results/episodes.jsonl (the scaffold-evaluation
convention). Classes come from the failed check names, the error category,
the termination, and, when the runner published a reference route for the
development set, the matched flags, the names of the differing arguments, the
order of the steps and their counts. The action classes are relative to that
reference route: they say where the episode left it, not that leaving it
caused the failure. No expected value is read or printed, and an error is
reported as a category, never as the runner's text. Standard library only.

Usage: triage.py <episodes.jsonl> [--json] [--against <other.jsonl>]
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict

ERROR_CATEGORIES = ("overflow", "empty_reply", "timeout", "endpoint_error")
ENDPOINT_NEEDLES = (
    ("contextwindow", "overflow"),
    ("context length", "overflow"),
    ("context_length", "overflow"),
    ("must have either content", "empty_reply"),
    ("empty", "empty_reply"),
    ("timed out", "timeout"),
    ("timeout", "timeout"),
)
TIMEOUT_TERMINATIONS = ("timeout", "max_steps", "max_turns", "budget", "step_limit")
AGENT_ENDINGS = ("", "stop", "end", "done", "agent_stop", "completed")
COMPONENT = {
    "check_failed": "checks before done; grounding when the check names a behaviour never looked at",
    "no_steps": "the loop itself (the first step never ran)",
    "ended_early": "the close phase, the step budget",
    "timeout": "the context policy, the step budget, grounding stop rules",
    "precheck_skipped": "the compiled plan",
    "order_violation": "the compiled plan (prerequisites in order)",
    "write_never_made": "the compiled plan, or grounding",
    "wrong_sibling_tool": "the decision by evidence",
    "argument_differs": "the action review (vocabulary, traced arguments, worked calculations)",
    "item_missed": "the intake item list, every item accounted for",
    "handoff_prefilled": "the handoff as written",
    "handoff_missing": "the handoff as written",
    "user_step_missing": "the handoff and waiting for the other party's result",
    "user_extra_calls": "the handoff as written, one handoff per step",
    "duplicate_write": "the action review, recovery by contract",
    "extra_write": "clarify or look up before acting",
    "empty_reply": "endpoint handling (thinking budget, repair layer)",
    "overflow": "the context policy",
    "endpoint_error": "endpoint handling",
    "state_mismatch_unexplained": "read two transcripts; ask the episode-auditor",
}


def read(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "trial", "passed"):
                if key not in row:
                    raise SystemExit(f"{path}:{number}: missing `{key}`")
            if not isinstance(row["passed"], bool):
                raise SystemExit(f"{path}:{number}: `passed` must be a JSON boolean")
            key = (row["id"], row["trial"])
            if key in seen:
                raise SystemExit(f"{path}:{number}: duplicate episode {row['id']} t{row['trial']}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no episodes")
    return rows


def error_category(error):
    """A category for the row's error: the category itself when the adapter wrote one, else from needles."""
    text = str(error).strip().lower()
    if text in ERROR_CATEGORIES:
        return text
    return next((category for needle, category in ENDPOINT_NEEDLES if needle in text), "endpoint_error")


def family(name):
    """The operation a tool name belongs to, without its leading verb and a
    trailing numeric suffix: approve_x_1234 and deny_x_1235 share `x`. A
    heuristic for the wrong_sibling_tool class, labelled as such."""
    parts = [p for p in (name or "").split("_") if p]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) > 1:
        parts = parts[1:]
    return "_".join(parts)


def key_of(entry):
    return (entry.get("name"), entry.get("tool"))


def classify_actions(row):
    """Where an action-scored episode left the published reference route: (turn, class, tool, detail)."""
    expected = row.get("expected_actions") or []
    calls = [c for c in (row.get("calls") or []) if c.get("kind") != "error"]
    turns = defaultdict(list)
    for call in calls:
        turns[key_of(call)].append(call.get("turn", 0))
    first_turn = {key: min(t) for key, t in turns.items()}
    made = Counter(key_of(c) for c in calls)
    expected_count = Counter(key_of(e) for e in expected)
    found = []

    for entry in expected:
        key = key_of(entry)
        kind = entry.get("kind")
        tool = entry.get("tool") or entry.get("name")
        if entry.get("matched"):
            continue
        if not entry.get("attempted"):
            if kind == "user":
                cls, detail = "user_step_missing", "reference step %d by the other party not made" % entry["index"]
            elif kind in ("read", "unlock"):
                cls, detail = "precheck_skipped", "reference step %d never attempted" % entry["index"]
            elif kind == "handoff":
                cls, detail = "handoff_missing", "reference step %d never attempted" % entry["index"]
            elif kind == "write":
                sibling = any(
                    c.get("kind") == "write" and key_of(c) != key and family(c.get("tool") or c.get("name")) == family(tool)
                    for c in calls
                )
                cls = "wrong_sibling_tool" if sibling else "write_never_made"
                detail = ("a write of the same family made instead (name heuristic)" if sibling
                          else "reference step %d never attempted" % entry["index"])
            else:
                cls, detail = "write_never_made", "reference step %d never attempted" % entry["index"]
            found.append((None, cls, tool, detail))
        elif entry.get("diff_keys"):
            cls = "handoff_prefilled" if kind == "handoff" else "argument_differs"
            found.append((first_turn.get(key), cls, tool, "args differ: " + ", ".join(entry["diff_keys"])))

    # Order: a reference step first made after a later reference step. Unlocks
    # and meta calls carry no state and are left out on both sides.
    ordered = [e for e in expected if e.get("kind") in ("read", "write", "handoff", "user")]
    for entry in ordered:
        key = key_of(entry)
        if key not in first_turn:
            continue
        turn = first_turn[key]
        later_steps = [e for e in ordered if e["index"] > entry["index"] and key_of(e) in first_turn]
        earliest_later = min((first_turn[key_of(e)] for e in later_steps), default=None)
        if earliest_later is not None and earliest_later < turn:
            found.append((turn, "order_violation", entry.get("tool") or entry.get("name"),
                          "made at turn %s after a later reference step at turn %s" % (turn, earliest_later)))

    for key, expected_n in expected_count.items():
        kind = next(e.get("kind") for e in expected if key_of(e) == key)
        if kind != "write":
            continue
        if 0 < made[key] < expected_n:
            found.append((max(turns[key]), "item_missed", key[1] or key[0], "%d of %d reference actions made" % (made[key], expected_n)))
        if made[key] > expected_n:
            repeated_at = sorted(turns[key])[expected_n]
            found.append((repeated_at, "duplicate_write", key[1] or key[0], "%d made, %d in the reference; repeat at turn %s" % (made[key], expected_n, repeated_at)))
    for call in calls:
        if call.get("kind") == "write" and call.get("requestor") == "assistant" and key_of(call) not in expected_count:
            found.append((call.get("turn", 0), "extra_write", call.get("tool") or call.get("name"), "not in the reference route"))
    expected_user = sum(1 for e in expected if e.get("kind") == "user")
    user_turns = sorted(c.get("turn", 0) for c in calls if c.get("kind") == "user")
    if len(user_turns) > expected_user:
        found.append((user_turns[expected_user], "user_extra_calls", None,
                      "%d calls by the other party, %d in the reference; first extra at turn %s" % (len(user_turns), expected_user, user_turns[expected_user])))
    return found


def classify(row):
    """The observed symptoms of one failed episode: (turn or None, class, check or tool, detail)."""
    found = []
    if row.get("error"):
        found.append((None, error_category(row["error"]), None, "error category only; read the permitted trace"))
    termination = str(row.get("termination") or "").lower()
    if any(t in termination for t in TIMEOUT_TERMINATIONS):
        found.append((None, "timeout", None, termination))
    steps = row.get("steps") if isinstance(row.get("steps"), int) else row.get("tool_calls")
    if isinstance(steps, int) and steps == 0:
        found.append((None, "no_steps", None, "no step made"))
    for check in row.get("checks") or []:
        if check.get("passed") is False:
            found.append((None, "check_failed", check.get("name"), "named check failed; cause not inferred"))
    if row.get("expected_actions") is not None:
        found.extend(classify_actions(row))
        untried = [e for e in row["expected_actions"] if not e.get("matched") and not e.get("attempted")]
        if untried and termination in AGENT_ENDINGS and not row.get("error"):
            found.append((None, "ended_early", None, "ended with %d reference step(s) untaken" % len(untried)))
    if not found:
        found.append((None, "state_mismatch_unexplained", None, "no symptom visible in the results file"))
    unique = {}
    for turn, cls, tool, detail in sorted(found, key=lambda f: (f[0] is None, f[0] or 0, f[1])):
        unique.setdefault((cls, tool), (turn, cls, tool, detail))
    return sorted(unique.values(), key=lambda f: (f[0] is None, f[0] or 0, f[1]))


def triage(rows):
    failed = [row for row in rows if not row["passed"]]
    histogram = Counter()
    tools = Counter()
    symptoms = []
    for row in failed:
        found = classify(row)
        for _turn, cls, tool, _detail in found:
            histogram[cls] += 1
            if tool:
                tools[tool] += 1
        turn, cls, tool, detail = found[0]
        symptoms.append({"id": row["id"], "trial": row["trial"], "turn": turn, "class": cls, "tool": tool, "detail": detail,
                         "classes": sorted({f[1] for f in found})})
    episodes_per_class = Counter()
    for item in symptoms:
        for cls in item["classes"]:
            episodes_per_class[cls] += 1

    def cost(key):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        return round(statistics.mean(values), 1) if values else None

    return {
        "episodes": len(rows),
        "failed": len(failed),
        "classes": dict(episodes_per_class.most_common()),
        "instances": dict(histogram.most_common()),
        "tools": dict(tools.most_common(12)),
        "symptoms": symptoms,
        "cost": {k: cost(k) for k in ("input_tokens", "tool_calls", "steps", "seconds")},
    }


def render(report, comparison):
    lines = [f"episodes {report['episodes']}, failed {report['failed']}",
             "observed symptom classes (episodes; action classes relative to the published reference route):"]
    for cls, count in report["classes"].items():
        lines.append(f"  {count:>3}  {cls:<28} -> usually {COMPONENT.get(cls, '?')}")
    lines.append("checks or tools most involved: " + (", ".join(f"{t} ({n})" for t, n in report["tools"].items()) or "-"))
    lines.append("representative symptom per failed episode (cause to be established on the transcript):")
    for item in report["symptoms"]:
        where = f"turn {item['turn']}" if item["turn"] is not None else "episode"
        lines.append(f"  {item['id']} t{item['trial']}: {where} {item['class']} {item['tool'] or ''} ({item['detail']})")
    lines.append("cost per episode (mean): " + ", ".join(f"{k} {v}" for k, v in report["cost"].items() if v is not None))
    if comparison:
        lines.append(f"against: flipped to pass {comparison['to_pass'] or '-'}; flipped to fail {comparison['to_fail'] or '-'}")
    return "\n".join(lines)


def compare(rows, other_rows):
    current = {(r["id"], r["trial"]): r["passed"] for r in rows}
    other = {(r["id"], r["trial"]): r["passed"] for r in other_rows}
    if set(current) != set(other):
        raise SystemExit("the two runs cover different task/trial pairs; no comparison")
    keys = sorted(current)
    return {
        "to_pass": [f"{i} t{t}" for i, t in keys if current[(i, t)] and not other[(i, t)]],
        "to_fail": [f"{i} t{t}" for i, t in keys if other[(i, t)] and not current[(i, t)]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--against", default=None)
    args = parser.parse_args()
    rows = read(args.path)
    report = triage(rows)
    comparison = compare(rows, read(args.against)) if args.against else None
    if args.json:
        json.dump({"triage": report, "comparison": comparison}, sys.stdout, indent=2)
        print()
    else:
        print(render(report, comparison))


if __name__ == "__main__":
    main()
