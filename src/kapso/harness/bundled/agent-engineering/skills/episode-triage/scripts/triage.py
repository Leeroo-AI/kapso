"""Symptom triage over results/episodes.jsonl (the scaffold-evaluation
convention). Classes come from the failed check names, the error category,
the termination, and, when the runner published a reference route for the
development set, the matched flags, the names of the differing arguments, the
order of the steps and their counts. The action classes are relative to that
reference route: they say where the episode left it, not that leaving it
caused the failure. No expected value is read or printed, and an error is
the category the adapter wrote, never the runner's text. `termination` is the
runner's own word, so the words that mean a limit ended the episode and the
ones that mean it simply ended are arguments, and the terminations neither
list reads are printed. Standard library only.

Usage: triage.py <episodes.jsonl> [--json] [--against <other.jsonl>]
                 [--limit-terminations a,b] [--endings a,b]
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

CATEGORY = re.compile(r"^[a-z][a-z0-9_]*$")
REQUESTORS = ("assistant", "user", "tool")
LIMIT_TERMINATIONS = "timeout,max_steps,max_turns,budget,step_limit"
ENDINGS = ",stop,end,done,agent_stop,completed"
COMPONENT = {
    "check_failed": "checks before done; grounding when the check names a behaviour never looked at",
    "no_steps": "the loop itself (the first step never ran)",
    "ended_early": "the close phase, the step budget",
    "timeout": "the context policy, the step budget, grounding stop rules",
    "precheck_skipped": "the compiled plan",
    "order_violation": "the compiled plan (prerequisites in order)",
    "write_never_made": "the compiled plan, or grounding",
    "step_never_made": "the compiled plan, or grounding",
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
    "no_symptom_in_results": "read two transcripts; ask the episode-auditor",
}
UNMAPPED = "read the permitted trace"


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
            if row.get("error") and not CATEGORY.match(str(row["error"])):
                raise SystemExit(f"{path}:{number}: `error` must be a category the adapter chose "
                                 "(lower case, digits, underscores), not the runner's text")
            for entry in (row.get("calls") or []) + (row.get("expected_actions") or []):
                if entry.get("requestor") not in REQUESTORS:
                    raise SystemExit(f"{path}:{number}: `requestor` must be one of {', '.join(REQUESTORS)}, "
                                     f"not {entry.get('requestor')!r}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no episodes")
    return rows


def family(entry):
    """The operation a call belongs to: the `family` the adapter wrote, or,
    without one, the tool's name less its leading verb and a trailing numeric
    suffix (approve_x_1234 and deny_x_1235 share `x`). Returns the family and
    whether it came from the name."""
    if entry.get("family"):
        return entry["family"], False
    parts = [p for p in (entry.get("tool") or entry.get("name") or "").split("_") if p]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) > 1:
        parts = parts[1:]
    return "_".join(parts), True


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
                wanted, by_name = family(entry)
                siblings = [c for c in calls if c.get("kind") == "write" and key_of(c) != key and family(c)[0] == wanted]
                heuristic = " (name heuristic)" if by_name or any(family(c)[1] for c in siblings) else ""
                cls = "wrong_sibling_tool" if siblings else "write_never_made"
                detail = ("a write of the same family made instead" + heuristic if siblings
                          else "reference step %d never attempted" % entry["index"])
            else:
                cls, detail = "step_never_made", "reference step %d (%s) never attempted" % (entry["index"], kind)
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


def reading(termination, limits, endings):
    """How a termination is read: `limit` when it contains one of the limit
    words, `ending` when it is one of the endings, else `unread`."""
    if any(word in termination for word in limits):
        return "limit"
    return "ending" if termination in endings else "unread"


def classify(row, limits, endings):
    """The observed symptoms of one failed episode: (turn or None, class, check or tool, detail)."""
    found = []
    if row.get("error"):
        found.append((None, str(row["error"]), None, "error category only; read the permitted trace"))
    termination = str(row.get("termination") or "").lower()
    read_as = reading(termination, limits, endings)
    if read_as == "limit":
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
        if untried and read_as == "ending" and not row.get("error"):
            found.append((None, "ended_early", None, "ended with %d reference step(s) untaken" % len(untried)))
    if not found:
        found.append((None, "no_symptom_in_results", None, "no symptom visible in the results file"))
    unique = {}
    for turn, cls, tool, detail in sorted(found, key=lambda f: (f[0] is None, f[0] or 0, f[1])):
        unique.setdefault((cls, tool), (turn, cls, tool, detail))
    return sorted(unique.values(), key=lambda f: (f[0] is None, f[0] or 0, f[1]))


def triage(rows, limits, endings):
    failed = [row for row in rows if not row["passed"]]
    histogram = Counter()
    tools = Counter()
    symptoms = []
    terminations = Counter(str(row.get("termination") or "").lower() for row in failed)
    for row in failed:
        found = classify(row, limits, endings)
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
        "terminations": {name: {"episodes": count, "read_as": reading(name, limits, endings)}
                         for name, count in terminations.most_common()},
        "symptoms": symptoms,
        "cost": {k: cost(k) for k in ("input_tokens", "tool_calls", "steps", "seconds")},
    }


def render(report, comparison):
    lines = [f"episodes {report['episodes']}, failed {report['failed']}",
             "observed symptom classes (episodes; action classes relative to the published reference route):"]
    for cls, count in report["classes"].items():
        lines.append(f"  {count:>3}  {cls:<28} -> usually {COMPONENT.get(cls, UNMAPPED)}")
    lines.append("checks or tools most involved: " + (", ".join(f"{t} ({n})" for t, n in report["tools"].items()) or "-"))
    lines.append("terminations of the failed episodes: " + (", ".join(
        f"{name or '(none)'} ({row['episodes']}, {row['read_as']})" for name, row in report["terminations"].items()) or "-"))
    if any(row["read_as"] == "unread" for row in report["terminations"].values()):
        lines.append("  an unread termination gives no timeout and no ended_early class: name the runner's words with "
                     "--limit-terminations or --endings")
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
    parser.add_argument("--limit-terminations", default=LIMIT_TERMINATIONS,
                        help="comma-separated words a termination contains when the runner's step, turn, time or cost limit ended the episode")
    parser.add_argument("--endings", default=ENDINGS,
                        help="comma-separated terminations that mean the episode simply ended (the scaffold or the other party stopped)")
    args = parser.parse_args()
    limits = [word.strip().lower() for word in args.limit_terminations.split(",") if word.strip()]
    endings = [word.strip().lower() for word in args.endings.split(",")]
    rows = read(args.path)
    report = triage(rows, limits, endings)
    comparison = compare(rows, read(args.against)) if args.against else None
    if args.json:
        json.dump({"triage": report, "comparison": comparison}, sys.stdout, indent=2)
        print()
    else:
        print(render(report, comparison))


if __name__ == "__main__":
    main()
