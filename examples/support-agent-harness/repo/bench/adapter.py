"""The adapter over the runner's public output: a results file of the task runner converted
to the harness-evaluation convention, one episode per conversation.

Only what the runner's public diagnostic contract exposes is converted: the outcome, the
termination, an error reduced to a category, the counts and the tokens, the calls made (names,
tool names, kinds, argument names) and, for the development tasks, the published reference
route as names, kinds and matched flags with the names of the arguments that differ. No
expected value, no expected argument, no assertion text is written. A conversation whose task
is not in the allowed set is refused, so a sealed task can never be converted by mistake, and
an episode is never silently omitted.

    from adapter import convert, summarize
    rows = convert(results_path, allowed_ids)
    summary = summarize(rows, k=1)
"""
import json
import statistics
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

WRAPPERS = ("call_discoverable_agent_tool", "unlock_discoverable_agent_tool",
            "give_discoverable_user_tool", "call_discoverable_user_tool")
READ_PREFIXES = ("get_", "list_", "lookup_", "check_", "search_", "view_", "find_", "calculate_", "verify_")
# An infrastructure error is reduced to one of these categories; the runner's text stays in its own files.
ERROR_CATEGORIES = (
    ("overflow", ("contextwindowexceeded", "context window", "context length", "maximum context")),
    ("empty_reply", ("must have either content or tool_calls", "empty")),
    ("timeout", ("timeout", "timed out")),
)


def error_category(text):
    if not text:
        return None
    lowered = str(text).lower()
    for category, markers in ERROR_CATEGORIES:
        if any(marker in lowered for marker in markers):
            return category
    return "endpoint_error"


def parsed_arguments(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {"_unparsable": True}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def tool_of(name, args):
    if name in WRAPPERS:
        return args.get("agent_tool_name") or args.get("discoverable_tool_name")
    return None


def kind_of(requestor, name, tool):
    if requestor == "user":
        return "user"
    if name.startswith("KB_") or name in ("shell", "grep"):
        return "search"
    if name == "get_current_time":
        return "meta"
    if name == "unlock_discoverable_agent_tool":
        return "unlock"
    if name == "give_discoverable_user_tool":
        return "handoff"
    if name == "call_discoverable_agent_tool":
        return "read" if (tool or "").startswith(READ_PREFIXES) else "write"
    if name.startswith(READ_PREFIXES):
        return "read"
    return "write"


def convert_one(simulation):
    reward = simulation.get("reward_info") or {}
    calls, made = [], []   # the convention's rows; (requestor, name, tool, parsed args) for matching only
    assistant_turns = 0
    for message in simulation.get("messages") or []:
        role = message.get("role")
        if role == "assistant":
            assistant_turns += 1
        if role in ("assistant", "user") and message.get("tool_calls"):
            for call in message["tool_calls"]:
                raw = call.get("arguments") or {}
                tool = tool_of(call["name"], raw if isinstance(raw, dict) else {})
                args = parsed_arguments(raw.get("arguments") if call["name"] in WRAPPERS and isinstance(raw, dict) else raw)
                made.append((role, call["name"], tool, args))
                calls.append({"turn": message.get("turn_idx"), "requestor": role, "name": call["name"], "tool": tool,
                              "kind": kind_of(role, call["name"], tool), "arg_keys": sorted(args.keys())})
        if role == "tool" and message.get("error"):
            calls.append({"turn": message.get("turn_idx"), "requestor": "tool", "name": "error", "tool": None,
                          "kind": "error", "arg_keys": [], "error": error_category(message.get("content"))})
    expected = []
    for index, check in enumerate(reward.get("action_checks") or []):
        action = check["action"]
        raw = action.get("arguments") or {}
        tool = tool_of(action["name"], raw if isinstance(raw, dict) else {})
        wanted = parsed_arguments(raw.get("arguments") if action["name"] in WRAPPERS and isinstance(raw, dict) else raw)
        requestor = "assistant" if action["requestor"] == "assistant" else "user"
        same = [args for (who, name, made_tool, args) in made if who == requestor and name == action["name"] and made_tool == tool]
        identical = any(args == wanted for args in same)
        diff_keys = []
        if same and not identical:
            best = None
            for args in same:
                differing = sorted(key for key in set(args) | set(wanted) if args.get(key) != wanted.get(key))
                if best is None or len(differing) < len(best):
                    best = differing
            diff_keys = best or []
        expected.append({"index": index, "requestor": requestor, "name": action["name"], "tool": tool,
                         "kind": kind_of(requestor, action["name"], tool),
                         "matched": bool(check.get("action_match")) or identical,
                         "attempted": bool(same), "diff_keys": diff_keys})
    usage = [message.get("usage") or {} for message in simulation.get("messages") or [] if message.get("role") == "assistant"]
    info = simulation.get("info") if isinstance(simulation.get("info"), dict) else {}
    return {
        "id": simulation["task_id"],
        "trial": simulation.get("trial"),
        "passed": float(reward.get("reward", 0.0)) >= 1.0,
        "basis": reward.get("reward_basis"),
        "termination": simulation.get("termination_reason"),
        "error": error_category(info.get("error")),
        "steps": assistant_turns,
        "messages": len(simulation.get("messages") or []),
        "tool_calls": sum(1 for call in calls if call["requestor"] == "assistant" and call["kind"] != "error"),
        "input_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
        "output_tokens": sum(item.get("completion_tokens", 0) for item in usage),
        "max_prompt_tokens": max((item.get("prompt_tokens", 0) for item in usage), default=0),
        "seconds": round(float(simulation.get("duration") or 0.0), 1),
        "expected_actions": expected,
        "calls": calls,
    }


def convert(results_path, allowed_ids):
    """Every conversation of the results file, refused when one is outside the allowed tasks."""
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    simulations = data["simulations"] if isinstance(data, dict) else data
    rows = []
    for simulation in simulations:
        if simulation["task_id"] not in allowed_ids:
            raise SystemExit(f"refusing to convert a task outside the allowed set: {simulation['task_id']}")
        rows.append(convert_one(simulation))
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def summarize(rows, k):
    """pass^1, pass^k, the per-task passes, the error count and the cost lines: the same shape the
    harness's pass^k script writes, so the sessions that read a summary read one shape."""
    by_task = defaultdict(list)
    trial_ids = set()
    for row in rows:
        by_task[row["id"]].append(row["passed"])
        trial_ids.add(row["trial"])
    n_trials = len(trial_ids)
    tasks = sorted(by_task)
    passes = {task: sum(by_task[task]) for task in tasks}
    rates = [passes[task] / n_trials for task in tasks]
    pass_k = statistics.mean(comb(passes[task], k) / comb(n_trials, k) for task in tasks) if n_trials >= k else None

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
        "pass_1": round(statistics.mean(rates), 4),
        "pass_k": round(pass_k, 4) if pass_k is not None else None,
        "pass_1_task_se": round(statistics.stdev(rates) / sqrt(len(tasks)), 4) if len(tasks) > 1 else 0.0,
        "mean_score": None,
        "passes_total": sum(passes.values()),
        "episodes": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "per_task": passes,
        "cost": {key: line(key) for key in ("input_tokens", "output_tokens", "tool_calls", "steps", "seconds")},
    }
