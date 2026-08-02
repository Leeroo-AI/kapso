#!/usr/bin/env python3
"""Ticket office + priority queues for Kaggle's per-account session limits.

Kaggle runs at most 2 GPU and 5 CPU sessions at once PER ACCOUNT, every lane
shares one account, and BOTH kinds of work consume a session:

  push   `kaggle kernels push` runs your kernel
         (hold the ticket from push until the kernel goes terminal)
  score  `kaggle competitions submit -k ... -v ...` RE-RUNS the kernel to
         score it (hold from submit until the submission leaves pending —
         scoring occupies a session for about the kernel's runtime)

Claim a ticket before either call. There is ONE QUEUE PER POOL (gpu, cpu),
ordered by priority tier then arrival:

    TASK=<the directory holding this file>
    T=$(python3 $TASK/kernel_slots.py acquire gpu push --ref <owner>/<slug> --lane <you>)
    kaggle kernels push -p kernel/      # ... poll status until terminal ...
    python3 $TASK/kernel_slots.py release "$T"

    T=$(python3 $TASK/kernel_slots.py acquire gpu score --ref <owner>/<slug> --lane <you>)
    kaggle competitions submit -c <comp> -k <owner>/<slug> -v <N> -f submission.csv -m "<idea>"
    # ... poll `kaggle competitions submissions` until it leaves pending ...
    python3 $TASK/kernel_slots.py release "$T"

    python3 $TASK/kernel_slots.py status     # queues + holders, both pools

Priority tiers (--priority), highest first:
    ship         deadline-critical shipping (endgame, harvest)
    first-score  scoring a version that has never scored — one step from value
    run          a normal kernel push (the default for push)
    reroll       resubmitting an already-scored version; debug pushes
Defaults: push -> run, score -> first-score. FIFO within a tier.

`acquire` BLOCKS until granted — a wait is backpressure, not failure, and never
a reason to weaken your recipe. It prints the queue state to stderr and the
ticket to stdout, so `$(...)` captures the ticket cleanly. Release promptly:
holding a ticket through your own analysis starves the other lanes.

Stale tickets are released against KAGGLE'S OWN TRUTH, not a blind timer: after
ttl_seconds a ticket becomes verify-eligible, and the next waiter checks the
kernel / submission state before releasing it. (A dead lane's kernel keeps
RUNNING on Kaggle, so timer-only reclaim over-granted the pool — run 5's lanes
hit `Maximum batch GPU session count of 2 reached` while holding tickets.)

Stdlib-only and standalone by design: lanes run in isolated session clones
where the kapso package may not be importable, so this is staged into the task
directory and invoked by path. The runner imports it directly for the harvest.
"""

import argparse
import fcntl
import json
import os
import re
import secrets
import subprocess
import sys
import time

CONFIG_NAME = ".kernel_slots_config.json"
LEDGER_NAME = ".kernel_slots.json"
LOCK_NAME = ".kernel_slots.lock"
KAGGLE_META_NAME = "kaggle.json"
POLL_SECONDS = 5.0
# A waiter that stopped heartbeating (its lane died mid-wait) must not hold a
# place in line; derived from the poll cadence, not a user knob.
WAITER_STALE_SECONDS = 6 * POLL_SECONDS

POOLS = ("gpu", "cpu")
KINDS = ("push", "score")
PRIORITY_TIERS = {"ship": 0, "first-score": 1, "run": 2, "reroll": 3}
DEFAULT_PRIORITY = {"push": "run", "score": "first-score"}
KERNEL_REF_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TERMINAL_KERNEL_STATUSES = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def _paths(task_dir: str) -> tuple:
    return (
        os.path.join(task_dir, CONFIG_NAME),
        os.path.join(task_dir, LEDGER_NAME),
        os.path.join(task_dir, LOCK_NAME),
    )


def load_limits(task_dir: str) -> dict:
    """Pool sizes + verification knobs, written by the runner from config.yaml."""
    config_path, _, _ = _paths(task_dir)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"{config_path} missing — the runner writes it from "
            "config.yaml session_budget.kernel_slots at launch"
        )
    with open(config_path) as handle:
        limits = json.load(handle)
    for key in ("gpu", "cpu", "ttl_seconds", "reap_interval_seconds",
                "verify_timeout_seconds"):
        if key not in limits:
            raise ValueError(f"{config_path} has no {key!r}")
    return limits


def empty_ledger() -> dict:
    return {
        "queue": {pool: [] for pool in POOLS},
        "tickets": {pool: [] for pool in POOLS},
        "last_reap": 0.0,
    }


def read_ledger(ledger_path: str) -> dict:
    """Current queues + tickets. Missing ledger is empty; corrupt raises."""
    if not os.path.isfile(ledger_path):
        return empty_ledger()
    with open(ledger_path) as handle:
        ledger = json.load(handle)
    for key in ("queue", "tickets", "last_reap"):
        if key not in ledger:
            raise ValueError(
                f"{ledger_path} is not a v2 ledger (missing {key!r}) — "
                "the runner deletes stale ledgers at launch"
            )
    return ledger


def _write_ledger(ledger_path: str, ledger: dict) -> None:
    with open(ledger_path, "w") as handle:
        json.dump(ledger, handle, indent=2)


def _with_lock(lock_path: str, action):
    """Run `action` holding an exclusive flock; always release it."""
    handle = open(lock_path, "w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    result = action()
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return result


def _prune_waiters(ledger: dict, now: float) -> None:
    for pool in POOLS:
        ledger["queue"][pool] = [
            w for w in ledger["queue"][pool]
            if now - w["last_seen"] < WAITER_STALE_SECONDS
        ]


def poll_once(task_dir: str, pool: str, kind: str, ref: str, lane: str,
              priority: str, waiter_id: str, now: float) -> "dict":
    """One atomic heartbeat-and-maybe-grant step; the whole decision is locked.

    Grants to the top `free` fresh waiters of the pool's queue (priority tier,
    then arrival), so one slow-polling head never strands a second free slot.
    Returns {"ticket": str|None, "snapshot": {...}}.
    """
    limits = load_limits(task_dir)
    _, ledger_path, lock_path = _paths(task_dir)

    def attempt():
        ledger = read_ledger(ledger_path)
        _prune_waiters(ledger, now)
        queue = ledger["queue"][pool]
        mine = next((w for w in queue if w["id"] == waiter_id), None)
        if mine is None:
            mine = {"id": waiter_id, "lane": lane, "kind": kind, "ref": ref,
                    "priority": priority, "enqueued": now, "last_seen": now}
            queue.append(mine)
        else:
            mine["last_seen"] = now
        queue.sort(key=lambda w: (PRIORITY_TIERS[w["priority"]], w["enqueued"]))
        free = limits[pool] - len(ledger["tickets"][pool])
        rank = next(i for i, w in enumerate(queue) if w["id"] == waiter_id)
        ticket = None
        if rank < free:
            ticket = f"{pool}-{kind}-{secrets.token_hex(6)}"
            ledger["tickets"][pool].append(
                {"ticket": ticket, "lane": lane, "kind": kind, "ref": ref,
                 "priority": priority, "acquired": now}
            )
            ledger["queue"][pool] = [w for w in queue if w["id"] != waiter_id]
        snapshot = {
            "in_use": len(ledger["tickets"][pool]),
            "limit": limits[pool],
            "holders": [f"{t['lane']}({t['kind']})"
                        for t in ledger["tickets"][pool]],
            "queued": [f"{w['lane']}({w['priority']})"
                       for w in ledger["queue"][pool]],
        }
        _write_ledger(ledger_path, ledger)
        return {"ticket": ticket, "snapshot": snapshot}

    return _with_lock(lock_path, attempt)


def _withdraw(task_dir: str, pool: str, waiter_id: str) -> None:
    """Leave the queue (acquire gave up); never strand a phantom waiter."""
    _, ledger_path, lock_path = _paths(task_dir)

    def attempt():
        ledger = read_ledger(ledger_path)
        ledger["queue"][pool] = [
            w for w in ledger["queue"][pool] if w["id"] != waiter_id
        ]
        _write_ledger(ledger_path, ledger)

    _with_lock(lock_path, attempt)


# ---------------------------------------------------------------------------
# Truth-based reap: stale tickets are released only when Kaggle confirms the
# session is over. Network runs OUTSIDE the flock; only the decisions lock.
# ---------------------------------------------------------------------------

def kaggle_kernel_status(ref: str, timeout_seconds: float) -> str:
    """Lowercased kernel status word from `kaggle kernels status`."""
    proc = subprocess.run(
        ["kaggle", "kernels", "status", ref],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    match = re.search(r'"(\w+)"', proc.stdout + proc.stderr)
    if not match:
        raise RuntimeError(
            f"kaggle kernels status {ref} gave no status "
            f"(exit {proc.returncode}): {(proc.stdout + proc.stderr)[-300:]}"
        )
    return match.group(1).lower()


def kaggle_pending_submissions(task_dir: str, timeout_seconds: float) -> int:
    """How many of the account's submissions are still scoring."""
    kaggle_meta = os.path.join(task_dir, KAGGLE_META_NAME)
    if not os.path.isfile(kaggle_meta):
        raise FileNotFoundError(
            f"{kaggle_meta} missing — score tickets cannot be verified "
            "without the competition slug"
        )
    with open(kaggle_meta) as handle:
        competition = json.load(handle)["competition"]
    proc = subprocess.run(
        ["kaggle", "competitions", "submissions", competition,
         "--format", "json", "-q"],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kaggle competitions submissions exited {proc.returncode}: "
            f"{proc.stderr[-300:]}"
        )
    start = proc.stdout.find("[")
    if start < 0:
        raise ValueError(
            f"no JSON payload in submissions output: {proc.stdout[:200]!r}")
    rows = json.loads(proc.stdout[start:])
    return sum(1 for row in rows if "pending" in str(row.get("status", "")).lower())


def maybe_reap(task_dir: str, now: float,
               kernel_status_fn=kaggle_kernel_status,
               pending_fn=kaggle_pending_submissions) -> list:
    """Verify-and-release stale tickets; returns the released ticket ids.

    Rate-limited through the ledger's last_reap stamp so eight waiting lanes
    do not each hammer Kaggle every poll. push tickets are released when their
    kernel is terminal; score tickets when the account has nothing pending
    (scoring runs cannot be attributed per-ticket from the API).
    """
    limits = load_limits(task_dir)
    _, ledger_path, lock_path = _paths(task_dir)

    def claim():
        ledger = read_ledger(ledger_path)
        stale = [
            dict(t) for pool in POOLS for t in ledger["tickets"][pool]
            if now - t["acquired"] > limits["ttl_seconds"]
        ]
        if not stale or now - ledger["last_reap"] < limits["reap_interval_seconds"]:
            return []
        ledger["last_reap"] = now
        _write_ledger(ledger_path, ledger)
        return stale

    stale = _with_lock(lock_path, claim)
    if not stale:
        return []

    verify_timeout = limits["verify_timeout_seconds"]
    releasable = []
    pending = None
    for ticket in stale:
        if ticket["kind"] == "push":
            if kernel_status_fn(ticket["ref"], verify_timeout) in TERMINAL_KERNEL_STATUSES:
                releasable.append(ticket["ticket"])
        else:
            if pending is None:
                pending = pending_fn(task_dir, verify_timeout)
            if pending == 0:
                releasable.append(ticket["ticket"])
    if not releasable:
        return []

    def drop():
        ledger = read_ledger(ledger_path)
        for pool in POOLS:
            ledger["tickets"][pool] = [
                t for t in ledger["tickets"][pool]
                if t["ticket"] not in releasable
            ]
        _write_ledger(ledger_path, ledger)
        return releasable

    return _with_lock(lock_path, drop)


def acquire_blocking(task_dir: str, pool: str, kind: str, ref: str, lane: str,
                     priority: "str | None" = None,
                     wait_seconds: float = 1800.0,
                     kernel_status_fn=kaggle_kernel_status,
                     pending_fn=kaggle_pending_submissions) -> "str | None":
    """Queue for a slot; the ticket, or None when wait_seconds expires.

    Expiry returns None instead of raising so callers with their own budget
    (the harvest) degrade to "skip the rest" without a banned try/except; the
    CLI turns None into a non-zero exit for the lanes.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not KERNEL_REF_PATTERN.match(ref):
        raise ValueError(
            f"--ref must be the kernel ref (<owner>/<slug>), got {ref!r} — "
            "the reap verifies a stale ticket against this kernel's status"
        )
    priority = priority or DEFAULT_PRIORITY[kind]
    if priority not in PRIORITY_TIERS:
        raise ValueError(
            f"priority must be one of {sorted(PRIORITY_TIERS)}, got {priority!r}")
    waiter_id = f"w-{secrets.token_hex(6)}"
    deadline = time.time() + wait_seconds
    announced = False
    while True:
        result = poll_once(task_dir, pool, kind, ref, lane, priority,
                           waiter_id, time.time())
        if result["ticket"]:
            return result["ticket"]
        snapshot = result["snapshot"]
        if time.time() >= deadline:
            _withdraw(task_dir, pool, waiter_id)
            print(f"[kernel_slots] no {pool} slot after {wait_seconds:.0f}s — "
                  f"held by {snapshot['holders']}, queue {snapshot['queued']}",
                  file=sys.stderr)
            return None
        if not announced:
            # To stderr: stdout is the ticket, so callers capture it cleanly.
            print(f"[kernel_slots] {pool} pool full "
                  f"({snapshot['in_use']}/{snapshot['limit']}, held by "
                  f"{snapshot['holders']}), queue {snapshot['queued']} — "
                  f"queueing as {priority}; backpressure, not failure",
                  file=sys.stderr)
            announced = True
        maybe_reap(task_dir, time.time(),
                   kernel_status_fn=kernel_status_fn, pending_fn=pending_fn)
        time.sleep(POLL_SECONDS)


def release(task_dir: str, ticket: str) -> bool:
    """Hand a slot back. False if the ticket was already gone (reaped)."""
    _, ledger_path, lock_path = _paths(task_dir)

    def attempt():
        ledger = read_ledger(ledger_path)
        found = False
        for pool in POOLS:
            kept = [t for t in ledger["tickets"][pool] if t["ticket"] != ticket]
            found = found or len(kept) != len(ledger["tickets"][pool])
            ledger["tickets"][pool] = kept
        _write_ledger(ledger_path, ledger)
        return found

    return _with_lock(lock_path, attempt)


def status(task_dir: str) -> dict:
    limits = load_limits(task_dir)
    _, ledger_path, lock_path = _paths(task_dir)

    def snapshot():
        ledger = read_ledger(ledger_path)
        _prune_waiters(ledger, time.time())
        _write_ledger(ledger_path, ledger)
        return ledger

    ledger = _with_lock(lock_path, snapshot)
    now = time.time()
    return {
        pool: {
            "in_use": len(ledger["tickets"][pool]),
            "limit": limits[pool],
            "tickets": [
                {"lane": t["lane"], "kind": t["kind"], "ref": t["ref"],
                 "held_seconds": int(now - t["acquired"])}
                for t in ledger["tickets"][pool]
            ],
            "queue": [
                {"lane": w["lane"], "priority": w["priority"],
                 "waited_seconds": int(now - w["enqueued"])}
                for w in sorted(
                    ledger["queue"][pool],
                    key=lambda w: (PRIORITY_TIERS[w["priority"]], w["enqueued"]),
                )
            ],
        }
        for pool in POOLS
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default=None,
                        choices=["acquire", "release", "status"])
    parser.add_argument("target", nargs="?", default="",
                        help="pool (gpu|cpu) for acquire; ticket for release")
    parser.add_argument("kind", nargs="?", default="",
                        help="push|score, for acquire")
    parser.add_argument("--ref", default="",
                        help="kernel ref <owner>/<slug> this session is for")
    parser.add_argument("--task-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="directory holding the ledger (default: this script's dir)")
    parser.add_argument("--lane", default="unknown", help="lane id, for status output")
    parser.add_argument("--priority", default=None, choices=sorted(PRIORITY_TIERS),
                        help="ship|first-score|run|reroll "
                             "(default: push->run, score->first-score)")
    parser.add_argument("--wait-seconds", type=float, default=1800.0,
                        help="how long to queue for a slot")
    args = parser.parse_args()

    if args.action is None:
        # "Run it with no arguments for the protocol" — the affordance RULES.md
        # promises; run 5's lanes learned the protocol by reading this file.
        print(__doc__)
        return

    if args.action == "status":
        print(json.dumps(status(args.task_dir), indent=2))
        return

    if args.action == "release":
        if not args.target:
            sys.exit("release needs a ticket")
        released = release(args.task_dir, args.target)
        print(f"released {args.target}" if released
              else f"{args.target} was already reclaimed — slot is free")
        return

    ticket = acquire_blocking(args.task_dir, args.target, args.kind, args.ref,
                              args.lane, args.priority, args.wait_seconds)
    if ticket is None:
        sys.exit(1)
    print(ticket)


if __name__ == "__main__":
    main()
