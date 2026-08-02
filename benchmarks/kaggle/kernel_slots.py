#!/usr/bin/env python3
"""Cross-lane ticket office for Kaggle's concurrent-session limits.

Kaggle allows only a fixed number of notebook sessions running at once PER
ACCOUNT (2 GPU, 5 CPU), and every parallel lane shares one account. Without
coordination the lanes race: each pushes, gets `Maximum batch GPU session count
of 2 reached`, and retries blind, so the winner is whoever happens to poll when
a slot frees rather than whoever waited longest (measured: 36 cap errors in one
run, retries 14 deep).

This is a flock'd ledger of slot tickets over the shared task directory. A lane
holds a ticket from `kernels push` until its kernel finishes, so the ledger
mirrors what Kaggle is actually running. A crashed lane cannot park a slot
forever: tickets older than the TTL are reclaimed.

Deliberately stdlib-only and standalone — lanes run in isolated session clones
with no guarantee the kapso package is importable, so this is staged into the
task directory and invoked by path.

Usage (see benchmarks/kaggle/RULES.md for the lane protocol):
    python3 kernel_slots.py acquire gpu --lane lane3          # blocks
    python3 kernel_slots.py release <ticket>
    python3 kernel_slots.py status
"""

import argparse
import fcntl
import json
import os
import secrets
import sys
import time

CONFIG_NAME = ".kernel_slots_config.json"
LEDGER_NAME = ".kernel_slots.json"
LOCK_NAME = ".kernel_slots.lock"
POLL_SECONDS = 5.0


def _paths(task_dir: str) -> tuple:
    return (
        os.path.join(task_dir, CONFIG_NAME),
        os.path.join(task_dir, LEDGER_NAME),
        os.path.join(task_dir, LOCK_NAME),
    )


def load_limits(task_dir: str) -> dict:
    """Pool sizes + TTL, written by the runner from config.yaml."""
    config_path, _, _ = _paths(task_dir)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"{config_path} missing — the runner writes it from "
            "config.yaml session_budget.kernel_slots at launch"
        )
    with open(config_path) as handle:
        limits = json.load(handle)
    for key in ("gpu", "cpu", "ttl_seconds"):
        if key not in limits:
            raise ValueError(f"{config_path} has no {key!r}")
    return limits


def read_ledger(ledger_path: str) -> dict:
    """Current tickets. A missing ledger is an empty one; a corrupt one raises."""
    if not os.path.isfile(ledger_path):
        return {"gpu": [], "cpu": []}
    with open(ledger_path) as handle:
        ledger = json.load(handle)
    for pool in ("gpu", "cpu"):
        ledger.setdefault(pool, [])
    return ledger


def prune(ledger: dict, ttl_seconds: float, now: float) -> dict:
    """Reclaim tickets from lanes that died without releasing."""
    for pool, held in ledger.items():
        ledger[pool] = [t for t in held if now - t["acquired"] < ttl_seconds]
    return ledger


def _with_lock(lock_path: str, action):
    """Run `action` holding an exclusive flock; always release it."""
    handle = open(lock_path, "w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    result = action()
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return result


def try_acquire(task_dir: str, pool: str, lane: str) -> "str | None":
    """One non-blocking attempt: a ticket string, or None when the pool is full."""
    limits = load_limits(task_dir)
    _, ledger_path, lock_path = _paths(task_dir)

    def attempt():
        ledger = prune(read_ledger(ledger_path), limits["ttl_seconds"], time.time())
        if len(ledger[pool]) >= limits[pool]:
            with open(ledger_path, "w") as handle:
                json.dump(ledger, handle, indent=2)
            return None
        ticket = f"{pool}-{secrets.token_hex(6)}"
        ledger[pool].append(
            {"ticket": ticket, "lane": lane, "acquired": time.time()}
        )
        with open(ledger_path, "w") as handle:
            json.dump(ledger, handle, indent=2)
        return ticket

    return _with_lock(lock_path, attempt)


def release(task_dir: str, ticket: str) -> bool:
    """Hand a slot back. False if the ticket was already gone (TTL-reclaimed)."""
    _, ledger_path, lock_path = _paths(task_dir)

    def attempt():
        ledger = read_ledger(ledger_path)
        found = False
        for pool, held in ledger.items():
            kept = [t for t in held if t["ticket"] != ticket]
            found = found or len(kept) != len(held)
            ledger[pool] = kept
        with open(ledger_path, "w") as handle:
            json.dump(ledger, handle, indent=2)
        return found

    return _with_lock(lock_path, attempt)


def status(task_dir: str) -> dict:
    limits = load_limits(task_dir)
    _, ledger_path, lock_path = _paths(task_dir)
    ledger = _with_lock(
        lock_path,
        lambda: prune(read_ledger(ledger_path), limits["ttl_seconds"], time.time()),
    )
    return {
        pool: {"in_use": len(ledger[pool]), "limit": limits[pool],
               "lanes": [t["lane"] for t in ledger[pool]]}
        for pool in ("gpu", "cpu")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "release", "status"])
    parser.add_argument("target", nargs="?", default="",
                        help="pool (gpu|cpu) for acquire; ticket for release")
    parser.add_argument("--task-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="directory holding the ledger (default: this script's dir)")
    parser.add_argument("--lane", default="unknown", help="lane id, for status output")
    parser.add_argument("--wait-seconds", type=float, default=1800.0,
                        help="how long to queue for a slot; 0 = one attempt")
    args = parser.parse_args()

    if args.action == "status":
        print(json.dumps(status(args.task_dir), indent=2))
        return

    if args.action == "release":
        if not args.target:
            sys.exit("release needs a ticket")
        released = release(args.task_dir, args.target)
        print(f"released {args.target}" if released
              else f"{args.target} was already reclaimed (TTL) — slot is free")
        return

    if args.target not in ("gpu", "cpu"):
        sys.exit("acquire needs a pool: gpu or cpu")
    deadline = time.time() + args.wait_seconds
    announced = False
    while True:
        ticket = try_acquire(args.task_dir, args.target, args.lane)
        if ticket:
            print(ticket)
            return
        if time.time() >= deadline:
            sys.exit(
                f"no {args.target} slot after {args.wait_seconds:.0f}s — "
                "every slot is held by a running kernel"
            )
        if not announced:
            busy = status(args.task_dir)[args.target]
            # To stderr: stdout is the ticket, so callers can capture it cleanly.
            print(f"[kernel_slots] {args.target} pool full "
                  f"({busy['in_use']}/{busy['limit']}, held by {busy['lanes']}) — "
                  f"queueing, this is backpressure not failure", file=sys.stderr)
            announced = True
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
