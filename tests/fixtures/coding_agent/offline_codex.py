#!/usr/bin/python3
"""Deterministic offline Codex wire fixture for the real container boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if tuple(sys.argv[1:]) == ("--version",):
        sys.stdout.write("codex-cli 0.144.1\n")
        return
    prompt = sys.stdin.read()
    arguments = tuple(sys.argv[1:])
    final_path = Path(arguments[arguments.index("--output-last-message") + 1])
    workspace = Path(arguments[arguments.index("--cd") + 1])
    if (
        os.geteuid() != 1001
        or os.getegid() != 1001
        or os.getgroups()
        or (workspace / ".git").exists()
    ):
        raise RuntimeError("offline provider received expanded authority")
    denied_paths = (
        "/kapso/workspace/.git",
        "/kapso/input/request.blob",
        "/run/docker.sock",
    )
    denied_probes = tuple(
        subprocess.run(
            ("/usr/bin/stat", path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        for path in denied_paths
    )
    if any(probe.returncode == 0 for probe in denied_probes):
        raise RuntimeError("offline provider resolved a protected authority")
    capabilities = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines()
        if line.startswith(("CapInh:", "CapPrm:", "CapEff:", "CapBnd:", "CapAmb:"))
    }
    if set(capabilities.values()) != {"0000000000000000"}:
        raise RuntimeError("offline provider retained a capability")
    write_probe = subprocess.run(
        ("/usr/bin/touch", str(workspace / "read-only-probe")),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if "OFFLINE_EDIT" in prompt:
        proposal = workspace / "proposal.py"
        proposal.write_bytes(proposal.read_bytes() + b"# offline edit\n")
    elif write_probe.returncode == 0:
        raise RuntimeError("offline read-only provider wrote its workspace")
    structured = {"answer": "offline boundary passed"}
    final_path.write_text(
        json.dumps(structured, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    events = (
        {"type": "thread.started", "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": json.dumps(structured, sort_keys=True, separators=(",", ":")),
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_output_tokens": 0,
            },
        },
    )
    for event in events:
        sys.stdout.write(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        )


if __name__ == "__main__":
    main()
