"""Exact root-to-supervisor bootstrap for the coding-agent main occurrence."""

from __future__ import annotations

import argparse
import os
import re

from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentInterpretationPolicy,
)

CODING_AGENT_SUPERVISOR_EXECUTABLE = "/usr/local/bin/kapso-coding-agent-supervisor"
CODING_AGENT_CONSUMER_EXECUTABLE = "/usr/local/bin/kapso-coding-agent-consumer"
_SETPRIV_EXECUTABLE = "/usr/bin/setpriv"
_CAPABILITY_STATUS_PATTERN = re.compile(
    rb"^(CapInh|CapPrm|CapEff|CapBnd|CapAmb):[\t ]+([0-9a-f]{16})$"
)
_ZERO_CAPABILITIES = b"0000000000000000"
_TRANSITION_CAPABILITIES = b"00000000000001e0"
_LINUX_IDENTITY_MAXIMUM = 2_147_483_647


class RunActionCodingAgentSupervisorError(RuntimeError):
    """The main occurrence cannot prove its exact temporary transition power."""


def coding_agent_supervisor_command(
    policy: CodingAgentInterpretationPolicy,
) -> tuple[str, ...]:
    """Project the exact root-bootstrap argv for one coding-agent request policy."""

    if type(policy) is not CodingAgentInterpretationPolicy:
        raise RunActionCodingAgentSupervisorError(
            "coding-agent supervisor requires one interpretation policy"
        )
    return (
        CODING_AGENT_SUPERVISOR_EXECUTABLE,
        "--supervisor-user-id",
        str(policy.supervisor_user_id),
        "--supervisor-group-id",
        str(policy.supervisor_group_id),
        "--provider-user-id",
        str(policy.provider_user_id),
        "--provider-group-id",
        str(policy.provider_group_id),
    )


def main() -> None:
    """Verify the four-capability root state and irreversibly enter the supervisor."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-user-id", type=int, required=True)
    parser.add_argument("--supervisor-group-id", type=int, required=True)
    parser.add_argument("--provider-user-id", type=int, required=True)
    parser.add_argument("--provider-group-id", type=int, required=True)
    arguments = parser.parse_args()
    identity_values = (
        arguments.supervisor_user_id,
        arguments.supervisor_group_id,
        arguments.provider_user_id,
        arguments.provider_group_id,
    )
    if (
        any(not 0 < value <= _LINUX_IDENTITY_MAXIMUM for value in identity_values)
        or arguments.supervisor_user_id == arguments.provider_user_id
        or arguments.supervisor_group_id == arguments.provider_group_id
        or os.geteuid() != 0
        or os.getegid() != 0
        or os.getgroups() != [0, arguments.provider_group_id]
    ):
        raise RunActionCodingAgentSupervisorError(
            "coding-agent supervisor bootstrap identity is not exact"
        )
    capability_fields = {
        match.group(1): match.group(2)
        for line in open("/proc/self/status", "rb").read().splitlines()
        if (match := _CAPABILITY_STATUS_PATTERN.fullmatch(line)) is not None
    }
    if capability_fields != {
        b"CapInh": _ZERO_CAPABILITIES,
        b"CapPrm": _TRANSITION_CAPABILITIES,
        b"CapEff": _TRANSITION_CAPABILITIES,
        b"CapBnd": _TRANSITION_CAPABILITIES,
        b"CapAmb": _ZERO_CAPABILITIES,
    }:
        raise RunActionCodingAgentSupervisorError(
            "coding-agent supervisor bootstrap capability set is not exact"
        )
    capability_names = "+kill,+setgid,+setpcap,+setuid"
    command = (
        _SETPRIV_EXECUTABLE,
        f"--reuid={arguments.supervisor_user_id}",
        f"--regid={arguments.supervisor_group_id}",
        (f"--groups={arguments.supervisor_group_id}," f"{arguments.provider_group_id}"),
        f"--inh-caps={capability_names}",
        f"--ambient-caps={capability_names}",
        "--no-new-privs",
        CODING_AGENT_CONSUMER_EXECUTABLE,
    )
    os.execve(_SETPRIV_EXECUTABLE, command, {"PATH": "/usr/bin:/bin"})


__all__ = [
    "coding_agent_supervisor_command",
    "CODING_AGENT_CONSUMER_EXECUTABLE",
    "CODING_AGENT_SUPERVISOR_EXECUTABLE",
    "main",
    "RunActionCodingAgentSupervisorError",
]


if __name__ == "__main__":
    main()
