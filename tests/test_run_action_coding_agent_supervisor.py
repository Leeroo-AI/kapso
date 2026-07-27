from __future__ import annotations

import pytest

from kapso.cross_run.launch.run_action_coding_agent_supervisor import (
    CODING_AGENT_SUPERVISOR_EXECUTABLE,
    RunActionCodingAgentSupervisorError,
    coding_agent_supervisor_command,
)
from test_run_action_coding_agent_contracts import interpretation_policy


def test_supervisor_command_threads_all_request_bound_identities():
    policy = interpretation_policy(
        supervisor_user_id=1100,
        supervisor_group_id=1200,
        provider_user_id=1300,
        provider_group_id=1400,
    )

    assert coding_agent_supervisor_command(policy) == (
        CODING_AGENT_SUPERVISOR_EXECUTABLE,
        "--supervisor-user-id",
        "1100",
        "--supervisor-group-id",
        "1200",
        "--provider-user-id",
        "1300",
        "--provider-group-id",
        "1400",
    )


def test_supervisor_command_rejects_every_other_object():
    with pytest.raises(
        RunActionCodingAgentSupervisorError,
        match="interpretation policy",
    ):
        coding_agent_supervisor_command(object())
