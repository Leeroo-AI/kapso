from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    RunActionCodingAgentContractError,
)
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    coding_agent_cli_preflight_command,
)
from kapso.cross_run.launch.run_action_coding_agent_runtime import (
    PROVIDER_SANDBOX_EXECUTABLE,
    ProviderSandboxPathRule,
    RunActionCodingAgentRuntimeError,
    apply_provider_landlock,
    coding_agent_provider_sandbox_environment,
    coding_agent_provider_sandbox_command,
)
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)

_READ_ONLY_LANDLOCK_ACCESS = (1 << 0) | (1 << 2) | (1 << 3)
_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def test_provider_environment_is_complete_and_contains_no_ambient_authority():
    assert dict(coding_agent_provider_sandbox_environment()) == {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/kapso/tmp/provider-home",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TERM": "dumb",
        "TMPDIR": "/kapso/tmp/provider-home",
    }


def test_sandbox_command_binds_complete_policy_and_provider_argv():
    request = run_action_request(interpretation_policy())
    command = ("/usr/bin/codex", "--version")

    projected = coding_agent_provider_sandbox_command(request, command)

    assert projected == (
        PROVIDER_SANDBOX_EXECUTABLE,
        "--landlock-abi-version",
        "7",
        "--supervisor-user-id",
        "1000",
        "--supervisor-group-id",
        "1000",
        "--provider-user-id",
        "1001",
        "--provider-group-id",
        "1001",
        "--workspace-access",
        "read_only",
        "--",
        *command,
    )


def test_sandbox_command_enforces_the_complete_argv_bound():
    policy = interpretation_policy(
        maximum_response_schema_bytes=128,
        maximum_cli_argument_bytes=245,
    )
    request = run_action_request(policy)

    with pytest.raises(
        RunActionCodingAgentRuntimeError,
        match="sandbox argv exceeds",
    ):
        coding_agent_provider_sandbox_command(
            request,
            coding_agent_cli_preflight_command(request),
        )


def test_sandbox_command_rejects_an_absolute_but_unbound_provider_argv():
    request = run_action_request(interpretation_policy())

    with pytest.raises(
        RunActionCodingAgentRuntimeError,
        match="request-derived projection",
    ):
        coding_agent_provider_sandbox_command(
            request,
            ("/usr/bin/codex", "exec", "substituted"),
        )


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"provider_user_id": 1000}, "identities must differ"),
        ({"provider_group_id": 1000}, "identities must differ"),
        ({"landlock_abi_version": 5}, "differs from the implemented policy"),
        ({"landlock_abi_version": 8}, "differs from the implemented policy"),
    ),
)
def test_interpretation_policy_rejects_unseparated_runtime_authority(
    changes,
    message,
):
    with pytest.raises(RunActionCodingAgentContractError, match=message):
        replace(interpretation_policy(), **changes)


def test_landlock_rejects_a_runtime_abi_substitution(tmp_path):
    allowed = tmp_path.resolve()
    observed = load_config(_CANONICAL_CONFIG_PATH)["cross_run"]["launch"][
        "coding_agent_landlock_abi_version"
    ]

    with pytest.raises(
        RunActionCodingAgentRuntimeError,
        match="ABI differs",
    ):
        apply_provider_landlock(
            expected_abi_version=observed + 1,
            path_rules=(
                ProviderSandboxPathRule(
                    str(allowed),
                    _READ_ONLY_LANDLOCK_ACCESS,
                ),
            ),
        )


def test_landlock_allows_only_the_descriptor_bound_hierarchy(tmp_path):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    (allowed / "record").write_text("allowed", encoding="utf-8")
    (denied / "record").write_text("denied", encoding="utf-8")
    abi = load_config(_CANONICAL_CONFIG_PATH)["cross_run"]["launch"][
        "coding_agent_landlock_abi_version"
    ]
    program = (
        "from kapso.cross_run.launch.run_action_coding_agent_runtime import "
        "ProviderSandboxPathRule,apply_provider_landlock;"
        "import pathlib,sys;"
        "apply_provider_landlock(expected_abi_version=int(sys.argv[1]),"
        "path_rules=(ProviderSandboxPathRule(sys.argv[2],13),));"
        "print(pathlib.Path(sys.argv[3]).read_text(encoding='utf-8'))"
    )

    allowed_completion = subprocess.run(
        (
            sys.executable,
            "-c",
            program,
            str(abi),
            str(allowed.resolve()),
            str((allowed / "record").resolve()),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    denied_completion = subprocess.run(
        (
            sys.executable,
            "-c",
            program,
            str(abi),
            str(allowed.resolve()),
            str((denied / "record").resolve()),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert allowed_completion.returncode == 0
    assert allowed_completion.stdout == b"allowed\n"
    assert allowed_completion.stderr == b""
    assert denied_completion.returncode != 0
    assert denied_completion.stdout == b""
    assert b"PermissionError" in denied_completion.stderr


def test_landlock_signal_scope_blocks_the_provider_from_its_parent(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    abi = load_config(_CANONICAL_CONFIG_PATH)["cross_run"]["launch"][
        "coding_agent_landlock_abi_version"
    ]
    program = (
        "from kapso.cross_run.launch.run_action_coding_agent_runtime import "
        "ProviderSandboxPathRule,apply_provider_landlock;"
        "import os,sys;"
        "apply_provider_landlock(expected_abi_version=int(sys.argv[1]),"
        "path_rules=(ProviderSandboxPathRule(sys.argv[2],13),));"
        "os.kill(os.getppid(),0)"
    )

    completion = subprocess.run(
        (
            sys.executable,
            "-c",
            program,
            str(abi),
            str(allowed.resolve()),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completion.returncode != 0
    assert completion.stdout == b""
    assert b"PermissionError" in completion.stderr
