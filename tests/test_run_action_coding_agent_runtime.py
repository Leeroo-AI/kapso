from __future__ import annotations

import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    RunActionCodingAgentContractError,
)
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    coding_agent_cli_preflight_command,
)
from kapso.cross_run.launch.run_action_coding_agent_layout import (
    coding_agent_provider_environment,
)
from kapso.cross_run.launch.run_action_coding_agent_runtime import (
    PROVIDER_SANDBOX_EXECUTABLE,
    ProviderSandboxDescriptorRule,
    ProviderSandboxDescriptors,
    RunActionCodingAgentRuntimeError,
    apply_provider_landlock,
    apply_provider_process_group_containment,
    coding_agent_provider_sandbox_command,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)

_READ_ONLY_LANDLOCK_ACCESS = (1 << 0) | (1 << 2) | (1 << 3)
_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


@pytest.fixture
def sandbox_descriptors(tmp_path):
    with ExitStack() as resources:
        descriptors = []
        for name in ("workspace", "home", "output", "support"):
            path = tmp_path / name
            path.mkdir()
            descriptor = os.open(
                path,
                os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            resources.callback(os.close, descriptor)
            descriptors.append(descriptor)
        yield ProviderSandboxDescriptors(*descriptors)


def test_provider_environment_is_complete_and_contains_no_ambient_authority():
    assert dict(coding_agent_provider_environment()) == {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/kapso/tmp/provider-home",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TERM": "dumb",
        "TMPDIR": "/kapso/tmp/provider-home",
    }


def test_sandbox_command_binds_complete_policy_and_provider_argv(
    sandbox_descriptors,
):
    request = run_action_request(interpretation_policy())
    command = ("/usr/bin/codex", "--version")

    projected = coding_agent_provider_sandbox_command(
        request,
        command,
        sandbox_descriptors,
    )

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
        "--workspace-descriptor",
        str(sandbox_descriptors.workspace_descriptor),
        "--home-descriptor",
        str(sandbox_descriptors.home_descriptor),
        "--output-descriptor",
        str(sandbox_descriptors.output_descriptor),
        "--support-descriptor",
        str(sandbox_descriptors.support_descriptor),
        "--",
        *command,
    )


def test_editing_preflight_receives_read_only_workspace_authority(
    sandbox_descriptors,
):
    request = run_action_request(
        interpretation_policy(
            workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        ),
        predecessor_digest=tree_or_blob_digest(b"predecessor"),
    )

    projected = coding_agent_provider_sandbox_command(
        request,
        coding_agent_cli_preflight_command(request),
        sandbox_descriptors,
    )

    assert projected[projected.index("--workspace-access") + 1] == "read_only"


def test_sandbox_command_enforces_the_complete_argv_bound(sandbox_descriptors):
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
            sandbox_descriptors,
        )


def test_sandbox_command_rejects_an_absolute_but_unbound_provider_argv(
    sandbox_descriptors,
):
    request = run_action_request(interpretation_policy())

    with pytest.raises(
        RunActionCodingAgentRuntimeError,
        match="request-derived projection",
    ):
        coding_agent_provider_sandbox_command(
            request,
            ("/usr/bin/codex", "exec", "substituted"),
            sandbox_descriptors,
        )


def test_sandbox_descriptors_reject_distinct_numbers_for_one_inode(tmp_path):
    path = tmp_path / "aliased"
    path.mkdir()
    with ExitStack() as resources:
        descriptors = []
        for _index in range(4):
            descriptor = os.open(
                path,
                os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            resources.callback(os.close, descriptor)
            descriptors.append(descriptor)
        aliased = ProviderSandboxDescriptors(*descriptors)

        with pytest.raises(
            RunActionCodingAgentRuntimeError,
            match="four distinct retained directories",
        ):
            coding_agent_provider_sandbox_command(
                run_action_request(interpretation_policy()),
                ("/usr/bin/codex", "--version"),
                aliased,
            )


def test_launcher_rejects_a_fifth_inherited_descriptor(
    tmp_path,
    sandbox_descriptors,
):
    extra_path = tmp_path / "ambient-secret"
    extra_path.write_text("not admitted", encoding="utf-8")
    with ExitStack() as resources:
        extra_descriptor = os.open(extra_path, os.O_RDONLY | os.O_CLOEXEC)
        resources.callback(os.close, extra_descriptor)
        command = (
            sys.executable,
            "-m",
            "kapso.cross_run.launch.run_action_coding_agent_runtime",
            "--landlock-abi-version",
            "7",
            "--supervisor-user-id",
            str(os.geteuid()),
            "--supervisor-group-id",
            str(os.getegid()),
            "--provider-user-id",
            str(os.geteuid() + 1),
            "--provider-group-id",
            str(os.getegid() + 1),
            "--workspace-access",
            "read_only",
            "--workspace-descriptor",
            str(sandbox_descriptors.workspace_descriptor),
            "--home-descriptor",
            str(sandbox_descriptors.home_descriptor),
            "--output-descriptor",
            str(sandbox_descriptors.output_descriptor),
            "--support-descriptor",
            str(sandbox_descriptors.support_descriptor),
            "--",
            "/usr/bin/codex",
            "--version",
        )

        completion = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(*sandbox_descriptors.all, extra_descriptor),
            check=False,
        )

    assert completion.returncode != 0
    assert completion.stdout == b""
    assert b"inherited an unadmitted descriptor" in completion.stderr


def test_launcher_accepts_exactly_the_four_admitted_descriptors(
    sandbox_descriptors,
):
    program = (
        "from kapso.cross_run.launch.run_action_coding_agent_runtime import "
        "ProviderSandboxDescriptors,_require_exact_inherited_descriptor_table;"
        "import sys;"
        "descriptors=ProviderSandboxDescriptors(*(int(value) for value in sys.argv[1:]));"
        "_require_exact_inherited_descriptor_table(descriptors);"
        "print('exact')"
    )

    completion = subprocess.run(
        (
            sys.executable,
            "-c",
            program,
            *(str(descriptor) for descriptor in sandbox_descriptors.all),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=sandbox_descriptors.all,
        check=False,
    )

    assert completion.returncode == 0
    assert completion.stdout == b"exact\n"
    assert completion.stderr == b""


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

    with ExitStack() as resources:
        descriptor = os.open(allowed, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        resources.callback(os.close, descriptor)
        with pytest.raises(
            RunActionCodingAgentRuntimeError,
            match="ABI differs",
        ):
            apply_provider_landlock(
                expected_abi_version=observed + 1,
                descriptor_rules=(
                    ProviderSandboxDescriptorRule(
                        descriptor,
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
        "ProviderSandboxDescriptorRule,apply_provider_landlock;"
        "import pathlib,sys;"
        "apply_provider_landlock(expected_abi_version=int(sys.argv[1]),"
        "descriptor_rules=(ProviderSandboxDescriptorRule(int(sys.argv[2]),13),));"
        "print(pathlib.Path(sys.argv[3]).read_text(encoding='utf-8'))"
    )
    with ExitStack() as resources:
        descriptor = os.open(allowed, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        resources.callback(os.close, descriptor)
        relocated = tmp_path / "relocated"
        allowed.rename(relocated)
        base_command = (
            sys.executable,
            "-c",
            program,
            str(abi),
            str(descriptor),
        )
        allowed_completion = subprocess.run(
            (*base_command, str((relocated / "record").resolve())),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(descriptor,),
            check=False,
        )
        denied_completion = subprocess.run(
            (*base_command, str((denied / "record").resolve())),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(descriptor,),
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
        "ProviderSandboxDescriptorRule,apply_provider_landlock;"
        "import os,sys;"
        "apply_provider_landlock(expected_abi_version=int(sys.argv[1]),"
        "descriptor_rules=(ProviderSandboxDescriptorRule(int(sys.argv[2]),13),));"
        "os.kill(os.getppid(),0)"
    )
    with ExitStack() as resources:
        descriptor = os.open(allowed, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        resources.callback(os.close, descriptor)
        completion = subprocess.run(
            (
                sys.executable,
                "-c",
                program,
                str(abi),
                str(descriptor),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(descriptor,),
            check=False,
        )

    assert completion.returncode != 0
    assert completion.stdout == b""
    assert b"PermissionError" in completion.stderr


@pytest.mark.parametrize("operation", ("os.setpgid(0,0)", "os.setsid()"))
def test_provider_descendants_cannot_escape_the_supervisor_process_group(operation):
    program = (
        "from kapso.cross_run.launch.run_action_coding_agent_runtime import "
        "apply_provider_process_group_containment;"
        "import os;"
        "apply_provider_process_group_containment();" + operation
    )

    completion = subprocess.run(
        (sys.executable, "-c", program),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completion.returncode != 0
    assert completion.stdout == b""
    assert b"PermissionError" in completion.stderr
