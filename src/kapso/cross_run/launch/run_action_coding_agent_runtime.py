"""Irreversible Linux sandbox projection for one native coding-agent provider."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import signal
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.coding_agent_compatibility import (
    CODING_AGENT_LANDLOCK_POLICY_ABI_VERSION,
)
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    coding_agent_cli_command,
    coding_agent_cli_preflight_command,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)

PROVIDER_SANDBOX_EXECUTABLE = (
    "/usr/local/bin/kapso-run-action-coding-agent-provider-sandbox"
)
PROVIDER_SETPRIV_EXECUTABLE = "/usr/bin/setpriv"
PROVIDER_WORKSPACE_PATH = "/kapso/tmp/provider-workspace"
PROVIDER_HOME_PATH = "/kapso/tmp/provider-home"
PROVIDER_OUTPUT_PATH = "/kapso/tmp/provider-output"
PROVIDER_SUPPORT_PATH = "/kapso/tmp/provider-support"

_PROVIDER_ENVIRONMENT = MappingProxyType(
    {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": PROVIDER_HOME_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TERM": "dumb",
        "TMPDIR": PROVIDER_HOME_PATH,
    }
)

_LANDLOCK_CREATE_RULESET_SYSCALL = 444
_LANDLOCK_ADD_RULE_SYSCALL = 445
_LANDLOCK_RESTRICT_SELF_SYSCALL = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_SCOPE_SIGNAL = 1 << 1
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_PDEATHSIG = 2

_ACCESS_EXECUTE = 1 << 0
_ACCESS_WRITE_FILE = 1 << 1
_ACCESS_READ_FILE = 1 << 2
_ACCESS_READ_DIR = 1 << 3
_ACCESS_REMOVE_DIR = 1 << 4
_ACCESS_REMOVE_FILE = 1 << 5
_ACCESS_MAKE_CHAR = 1 << 6
_ACCESS_MAKE_DIR = 1 << 7
_ACCESS_MAKE_REG = 1 << 8
_ACCESS_MAKE_SOCK = 1 << 9
_ACCESS_MAKE_FIFO = 1 << 10
_ACCESS_MAKE_BLOCK = 1 << 11
_ACCESS_MAKE_SYM = 1 << 12
_ACCESS_REFER = 1 << 13
_ACCESS_TRUNCATE = 1 << 14
_ACCESS_IOCTL_DEV = 1 << 15

_HANDLED_FILESYSTEM_ACCESS = (1 << 16) - 1
_READ_ONLY_ACCESS = _ACCESS_EXECUTE | _ACCESS_READ_FILE | _ACCESS_READ_DIR
_READ_WRITE_REGULAR_ACCESS = (
    _READ_ONLY_ACCESS
    | _ACCESS_WRITE_FILE
    | _ACCESS_TRUNCATE
    | _ACCESS_REMOVE_DIR
    | _ACCESS_REMOVE_FILE
    | _ACCESS_MAKE_DIR
    | _ACCESS_MAKE_REG
    | _ACCESS_MAKE_SOCK
    | _ACCESS_MAKE_FIFO
    | _ACCESS_MAKE_SYM
    | _ACCESS_REFER
)
_DEVICE_ACCESS = _ACCESS_READ_FILE | _ACCESS_WRITE_FILE | _ACCESS_IOCTL_DEV
_CAPABILITY_STATUS_PATTERN = re.compile(
    rb"^(CapInh|CapPrm|CapEff|CapBnd|CapAmb):[\t ]+([0-9a-f]{16})$"
)
_NO_NEW_PRIVILEGES_STATUS_PATTERN = re.compile(rb"^NoNewPrivs:[\t ]+([01])$")
_IDENTITY_STATUS_PATTERN = re.compile(
    rb"^(Uid|Gid):[\t ]+([0-9]+)[\t ]+([0-9]+)[\t ]+([0-9]+)[\t ]+([0-9]+)$"
)
_LINUX_IDENTITY_MAXIMUM = 2_147_483_647


class RunActionCodingAgentRuntimeError(RuntimeError):
    """The provider launcher cannot prove its exact irreversible sandbox."""


class _LandlockRulesetAttribute(ctypes.Structure):
    _fields_ = (
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    )


class _LandlockPathBeneathAttribute(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    )


@dataclass(frozen=True)
class ProviderSandboxPathRule:
    """One descriptor-resolved hierarchy and its maximum Landlock access."""

    path: str
    allowed_access: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise RunActionCodingAgentRuntimeError(
                "provider sandbox path rule is invalid"
            )
        parsed = PurePosixPath(self.path)
        if (
            not parsed.is_absolute()
            or parsed.as_posix() != self.path
            or ".." in parsed.parts
            or type(self.allowed_access) is not int
            or self.allowed_access <= 0
            or self.allowed_access & ~_HANDLED_FILESYSTEM_ACCESS
        ):
            raise RunActionCodingAgentRuntimeError(
                "provider sandbox path rule is invalid"
            )


def coding_agent_provider_sandbox_environment() -> Mapping[str, str]:
    """Return the provider's complete post-boundary process environment."""

    return _PROVIDER_ENVIRONMENT


def coding_agent_provider_sandbox_command(
    request: CodingAgentRunActionRequest,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    """Wrap one exact provider argv in the fixed Landlock and identity boundary."""

    if (
        type(request) is not CodingAgentRunActionRequest
        or type(command) is not tuple
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
        or not PurePosixPath(command[0]).is_absolute()
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider sandbox command requires one exact absolute argv"
        )
    if command != coding_agent_cli_preflight_command(
        request
    ) and command != coding_agent_cli_command(request):
        raise RunActionCodingAgentRuntimeError(
            "provider sandbox command differs from its request-derived projection"
        )
    policy = request.interpretation_policy
    projected = (
        PROVIDER_SANDBOX_EXECUTABLE,
        "--landlock-abi-version",
        str(policy.landlock_abi_version),
        "--supervisor-user-id",
        str(policy.supervisor_user_id),
        "--supervisor-group-id",
        str(policy.supervisor_group_id),
        "--provider-user-id",
        str(policy.provider_user_id),
        "--provider-group-id",
        str(policy.provider_group_id),
        "--workspace-access",
        policy.workspace_access.value,
        "--",
        *command,
    )
    encoded_size = sum(len(argument.encode("utf-8")) + 1 for argument in projected)
    if encoded_size > policy.maximum_cli_argument_bytes:
        raise RunActionCodingAgentRuntimeError(
            "provider sandbox argv exceeds its exact byte limit"
        )
    return projected


def apply_provider_landlock(
    *,
    expected_abi_version: int,
    path_rules: tuple[ProviderSandboxPathRule, ...],
) -> None:
    """Install one fail-closed filesystem and signal domain on this thread."""

    if (
        type(expected_abi_version) is not int
        or expected_abi_version != CODING_AGENT_LANDLOCK_POLICY_ABI_VERSION
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider Landlock ABI differs from its pinned policy"
        )
    if (
        type(path_rules) is not tuple
        or not path_rules
        or any(type(rule) is not ProviderSandboxPathRule for rule in path_rules)
        or tuple(rule.path for rule in path_rules)
        != tuple(sorted({rule.path for rule in path_rules}))
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider Landlock policy is invalid or noncanonical"
        )
    system_call = ctypes.CDLL(None, use_errno=True).syscall
    system_call.restype = ctypes.c_long
    observed_abi_version = system_call(
        _LANDLOCK_CREATE_RULESET_SYSCALL,
        ctypes.c_void_p(),
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    if observed_abi_version != expected_abi_version:
        raise RunActionCodingAgentRuntimeError(
            "provider Landlock ABI differs from its pinned policy"
        )
    ruleset_attribute = _LandlockRulesetAttribute(
        handled_access_fs=_HANDLED_FILESYSTEM_ACCESS,
        handled_access_net=0,
        scoped=_LANDLOCK_SCOPE_SIGNAL,
    )
    ruleset_descriptor = system_call(
        _LANDLOCK_CREATE_RULESET_SYSCALL,
        ctypes.byref(ruleset_attribute),
        ctypes.sizeof(ruleset_attribute),
        0,
    )
    if ruleset_descriptor < 0:
        _raise_system_call_error("create provider Landlock ruleset")
    with ExitStack() as descriptors:
        descriptors.callback(os.close, ruleset_descriptor)
        for rule in path_rules:
            path_descriptor = os.open(
                rule.path,
                os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, path_descriptor)
            metadata = os.fstat(path_descriptor)
            allowed_access = _access_for_file_type(
                rule.allowed_access,
                metadata.st_mode,
            )
            if allowed_access == 0:
                raise RunActionCodingAgentRuntimeError(
                    "provider Landlock rule grants no applicable access"
                )
            path_attribute = _LandlockPathBeneathAttribute(
                allowed_access=allowed_access,
                parent_fd=path_descriptor,
            )
            if (
                system_call(
                    _LANDLOCK_ADD_RULE_SYSCALL,
                    ruleset_descriptor,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(path_attribute),
                    0,
                )
                != 0
            ):
                _raise_system_call_error("add provider Landlock path rule")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _raise_system_call_error("set provider no-new-privileges")
        if (
            system_call(
                _LANDLOCK_RESTRICT_SELF_SYSCALL,
                ruleset_descriptor,
                0,
            )
            != 0
        ):
            _raise_system_call_error("enforce provider Landlock ruleset")


def main() -> None:
    """Apply the fixed domain, erase privilege, verify it, and exec the provider."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-and-exec", action="store_true")
    parser.add_argument("--landlock-abi-version", type=int, required=True)
    parser.add_argument("--supervisor-user-id", type=int, required=True)
    parser.add_argument("--supervisor-group-id", type=int, required=True)
    parser.add_argument("--provider-user-id", type=int, required=True)
    parser.add_argument("--provider-group-id", type=int, required=True)
    parser.add_argument(
        "--workspace-access",
        choices=(
            RunFrontierWorkspaceAccess.READ_ONLY.value,
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE.value,
        ),
        required=True,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = tuple(arguments.command)
    if command[:1] == ("--",):
        command = command[1:]
    _require_launcher_arguments(arguments, command)
    if arguments.verify_and_exec:
        _require_provider_identity_and_privilege(arguments)
        os.execve(
            command[0],
            command,
            dict(coding_agent_provider_sandbox_environment()),
        )
    if (
        os.geteuid() != arguments.supervisor_user_id
        or os.getegid() != arguments.supervisor_group_id
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider launcher lacks its supervisor identity"
        )
    path_rules = _fixed_provider_path_rules(arguments.workspace_access)
    apply_provider_landlock(
        expected_abi_version=arguments.landlock_abi_version,
        path_rules=path_rules,
    )
    verification_command = (
        PROVIDER_SANDBOX_EXECUTABLE,
        "--verify-and-exec",
        "--landlock-abi-version",
        str(arguments.landlock_abi_version),
        "--supervisor-user-id",
        str(arguments.supervisor_user_id),
        "--supervisor-group-id",
        str(arguments.supervisor_group_id),
        "--provider-user-id",
        str(arguments.provider_user_id),
        "--provider-group-id",
        str(arguments.provider_group_id),
        "--workspace-access",
        arguments.workspace_access,
        "--",
        *command,
    )
    privilege_drop_command = (
        PROVIDER_SETPRIV_EXECUTABLE,
        f"--reuid={arguments.provider_user_id}",
        f"--regid={arguments.provider_group_id}",
        "--clear-groups",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--no-new-privs",
        "--pdeathsig=SIGKILL",
        *verification_command,
    )
    os.execve(
        PROVIDER_SETPRIV_EXECUTABLE,
        privilege_drop_command,
        dict(coding_agent_provider_sandbox_environment()),
    )


def _fixed_provider_path_rules(
    workspace_access: str,
) -> tuple[ProviderSandboxPathRule, ...]:
    workspace_rule = ProviderSandboxPathRule(
        PROVIDER_WORKSPACE_PATH,
        (
            _READ_WRITE_REGULAR_ACCESS
            if workspace_access == RunFrontierWorkspaceAccess.EDIT_WORKSPACE.value
            else _READ_ONLY_ACCESS
        ),
    )
    rules = (
        ProviderSandboxPathRule("/dev/null", _DEVICE_ACCESS),
        ProviderSandboxPathRule("/dev/urandom", _DEVICE_ACCESS),
        ProviderSandboxPathRule("/etc/group", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/hosts", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/ld.so.cache", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/nsswitch.conf", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/passwd", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/resolv.conf", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/etc/ssl", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule(PROVIDER_HOME_PATH, _READ_WRITE_REGULAR_ACCESS),
        ProviderSandboxPathRule(PROVIDER_OUTPUT_PATH, _READ_WRITE_REGULAR_ACCESS),
        ProviderSandboxPathRule(PROVIDER_SUPPORT_PATH, _READ_ONLY_ACCESS),
        workspace_rule,
        ProviderSandboxPathRule("/proc/self/status", _READ_ONLY_ACCESS),
        ProviderSandboxPathRule("/usr", _READ_ONLY_ACCESS),
    )
    return tuple(sorted(rules, key=lambda rule: rule.path))


def _require_launcher_arguments(
    arguments: argparse.Namespace,
    command: tuple[str, ...],
) -> None:
    identity_values = (
        arguments.supervisor_user_id,
        arguments.supervisor_group_id,
        arguments.provider_user_id,
        arguments.provider_group_id,
    )
    if (
        arguments.landlock_abi_version != CODING_AGENT_LANDLOCK_POLICY_ABI_VERSION
        or any(not 0 < value <= _LINUX_IDENTITY_MAXIMUM for value in identity_values)
        or arguments.supervisor_user_id == arguments.provider_user_id
        or arguments.supervisor_group_id == arguments.provider_group_id
        or not command
        or any(not isinstance(value, str) or not value for value in command)
        or not PurePosixPath(command[0]).is_absolute()
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider launcher arguments are invalid"
        )


def _require_provider_identity_and_privilege(arguments: argparse.Namespace) -> None:
    if (
        os.geteuid() != arguments.provider_user_id
        or os.getegid() != arguments.provider_group_id
        or os.getgroups()
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider identity was not dropped exactly"
        )
    descriptor = os.open(
        "/proc/self/status",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    capability_fields = {}
    identity_fields = {}
    no_new_privileges = None
    for line in payload.splitlines():
        capability_match = _CAPABILITY_STATUS_PATTERN.fullmatch(line)
        if capability_match is not None:
            capability_fields[capability_match.group(1)] = capability_match.group(2)
        identity_match = _IDENTITY_STATUS_PATTERN.fullmatch(line)
        if identity_match is not None:
            identity_fields[identity_match.group(1)] = tuple(
                int(identity_match.group(index)) for index in range(2, 6)
            )
        no_new_privileges_match = _NO_NEW_PRIVILEGES_STATUS_PATTERN.fullmatch(line)
        if no_new_privileges_match is not None:
            no_new_privileges = no_new_privileges_match.group(1)
    if (
        identity_fields
        != {
            b"Uid": (arguments.provider_user_id,) * 4,
            b"Gid": (arguments.provider_group_id,) * 4,
        }
        or set(capability_fields)
        != {b"CapInh", b"CapPrm", b"CapEff", b"CapBnd", b"CapAmb"}
        or any(value != b"0000000000000000" for value in capability_fields.values())
        or no_new_privileges != b"1"
    ):
        raise RunActionCodingAgentRuntimeError(
            "provider privilege was not erased exactly"
        )
    parent_death_signal = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    if (
        libc.prctl(
            _PR_GET_PDEATHSIG,
            ctypes.byref(parent_death_signal),
            0,
            0,
            0,
        )
        != 0
    ):
        _raise_system_call_error("read provider parent-death signal")
    if parent_death_signal.value != signal.SIGKILL:
        raise RunActionCodingAgentRuntimeError(
            "provider parent-death signal is not irreversible containment"
        )


def _access_for_file_type(allowed_access: int, mode: int) -> int:
    if stat.S_ISDIR(mode):
        return allowed_access
    if stat.S_ISREG(mode):
        return allowed_access & (
            _ACCESS_EXECUTE | _ACCESS_WRITE_FILE | _ACCESS_READ_FILE | _ACCESS_TRUNCATE
        )
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return allowed_access & _DEVICE_ACCESS
    raise RunActionCodingAgentRuntimeError(
        "provider Landlock path is not a directory, regular file, or device"
    )


def _raise_system_call_error(action: str) -> None:
    error_number = ctypes.get_errno()
    raise OSError(error_number, action)


__all__ = [
    "apply_provider_landlock",
    "coding_agent_provider_sandbox_environment",
    "coding_agent_provider_sandbox_command",
    "main",
    "PROVIDER_HOME_PATH",
    "PROVIDER_OUTPUT_PATH",
    "PROVIDER_SANDBOX_EXECUTABLE",
    "PROVIDER_SUPPORT_PATH",
    "PROVIDER_WORKSPACE_PATH",
    "ProviderSandboxPathRule",
    "RunActionCodingAgentRuntimeError",
]


if __name__ == "__main__":
    main()
