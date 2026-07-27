"""Fixed-path provider-native consumer for one coding-agent run action."""

from __future__ import annotations

import fcntl
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import Thread
from typing import Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.knowledge.access import PriorKnowledgeAccess
from kapso.cross_run.launch.run_action_atomic_publication import (
    link_run_action_anonymous_file_no_replace,
    open_run_action_anonymous_file,
    require_run_action_descriptor_payload,
    write_run_action_full_payload,
)
from kapso.cross_run.launch.run_action_coding_agent_candidate import (
    CodingAgentPublishedCandidate,
    publish_coding_agent_result_candidate,
)
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    coding_agent_cli_command,
    coding_agent_cli_final_output_path,
    coding_agent_cli_preflight_command,
    coding_agent_cli_prior_knowledge_audit_path,
    coding_agent_cli_provider_environment,
    coding_agent_cli_support_payloads,
    coding_agent_cli_temporary_path,
    coding_agent_cli_workspace_path,
    interpret_coding_agent_cli_completion,
    validate_coding_agent_cli_preflight,
    validate_coding_agent_cli_prior_knowledge_trace,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_RESULT_PROTOCOL_VERSION,
    CodingAgentPriorKnowledgeAccessEvent,
    CodingAgentPriorKnowledgeAccessKind,
    CodingAgentRunActionRequest,
    CodingAgentRunActionResultEnvelope,
    read_canonical_coding_agent_request,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceFrontierIdentity,
    RunWorkspaceRegularTreeIdentity,
    RunWorkspaceSourceTreeIdentity,
    inspect_run_workspace_frontier_with_limits,
    inspect_run_workspace_regular_tree,
    inspect_run_workspace_source_regular_tree,
    inspect_run_workspace_source_tree,
)

NATIVE_CODING_AGENT_CONSUMER_ID = "kapso.native_coding_agent_consumer"
NATIVE_CODING_AGENT_CONSUMER_VERSION = "kapso.native_coding_agent_consumer.v1"

_REQUEST_PATH = "/kapso/input/request.blob"
_REQUEST_SIZE_LIMIT_PATH = "/kapso/input/request.maximum_bytes"
_GIT_EXECUTABLE = "/usr/bin/git"
_KILL_EXECUTABLE = "/bin/kill"
_GIT_COMMIT_TIME = "@0 +0000"
_TRUSTED_GIT_CONFIGURATION_ARGUMENTS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
)
_SUPPORT_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_MAXIMUM_UNSIGNED_64 = (1 << 64) - 1


class RunActionCodingAgentConsumerError(RuntimeError):
    """The fixed runtime projection or provider completion is invalid."""


@dataclass(frozen=True)
class BoundedCodingAgentProcessCompletion:
    """One reaped process and its complete bounded output streams."""

    return_code: int
    output_payload: bytes
    diagnostic_payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.return_code) is not int
            or type(self.output_payload) is not bytes
            or type(self.diagnostic_payload) is not bytes
        ):
            raise RunActionCodingAgentConsumerError(
                "coding-agent process completion is invalid"
            )


@dataclass(frozen=True)
class _CodingAgentSupportFileObservation:
    name: str
    descriptor: int
    payload: bytes
    metadata: tuple[int, ...]


class CodingAgentProcessRunner(Protocol):
    """The bounded process seam used by the fixed consumer and process tests."""

    def run(
        self,
        command: tuple[str, ...],
        *,
        stdin_payload: bytes | None,
        stdin_directory: str,
        working_directory: str,
        timeout_nanoseconds: int,
        termination_grace_nanoseconds: int,
        maximum_output_bytes: int,
        maximum_diagnostic_bytes: int,
        environment: Mapping[str, str] | None,
    ) -> BoundedCodingAgentProcessCompletion: ...


@dataclass(frozen=True)
class BoundedCodingAgentProcessRunner:
    """Run one process without buffering either provider stream past its bound."""

    def run(
        self,
        command: tuple[str, ...],
        *,
        stdin_payload: bytes | None,
        stdin_directory: str,
        working_directory: str,
        timeout_nanoseconds: int,
        termination_grace_nanoseconds: int,
        maximum_output_bytes: int,
        maximum_diagnostic_bytes: int,
        environment: Mapping[str, str] | None,
    ) -> BoundedCodingAgentProcessCompletion:
        _require_process_inputs(
            command,
            stdin_payload,
            stdin_directory,
            working_directory,
            timeout_nanoseconds,
            termination_grace_nanoseconds,
            maximum_output_bytes,
            maximum_diagnostic_bytes,
            environment,
        )
        deadline = time.monotonic_ns() + timeout_nanoseconds
        with ExitStack() as resources:
            stdin_descriptor = subprocess.DEVNULL
            if stdin_payload is not None:
                stdin_directory_descriptor = os.open(
                    stdin_directory,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                )
                resources.callback(os.close, stdin_directory_descriptor)
                stdin_descriptor = open_run_action_anonymous_file(
                    stdin_directory_descriptor,
                    _SUPPORT_FILE_MODE,
                )
                resources.callback(os.close, stdin_descriptor)
                write_run_action_full_payload(stdin_descriptor, stdin_payload)
                require_run_action_descriptor_payload(
                    stdin_descriptor,
                    stdin_payload,
                )
                os.lseek(stdin_descriptor, 0, os.SEEK_SET)
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                stdin=stdin_descriptor,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
            resources.callback(_terminate_process_group, process)
            if process.stdout is None or process.stderr is None:
                raise RunActionCodingAgentConsumerError(
                    "coding-agent process lacks provider pipes"
                )
            resources.callback(process.stderr.close)
            resources.callback(process.stdout.close)
            completion_read_descriptor, completion_write_descriptor = os.pipe2(
                os.O_CLOEXEC | os.O_NONBLOCK
            )
            resources.callback(os.close, completion_read_descriptor)
            completion_keeper_descriptor = os.dup(completion_read_descriptor)
            waiter = Thread(
                target=_wait_and_signal_process,
                args=(
                    process,
                    completion_write_descriptor,
                    completion_keeper_descriptor,
                ),
                daemon=True,
            )
            waiter.start()
            resources.callback(
                _terminate_and_join_process,
                process,
                waiter,
                termination_grace_nanoseconds,
            )
            selector = resources.enter_context(selectors.DefaultSelector())
            output_descriptor = process.stdout.fileno()
            diagnostic_descriptor = process.stderr.fileno()
            selector.register(
                completion_read_descriptor,
                selectors.EVENT_READ,
                ("completion", None),
            )
            selector.register(
                output_descriptor,
                selectors.EVENT_READ,
                ("output", maximum_output_bytes),
            )
            selector.register(
                diagnostic_descriptor,
                selectors.EVENT_READ,
                ("diagnostic", maximum_diagnostic_bytes),
            )
            buffers = {
                "output": bytearray(),
                "diagnostic": bytearray(),
            }
            violation = None
            termination_deadline = None
            process_completed = False
            while selector.get_map():
                active_deadline = (
                    deadline if termination_deadline is None else termination_deadline
                )
                remaining_nanoseconds = active_deadline - time.monotonic_ns()
                if remaining_nanoseconds <= 0:
                    if violation is not None:
                        raise RunActionCodingAgentConsumerError(violation)
                    if process_completed:
                        _kill_process_group(process)
                        raise RunActionCodingAgentConsumerError(
                            "coding-agent descendants retained provider streams"
                        )
                    if process.returncode is None:
                        _terminate_process_group(process)
                    violation = "coding-agent process exceeded its exact timeout"
                    termination_deadline = (
                        time.monotonic_ns() + termination_grace_nanoseconds
                    )
                    continue
                events = selector.select(remaining_nanoseconds / 1_000_000_000)
                if not events:
                    continue
                for key, _mask in events:
                    stream_name, maximum_bytes = key.data
                    if stream_name == "completion":
                        if os.read(key.fd, 1) != b"\x01":
                            raise RunActionCodingAgentConsumerError(
                                "coding-agent process completion signal is invalid"
                            )
                        selector.unregister(key.fd)
                        _kill_process_group(process)
                        process.wait()
                        process_completed = True
                        if type(process.returncode) is not int:
                            raise RunActionCodingAgentConsumerError(
                                "coding-agent process was not exactly reaped"
                            )
                        if selector.get_map():
                            termination_deadline = min(
                                deadline,
                                time.monotonic_ns() + termination_grace_nanoseconds,
                            )
                        continue
                    buffer = buffers[stream_name]
                    remaining_capacity = maximum_bytes - len(buffer)
                    chunk = os.read(key.fd, remaining_capacity + 1)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    if len(chunk) > remaining_capacity:
                        if remaining_capacity > 0:
                            buffer.extend(chunk[:remaining_capacity])
                        if process.returncode is None:
                            _terminate_process_group(process)
                        violation = (
                            f"coding-agent provider {stream_name} exceeded its "
                            "exact byte limit"
                        )
                        termination_deadline = (
                            time.monotonic_ns() + termination_grace_nanoseconds
                        )
                    else:
                        buffer.extend(chunk)
            if violation is not None:
                raise RunActionCodingAgentConsumerError(violation)
            if not process_completed or type(process.returncode) is not int:
                raise RunActionCodingAgentConsumerError(
                    "coding-agent process lacks exact completion"
                )
            return BoundedCodingAgentProcessCompletion(
                return_code=process.returncode,
                output_payload=bytes(buffers["output"]),
                diagnostic_payload=bytes(buffers["diagnostic"]),
            )


def consume_coding_agent_run_action(
    *,
    request_payload: bytes,
    workspace_descriptor: int,
    temporary_directory_descriptor: int,
    process_runner: CodingAgentProcessRunner,
) -> CodingAgentPublishedCandidate:
    """Consume one canonical request and publish its sole terminal candidate."""

    request = read_canonical_coding_agent_request(request_payload)
    policy = request.interpretation_policy
    if (
        len(request_payload) > policy.maximum_request_bytes
        or policy.consumer_id != NATIVE_CODING_AGENT_CONSUMER_ID
        or policy.consumer_version != NATIVE_CODING_AGENT_CONSUMER_VERSION
    ):
        raise RunActionCodingAgentConsumerError(
            "coding-agent request differs from the fixed native consumer"
        )
    baseline = _inspect_clean_workspace(workspace_descriptor, request)
    editing = policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
    read_only_physical_baseline = (
        None
        if editing
        else _inspect_source_physical_workspace(workspace_descriptor, request)
    )
    if editing and baseline.source_tree_digest != (
        request.edit_predecessor_source_tree_digest
    ):
        raise RunActionCodingAgentConsumerError(
            "editable coding-agent workspace differs from its predecessor"
        )
    git_metadata_before = _inspect_git_metadata(workspace_descriptor, request)
    _require_empty_temporary_directory(temporary_directory_descriptor)
    preflight = process_runner.run(
        coding_agent_cli_preflight_command(request),
        stdin_payload=None,
        stdin_directory=coding_agent_cli_temporary_path(),
        working_directory=coding_agent_cli_workspace_path(),
        timeout_nanoseconds=policy.timeout_nanoseconds,
        termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
        maximum_output_bytes=policy.maximum_provider_output_bytes,
        maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
        environment=coding_agent_cli_provider_environment(),
    )
    validate_coding_agent_cli_preflight(
        request=request,
        return_code=preflight.return_code,
        output_payload=preflight.output_payload,
        diagnostic_payload=preflight.diagnostic_payload,
    )
    with ExitStack() as support_file_resources:
        support_files = _materialize_cli_support(
            temporary_directory_descriptor,
            request,
            support_file_resources,
        )
        _require_cli_support_unchanged(
            temporary_directory_descriptor,
            support_files,
        )
        started_nanoseconds = time.monotonic_ns()
        process = process_runner.run(
            coding_agent_cli_command(request),
            stdin_payload=request.prompt.encode("utf-8"),
            stdin_directory=coding_agent_cli_temporary_path(),
            working_directory=coding_agent_cli_workspace_path(),
            timeout_nanoseconds=policy.timeout_nanoseconds,
            termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
            maximum_output_bytes=policy.maximum_provider_output_bytes,
            maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
            environment=coding_agent_cli_provider_environment(),
        )
        duration_nanoseconds = time.monotonic_ns() - started_nanoseconds
        final_output_payload = _read_provider_final(
            temporary_directory_descriptor,
            request,
        )
        outcome = interpret_coding_agent_cli_completion(
            request=request,
            return_code=process.return_code,
            provider_output_payload=process.output_payload,
            provider_diagnostic_payload=process.diagnostic_payload,
            final_output_payload=final_output_payload,
        )
        prior_knowledge_accesses = _read_prior_knowledge_accesses(
            temporary_directory_descriptor,
            request,
        )
        validate_coding_agent_cli_prior_knowledge_trace(
            request=request,
            outcome=outcome,
            accesses=prior_knowledge_accesses,
        )
        _require_cli_support_unchanged(
            temporary_directory_descriptor,
            support_files,
        )
        if editing:
            uncommitted_successor = _inspect_workspace(
                workspace_descriptor,
                request,
            )
            if uncommitted_successor.source_tree_digest == (
                baseline.source_tree_digest
            ):
                raise RunActionCodingAgentConsumerError(
                    "coding-agent edit did not change the source tree"
                )
            if (
                _inspect_git_metadata(workspace_descriptor, request)
                != git_metadata_before
            ):
                raise RunActionCodingAgentConsumerError(
                    "coding-agent changed trusted Git metadata"
                )
            _commit_coding_agent_workspace(
                request=request,
                predecessor=baseline,
                process_runner=process_runner,
            )
            successor = _inspect_clean_workspace(
                workspace_descriptor,
                request,
            )
            if (
                successor.parent_commit_shas != (baseline.commit_sha,)
                or successor.source_tree_digest
                != uncommitted_successor.source_tree_digest
            ):
                raise RunActionCodingAgentConsumerError(
                    "trusted coding-agent commit is not the exact direct successor"
                )
        else:
            if (
                read_only_physical_baseline is None
                or _inspect_source_physical_workspace(workspace_descriptor, request)
                != read_only_physical_baseline
            ):
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent physical workspace changed during "
                    "execution"
                )
            if (
                _inspect_git_metadata(workspace_descriptor, request)
                != git_metadata_before
            ):
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent physical Git metadata changed during "
                    "execution"
                )
            observed_source = _inspect_workspace(
                workspace_descriptor,
                request,
            )
            if observed_source.source_tree_digest != baseline.source_tree_digest:
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent workspace changed during execution"
                )
            successor = _inspect_clean_workspace(
                workspace_descriptor,
                request,
            )
            if successor != baseline:
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent workspace changed during execution"
                )
            if (
                _inspect_source_physical_workspace(workspace_descriptor, request)
                != read_only_physical_baseline
            ):
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent physical workspace changed during "
                    "reconciliation"
                )
            if (
                _inspect_git_metadata(workspace_descriptor, request)
                != git_metadata_before
            ):
                raise RunActionCodingAgentConsumerError(
                    "read-only coding-agent physical Git metadata changed during "
                    "reconciliation"
                )
        result = CodingAgentRunActionResultEnvelope(
            protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
            consumer_id=policy.consumer_id,
            consumer_version=policy.consumer_version,
            operation_id=request.operation_id,
            request_digest=request.request_digest,
            structured_output=outcome.structured_output,
            duration_nanoseconds=duration_nanoseconds,
            input_tokens=outcome.input_tokens,
            cached_input_tokens=outcome.cached_input_tokens,
            output_tokens=outcome.output_tokens,
            reasoning_output_tokens=outcome.reasoning_output_tokens,
            cost_usd=outcome.cost_usd,
            provider_event_stream_digest=outcome.provider_event_stream_digest,
            provider_diagnostic_stream_digest=(
                outcome.provider_diagnostic_stream_digest
            ),
            prior_knowledge_accesses=prior_knowledge_accesses,
            edited_source_tree_digest=(
                successor.source_tree_digest if editing else None
            ),
        )
        result.validate_against(policy=policy, request=request)
        payload = result.to_json_bytes()
        return publish_coding_agent_result_candidate(
            temporary_directory_descriptor,
            payload,
            maximum_size_bytes=policy.maximum_raw_result_bytes,
        )


def main() -> None:
    """Execute only the fixed container projection and propagate every failure."""

    if len(sys.argv) != 1:
        raise RunActionCodingAgentConsumerError(
            "coding-agent consumer accepts no arguments"
        )
    os.umask(0o077)
    with ExitStack() as descriptors:
        request_size_limit = _read_native_consumer_request_size_limit()
        request_descriptor = os.open(
            _REQUEST_PATH,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, request_descriptor)
        request_payload = _read_exact_regular_descriptor(
            request_descriptor,
            maximum_bytes=request_size_limit,
            name="coding-agent request",
            allowed_modes={0o600},
        )
        request = read_canonical_coding_agent_request(request_payload)
        if request.interpretation_policy.maximum_request_bytes != request_size_limit:
            raise RunActionCodingAgentConsumerError(
                "coding-agent request bound differs from its trusted projection"
            )
        workspace_descriptor = os.open(
            coding_agent_cli_workspace_path(),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, workspace_descriptor)
        temporary_directory_descriptor = os.open(
            coding_agent_cli_temporary_path(),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, temporary_directory_descriptor)
        consume_coding_agent_run_action(
            request_payload=request_payload,
            workspace_descriptor=workspace_descriptor,
            temporary_directory_descriptor=temporary_directory_descriptor,
            process_runner=BoundedCodingAgentProcessRunner(),
        )


def _require_process_inputs(
    command: tuple[str, ...],
    stdin_payload: bytes | None,
    stdin_directory: str,
    working_directory: str,
    timeout_nanoseconds: int,
    termination_grace_nanoseconds: int,
    maximum_output_bytes: int,
    maximum_diagnostic_bytes: int,
    environment: Mapping[str, str] | None,
) -> None:
    if (
        type(command) is not tuple
        or not command
        or any(not isinstance(value, str) or not value for value in command)
        or (stdin_payload is not None and type(stdin_payload) is not bytes)
        or not isinstance(stdin_directory, str)
        or not PurePosixPath(stdin_directory).is_absolute()
        or not isinstance(working_directory, str)
        or not PurePosixPath(working_directory).is_absolute()
        or type(timeout_nanoseconds) is not int
        or timeout_nanoseconds <= 0
        or type(termination_grace_nanoseconds) is not int
        or not 0 < termination_grace_nanoseconds < timeout_nanoseconds
        or type(maximum_output_bytes) is not int
        or maximum_output_bytes <= 0
        or type(maximum_diagnostic_bytes) is not int
        or maximum_diagnostic_bytes <= 0
        or (
            environment is not None
            and (
                not isinstance(environment, Mapping)
                or any(
                    not isinstance(key, str)
                    or not key
                    or "=" in key
                    or "\x00" in key
                    or not isinstance(value, str)
                    or "\x00" in value
                    for key, value in environment.items()
                )
            )
        )
    ):
        raise RunActionCodingAgentConsumerError(
            "coding-agent process inputs are invalid or unbounded"
        )


def _read_native_consumer_request_size_limit() -> int:
    descriptor = os.open(
        _REQUEST_SIZE_LIMIT_PATH,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        payload = _read_exact_regular_descriptor(
            descriptor,
            maximum_bytes=21,
            name="coding-agent projected request bound",
            allowed_modes={0o600},
        )
    if re.fullmatch(rb"[1-9][0-9]{0,19}\n", payload) is None:
        raise RunActionCodingAgentConsumerError(
            "coding-agent projected request bound is not canonical"
        )
    maximum_request_bytes = int(payload[:-1])
    if maximum_request_bytes > _MAXIMUM_UNSIGNED_64:
        raise RunActionCodingAgentConsumerError(
            "coding-agent projected request bound exceeds its wire integer"
        )
    return maximum_request_bytes


def _wait_and_signal_process(
    process: subprocess.Popen,
    completion_write_descriptor: int,
    completion_keeper_descriptor: int,
) -> None:
    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    if os.write(completion_write_descriptor, b"\x01") != 1:
        raise RunActionCodingAgentConsumerError(
            "coding-agent process completion signal made no progress"
        )
    os.close(completion_write_descriptor)
    os.close(completion_keeper_descriptor)


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.returncode is None:
        _kill_process_group(process)


def _terminate_and_join_process(
    process: subprocess.Popen,
    waiter: Thread,
    termination_grace_nanoseconds: int,
) -> None:
    if process.returncode is None:
        _kill_process_group(process)
    waiter.join(termination_grace_nanoseconds / 1_000_000_000)
    if waiter.is_alive():
        raise RunActionCodingAgentConsumerError(
            "coding-agent process did not terminate within its exact grace"
        )
    process.wait()
    if type(process.returncode) is not int:
        raise RunActionCodingAgentConsumerError(
            "coding-agent process did not terminate within its exact grace"
        )


def _kill_process_group(process: subprocess.Popen) -> None:
    completed = subprocess.run(
        (
            _KILL_EXECUTABLE,
            f"-{signal.Signals(signal.SIGKILL).name}",
            "--",
            f"-{process.pid}",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RunActionCodingAgentConsumerError(
            "coding-agent process-group kill command failed"
        )


def _inspect_workspace(
    workspace_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> RunWorkspaceSourceTreeIdentity:
    policy = request.interpretation_policy
    return inspect_run_workspace_source_tree(
        workspace_descriptor,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )


def _inspect_clean_workspace(
    workspace_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> RunWorkspaceFrontierIdentity:
    policy = request.interpretation_policy
    return inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=policy.workspace_git_branch,
        maximum_source_entries=policy.maximum_workspace_entries,
        maximum_source_bytes=policy.maximum_workspace_bytes,
        maximum_git_entries=policy.maximum_workspace_git_entries,
        maximum_git_bytes=policy.maximum_workspace_git_bytes,
        expected_commit_sha=None,
    )


def _inspect_git_metadata(
    workspace_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> RunWorkspaceRegularTreeIdentity:
    git_descriptor = os.open(
        ".git",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=workspace_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, git_descriptor)
        policy = request.interpretation_policy
        return inspect_run_workspace_regular_tree(
            git_descriptor,
            maximum_entries=policy.maximum_workspace_git_entries,
            maximum_bytes=policy.maximum_workspace_git_bytes,
        )


def _inspect_source_physical_workspace(
    workspace_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> RunWorkspaceRegularTreeIdentity:
    policy = request.interpretation_policy
    return inspect_run_workspace_source_regular_tree(
        workspace_descriptor,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )


def _commit_coding_agent_workspace(
    *,
    request: CodingAgentRunActionRequest,
    predecessor: RunWorkspaceFrontierIdentity,
    process_runner: CodingAgentProcessRunner,
) -> None:
    policy = request.interpretation_policy
    environment = {
        "GIT_AUTHOR_DATE": _GIT_COMMIT_TIME,
        "GIT_AUTHOR_EMAIL": policy.git_commit_author_email,
        "GIT_AUTHOR_NAME": policy.git_commit_author_name,
        "GIT_COMMITTER_DATE": _GIT_COMMIT_TIME,
        "GIT_COMMITTER_EMAIL": policy.git_commit_author_email,
        "GIT_COMMITTER_NAME": policy.git_commit_author_name,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }
    _require_git_completion(
        process_runner.run(
            (
                _GIT_EXECUTABLE,
                *_TRUSTED_GIT_CONFIGURATION_ARGUMENTS,
                "-C",
                coding_agent_cli_workspace_path(),
                "add",
                "--all",
                "--",
                ".",
            ),
            stdin_payload=None,
            stdin_directory=coding_agent_cli_temporary_path(),
            working_directory=coding_agent_cli_workspace_path(),
            timeout_nanoseconds=policy.timeout_nanoseconds,
            termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
            maximum_output_bytes=policy.maximum_provider_output_bytes,
            maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
            environment=environment,
        ),
        expected_output=b"",
        name="Git index update",
    )
    tree = _require_git_object_id_completion(
        process_runner.run(
            (
                _GIT_EXECUTABLE,
                *_TRUSTED_GIT_CONFIGURATION_ARGUMENTS,
                "-C",
                coding_agent_cli_workspace_path(),
                "write-tree",
            ),
            stdin_payload=None,
            stdin_directory=coding_agent_cli_temporary_path(),
            working_directory=coding_agent_cli_workspace_path(),
            timeout_nanoseconds=policy.timeout_nanoseconds,
            termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
            maximum_output_bytes=policy.maximum_provider_output_bytes,
            maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
            environment=environment,
        ),
        name="Git tree creation",
    )
    commit = _require_git_object_id_completion(
        process_runner.run(
            (
                _GIT_EXECUTABLE,
                *_TRUSTED_GIT_CONFIGURATION_ARGUMENTS,
                "-C",
                coding_agent_cli_workspace_path(),
                "commit-tree",
                tree,
                "-p",
                predecessor.commit_sha,
            ),
            stdin_payload=(
                f"Kapso coding-agent action {request.operation_id}\n".encode("ascii")
            ),
            stdin_directory=coding_agent_cli_temporary_path(),
            working_directory=coding_agent_cli_workspace_path(),
            timeout_nanoseconds=policy.timeout_nanoseconds,
            termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
            maximum_output_bytes=policy.maximum_provider_output_bytes,
            maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
            environment=environment,
        ),
        name="Git commit creation",
    )
    _require_git_completion(
        process_runner.run(
            (
                _GIT_EXECUTABLE,
                *_TRUSTED_GIT_CONFIGURATION_ARGUMENTS,
                "-C",
                coding_agent_cli_workspace_path(),
                "update-ref",
                f"refs/heads/{policy.workspace_git_branch}",
                commit,
                predecessor.commit_sha,
            ),
            stdin_payload=None,
            stdin_directory=coding_agent_cli_temporary_path(),
            working_directory=coding_agent_cli_workspace_path(),
            timeout_nanoseconds=policy.timeout_nanoseconds,
            termination_grace_nanoseconds=policy.termination_grace_nanoseconds,
            maximum_output_bytes=policy.maximum_provider_output_bytes,
            maximum_diagnostic_bytes=policy.maximum_provider_diagnostic_bytes,
            environment=environment,
        ),
        expected_output=b"",
        name="Git branch advance",
    )


def _require_git_object_id_completion(
    completion: BoundedCodingAgentProcessCompletion,
    *,
    name: str,
) -> str:
    _require_git_completion(
        completion,
        expected_output=None,
        name=name,
    )
    payload = completion.output_payload
    object_id = payload.removesuffix(b"\n")
    if (
        payload != object_id + b"\n"
        or len(object_id) != 40
        or any(character not in b"0123456789abcdef" for character in object_id)
    ):
        raise RunActionCodingAgentConsumerError(
            f"{name} returned a noncanonical object ID"
        )
    return object_id.decode("ascii")


def _require_git_completion(
    completion: BoundedCodingAgentProcessCompletion,
    *,
    expected_output: bytes | None,
    name: str,
) -> None:
    if (
        type(completion) is not BoundedCodingAgentProcessCompletion
        or completion.return_code != 0
        or completion.diagnostic_payload
        or (
            expected_output is not None and completion.output_payload != expected_output
        )
    ):
        raise RunActionCodingAgentConsumerError(f"{name} did not complete exactly")


def _require_empty_temporary_directory(directory_descriptor: int) -> None:
    metadata = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE
        or metadata.st_nlink < 2
        or tuple(os.listdir(directory_descriptor))
    ):
        raise RunActionCodingAgentConsumerError(
            "coding-agent temporary directory is not empty private authority"
        )


def _materialize_cli_support(
    directory_descriptor: int,
    request: CodingAgentRunActionRequest,
    resources: ExitStack,
) -> tuple[_CodingAgentSupportFileObservation, ...]:
    os.mkdir("home", mode=_PRIVATE_DIRECTORY_MODE, dir_fd=directory_descriptor)
    with ExitStack() as descriptors:
        home_descriptor = os.open(
            "home",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        descriptors.callback(os.close, home_descriptor)
        os.fchmod(home_descriptor, _PRIVATE_DIRECTORY_MODE)
        os.fsync(home_descriptor)
    observations = []
    for absolute_path, payload in coding_agent_cli_support_payloads(request).items():
        path = PurePosixPath(absolute_path)
        if path.parent.as_posix() != coding_agent_cli_temporary_path():
            raise RunActionCodingAgentConsumerError(
                "coding-agent support path escapes its temporary authority"
            )
        observations.append(
            _publish_support_file(
                directory_descriptor,
                path.name,
                payload,
                resources,
            )
        )
    os.fsync(directory_descriptor)
    return tuple(observations)


def _publish_support_file(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    resources: ExitStack,
) -> _CodingAgentSupportFileObservation:
    if (
        not name
        or "/" in name
        or name in {".", "..", "result.candidate"}
        or name in os.listdir(directory_descriptor)
    ):
        raise RunActionCodingAgentConsumerError(
            "coding-agent support file target is invalid or occupied"
        )
    descriptor = open_run_action_anonymous_file(
        directory_descriptor,
        _SUPPORT_FILE_MODE,
    )
    resources.callback(os.close, descriptor)
    write_run_action_full_payload(descriptor, payload)
    require_run_action_descriptor_payload(descriptor, payload)
    os.fchmod(descriptor, _SUPPORT_FILE_MODE)
    os.fsync(descriptor)
    link_run_action_anonymous_file_no_replace(
        descriptor,
        directory_descriptor,
        name,
    )
    os.fsync(directory_descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != _SUPPORT_FILE_MODE
        or metadata.st_size != len(payload)
    ):
        raise RunActionCodingAgentConsumerError(
            "coding-agent support file publication is invalid"
        )
    return _CodingAgentSupportFileObservation(
        name=name,
        descriptor=descriptor,
        payload=payload,
        metadata=_stable_regular_file(metadata),
    )


def _require_cli_support_unchanged(
    directory_descriptor: int,
    observations: tuple[_CodingAgentSupportFileObservation, ...],
) -> None:
    if not observations:
        raise RunActionCodingAgentConsumerError(
            "coding-agent support-file observation is empty"
        )
    for observation in observations:
        if (
            _stable_regular_file(os.fstat(observation.descriptor))
            != observation.metadata
        ):
            raise RunActionCodingAgentConsumerError(
                "coding-agent support file changed during provider execution"
            )
        descriptor = os.open(
            observation.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        with ExitStack() as descriptors:
            descriptors.callback(os.close, descriptor)
            payload = _read_exact_regular_descriptor(
                descriptor,
                maximum_bytes=len(observation.payload),
                name="coding-agent support file",
                allowed_modes={_SUPPORT_FILE_MODE},
            )
            if (
                payload != observation.payload
                or _stable_regular_file(os.fstat(descriptor)) != observation.metadata
            ):
                raise RunActionCodingAgentConsumerError(
                    "coding-agent support path was substituted"
                )


def _read_provider_final(
    temporary_directory_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> bytes | None:
    absolute_path = coding_agent_cli_final_output_path(request)
    if absolute_path is None:
        return None
    path = PurePosixPath(absolute_path)
    if path.parent.as_posix() != coding_agent_cli_temporary_path():
        raise RunActionCodingAgentConsumerError(
            "coding-agent provider final path escapes temporary authority"
        )
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=temporary_directory_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        return _read_exact_regular_descriptor(
            descriptor,
            maximum_bytes=request.interpretation_policy.maximum_raw_result_bytes,
            name="Codex final output",
            allowed_modes={0o600},
        )


def _read_prior_knowledge_accesses(
    temporary_directory_descriptor: int,
    request: CodingAgentRunActionRequest,
) -> tuple[CodingAgentPriorKnowledgeAccessEvent, ...]:
    absolute_path = coding_agent_cli_prior_knowledge_audit_path()
    path = PurePosixPath(absolute_path)
    if path.parent.as_posix() != coding_agent_cli_temporary_path():
        raise RunActionCodingAgentConsumerError(
            "prior-knowledge audit path escapes temporary authority"
        )
    entries = set(os.listdir(temporary_directory_descriptor))
    if path.name not in entries:
        return ()
    if request.prior_knowledge is None:
        raise RunActionCodingAgentConsumerError(
            "coding-agent produced an undeclared prior-knowledge audit"
        )
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=temporary_directory_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        payload = _read_exact_regular_descriptor(
            descriptor,
            maximum_bytes=(
                request.interpretation_policy.maximum_prior_knowledge_audit_bytes
            ),
            name="prior-knowledge audit",
            allowed_modes={0o600},
        )
    return _interpret_prior_knowledge_audit(request, payload)


def _interpret_prior_knowledge_audit(
    request: CodingAgentRunActionRequest,
    payload: bytes,
) -> tuple[CodingAgentPriorKnowledgeAccessEvent, ...]:
    if request.prior_knowledge is None or not payload or not payload.endswith(b"\n"):
        raise RunActionCodingAgentConsumerError(
            "prior-knowledge audit is absent, empty, or incomplete"
        )
    access = PriorKnowledgeAccess(request.prior_knowledge)
    packet = access.packet
    listed_ids = tuple(record["record_id"] for record in access.list_records())
    events = []
    for line in payload.splitlines():
        if not line:
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit contains a blank event"
            )
        event = parse_json_bytes(line)
        expected_fields = {
            "arguments",
            "operation_id",
            "prior_knowledge_snapshot_id",
            "response_digest",
            "returned_ids",
            "tool_name",
        }
        if not isinstance(event, dict) or set(event) != expected_fields:
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit event fields are invalid"
            )
        if (
            canonical_json_bytes(event) != line
            or event["operation_id"] != request.operation_id
            or event["prior_knowledge_snapshot_id"]
            != packet.prior_knowledge_snapshot_id
            or not isinstance(event["arguments"], dict)
            or not isinstance(event["returned_ids"], list)
        ):
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit identity or encoding changed"
            )
        returned_ids = tuple(event["returned_ids"])
        if returned_ids != tuple(sorted(set(returned_ids))) or not isinstance(
            event["response_digest"], str
        ):
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit returned IDs are invalid"
            )
        if event["tool_name"] == "list_prior_knowledge":
            if event["arguments"] or returned_ids != listed_ids:
                raise RunActionCodingAgentConsumerError(
                    "prior-knowledge list audit is inconsistent"
                )
            response_payload = access.list_response_payload()
            access_kind = CodingAgentPriorKnowledgeAccessKind.LIST
            record_id = None
        elif event["tool_name"] == "get_prior_knowledge_record":
            if set(event["arguments"]) != {"record_id"}:
                raise RunActionCodingAgentConsumerError(
                    "prior-knowledge get audit arguments are invalid"
                )
            record_id = event["arguments"]["record_id"]
            if returned_ids != (record_id,):
                raise RunActionCodingAgentConsumerError(
                    "prior-knowledge get audit membership is inconsistent"
                )
            response_payload = access.record_response_payload(record_id)
            access_kind = CodingAgentPriorKnowledgeAccessKind.GET
        else:
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit names an unknown tool"
            )
        response_digest = tree_or_blob_digest(canonical_json_bytes(response_payload))
        if event["response_digest"] != response_digest:
            raise RunActionCodingAgentConsumerError(
                "prior-knowledge audit response digest is inconsistent"
            )
        events.append(
            CodingAgentPriorKnowledgeAccessEvent(
                access_kind=access_kind,
                record_id=record_id,
                returned_record_ids=returned_ids,
                response_digest=response_digest,
            )
        )
    return tuple(events)


def _read_exact_regular_descriptor(
    descriptor: int,
    *,
    maximum_bytes: int,
    name: str,
    allowed_modes: set[int],
) -> bytes:
    before = os.fstat(descriptor)
    if (
        type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_gid != os.getegid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) not in allowed_modes
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise RunActionCodingAgentConsumerError(
            f"{name} is not one complete bounded private file"
        )
    payload = bytearray()
    remaining = before.st_size + 1
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        payload.extend(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if len(payload) != before.st_size or _stable_regular_file(
        after
    ) != _stable_regular_file(before):
        raise RunActionCodingAgentConsumerError(f"{name} changed while reading")
    return bytes(payload)


def _stable_regular_file(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = [
    "BoundedCodingAgentProcessCompletion",
    "BoundedCodingAgentProcessRunner",
    "CodingAgentProcessRunner",
    "NATIVE_CODING_AGENT_CONSUMER_ID",
    "NATIVE_CODING_AGENT_CONSUMER_VERSION",
    "RunActionCodingAgentConsumerError",
    "consume_coding_agent_run_action",
    "main",
]
