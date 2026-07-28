from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch.run_action_coding_agent_consumer import (
    NATIVE_CODING_AGENT_CONSUMER_ID,
    NATIVE_CODING_AGENT_CONSUMER_VERSION,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentPriorKnowledgeAccessKind,
    CodingAgentProviderEgressMode,
    read_canonical_coding_agent_result,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_source_tree,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_coding_agent_consumer import (
    _open_runtime_descriptors,
    _runtime_directories,
)
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)
from test_prior_knowledge_gate import citable_access_materialization

_CONFIG_PATH = Path(__file__).parents[1] / "src" / "kapso" / "config.yaml"
_PRODUCTION_IMAGE = "kapso/coding-agent-production:m9"
_EXPECTED_ANSWER = "authenticated prior-knowledge sidecar passed"


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _restore_fixture_ownership(path: Path) -> None:
    subprocess.run(
        (
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/chown",
            "-R",
            f"{os.geteuid()}:{os.getegid()}",
            "--",
            path.as_posix(),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )


@pytest.fixture
def owned_tmp_path(tmp_path):
    yield tmp_path
    _restore_fixture_ownership(tmp_path)


def _start_broker(
    *,
    socket_path: Path,
    settings,
    resources: ExitStack,
) -> subprocess.Popen:
    readiness_read_descriptor, readiness_write_descriptor = os.pipe2(os.O_CLOEXEC)
    resources.callback(os.close, readiness_read_descriptor)
    command = [
        sys.executable,
        "-m",
        "kapso.cross_run.launch.run_action_coding_agent_egress_broker",
        "--socket-path",
        socket_path.as_posix(),
    ]
    for authority in settings.coding_agent_egress_connect_authorities:
        command.extend(("--authority", authority))
    command.extend(
        (
            "--maximum-header-bytes",
            str(settings.coding_agent_egress_connect_header_size_bytes),
            "--backlog",
            str(settings.coding_agent_egress_relay_backlog),
            "--chunk-size-bytes",
            str(settings.coding_agent_egress_relay_chunk_size_bytes),
            "--connect-timeout-seconds",
            str(settings.coding_agent_egress_connect_timeout_seconds),
            "--readiness-descriptor",
            str(readiness_write_descriptor),
        )
    )
    with os.fdopen(readiness_write_descriptor, "wb"):
        broker = subprocess.Popen(
            tuple(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(readiness_write_descriptor,),
        )
    resources.callback(_terminate_process_group, broker)
    readable, _writable, _exceptional = select.select(
        (readiness_read_descriptor,),
        (),
        (),
        settings.coding_agent_egress_connect_timeout_seconds,
    )
    if (
        readable != [readiness_read_descriptor]
        or os.read(readiness_read_descriptor, 1) != b"\x01"
        or broker.poll() is not None
    ):
        raise AssertionError("host egress broker did not become ready exactly")
    return broker


def test_real_codex_runs_inside_the_network_isolated_production_image(
    owned_tmp_path,
):
    tmp_path = owned_tmp_path
    cross_run = CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])
    settings = cross_run.launch
    workspace, _source, temporary = _runtime_directories(tmp_path)
    input_directory = tmp_path / "input"
    input_directory.mkdir(mode=0o700)
    credential_directory = tmp_path / "credential"
    credential_directory.mkdir(mode=0o700)
    credential_path = credential_directory / "auth.json"
    subprocess.run(
        (
            "/usr/bin/cp",
            "--reflink=never",
            "--",
            settings.coding_agent_codex_auth_source_path,
            credential_path.as_posix(),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    subprocess.run(
        (
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/chown",
            f"{settings.coding_agent_supervisor_user_id}:"
            f"{settings.coding_agent_provider_group_id}",
            "--",
            credential_path.as_posix(),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    credential_path.chmod(0o440)
    with ExitStack() as resources:
        workspace_descriptor, _temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            resources,
        )
        inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )
    policy = interpretation_policy(
        consumer_id=NATIVE_CODING_AGENT_CONSUMER_ID,
        consumer_version=NATIVE_CODING_AGENT_CONSUMER_VERSION,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        model="gpt-5.6-sol",
        effort="xhigh",
        web_search_enabled=False,
        provider_egress_mode=CodingAgentProviderEgressMode.HTTPS_CONNECT_PROXY,
        egress_relay_port=settings.coding_agent_egress_relay_port,
        egress_connect_authorities=settings.coding_agent_egress_connect_authorities,
        maximum_egress_connect_header_bytes=(
            settings.coding_agent_egress_connect_header_size_bytes
        ),
        egress_relay_backlog=settings.coding_agent_egress_relay_backlog,
        egress_relay_chunk_size_bytes=(
            settings.coding_agent_egress_relay_chunk_size_bytes
        ),
        maximum_request_bytes=settings.run_action_request_size_bytes,
        maximum_response_schema_bytes=(
            settings.coding_agent_response_schema_size_bytes
        ),
        maximum_cli_argument_bytes=settings.coding_agent_cli_argument_size_bytes,
        maximum_prior_knowledge_audit_bytes=(
            settings.coding_agent_prior_knowledge_audit_size_bytes
        ),
        prior_knowledge_relay_chunk_size_bytes=(
            settings.coding_agent_prior_knowledge_relay_chunk_size_bytes
        ),
        maximum_raw_result_bytes=settings.run_action_result_size_bytes,
    )
    request = run_action_request(
        policy,
        prior_knowledge=citable_access_materialization(),
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": [_EXPECTED_ANSWER]}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    request = type(request)(
        **{
            **request.to_dict(),
            "prompt": (
                "Without changing the workspace, you MUST first call "
                "prior_knowledge.list_prior_knowledge, then call "
                "prior_knowledge.get_prior_knowledge_record for the single record "
                "returned by that list. Only after both calls succeed, return the "
                f"required JSON answer exactly as {_EXPECTED_ANSWER!r}."
            ),
        }
    )
    request_path = input_directory / "request.blob"
    request_path.write_bytes(request.to_json_bytes())
    request_path.chmod(0o400)
    broker_directory = tmp_path / "broker"
    broker_directory.mkdir(mode=0o700)
    broker_path = broker_directory / "broker.sock"
    with ExitStack() as resources:
        broker = _start_broker(
            socket_path=broker_path,
            settings=settings,
            resources=resources,
        )
        completed = subprocess.run(
            (
                "/usr/bin/docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "KILL",
                "--cap-add",
                "SETGID",
                "--cap-add",
                "SETPCAP",
                "--cap-add",
                "SETUID",
                "--group-add",
                str(policy.provider_group_id),
                "--security-opt",
                "apparmor=docker-default",
                "--security-opt",
                "seccomp=builtin",
                "--pids-limit",
                "128",
                "--init",
                "--user",
                f"{policy.supervisor_user_id}:{policy.supervisor_group_id}",
                "--workdir",
                "/kapso/workspace",
                "--mount",
                f"type=bind,src={workspace},dst=/kapso/workspace",
                "--mount",
                f"type=bind,src={input_directory},dst=/kapso/input,readonly",
                "--mount",
                f"type=bind,src={temporary},dst=/kapso/tmp",
                "--mount",
                f"type=bind,src={credential_path},"
                "dst=/kapso/credentials/credentials,readonly",
                "--mount",
                f"type=bind,src={broker_path},"
                "dst=/kapso/egress/broker.sock,readonly",
                _PRODUCTION_IMAGE,
                "/usr/local/bin/kapso-coding-agent-supervisor",
                "--supervisor-user-id",
                str(policy.supervisor_user_id),
                "--supervisor-group-id",
                str(policy.supervisor_group_id),
                "--provider-user-id",
                str(policy.provider_user_id),
                "--provider-group-id",
                str(policy.provider_group_id),
                "--maximum-request-bytes",
                str(policy.maximum_request_bytes),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=policy.timeout_nanoseconds / 1_000_000_000,
        )
        if completed.returncode != 0:
            _terminate_process_group(broker)
            broker_diagnostic = b"" if broker.stderr is None else broker.stderr.read()
            raise AssertionError(
                "real coding-agent image failed:\n"
                + completed.stderr.decode("utf-8")
                + "\nhost broker:\n"
                + broker_diagnostic.decode("utf-8")
            )
    result = read_canonical_coding_agent_result(
        (temporary / "result.candidate").read_bytes()
    )
    result.validate_against(policy=policy, request=request)
    assert result.structured_output == {"answer": _EXPECTED_ANSWER}
    assert tuple(access.access_kind for access in result.prior_knowledge_accesses) == (
        CodingAgentPriorKnowledgeAccessKind.LIST,
        CodingAgentPriorKnowledgeAccessKind.GET,
    )
    assert result.prior_knowledge_accesses[1].returned_record_ids == (
        request.prior_knowledge.prior_knowledge_snapshot.selected_record_ids[0],
    )
    assert result.input_tokens > 0
    assert result.output_tokens > 0
