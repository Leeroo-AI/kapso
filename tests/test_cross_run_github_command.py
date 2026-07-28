import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import kapso.cross_run.github.command as command_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.github.command import (
    BoundedJsonResponse,
    CommandOutputKind,
    CommandRequest,
    CommandResult,
    GitHubCommandClient,
    GitHubCompareAndSwapError,
    GitHubCommandError,
    SubprocessCommandRunner,
)
from kapso.cross_run.settings import CrossRunSettings
from cross_run_github_fixtures import release_attestation

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def github_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).github


@dataclass
class RecordingRunner:
    outputs: list[object]
    requests: list[CommandRequest] = field(default_factory=list)

    def run(self, request):
        self.requests.append(request)
        output = self.outputs.pop(0)
        if request.output_kind is CommandOutputKind.FILE:
            assert isinstance(output, bytes)
            request.output_path.write_bytes(output)
            stdout = b""
            result_output = request.output_path
        else:
            stdout = (
                output if isinstance(output, bytes) else canonical_json_bytes(output)
            )
            result_output = output
        return CommandResult(
            request=request,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            output=result_output,
        )


@dataclass
class ScriptedRunner:
    responses: list[tuple[int, object, bytes]]
    requests: list[CommandRequest] = field(default_factory=list)

    def run(self, request):
        self.requests.append(request)
        returncode, output, stderr = self.responses.pop(0)
        stdout = canonical_json_bytes(output) if output is not None else b""
        return CommandResult(
            request=request,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            output=output,
        )


def test_github_client_builds_exact_safe_api_argv_and_canonical_stdin(tmp_path):
    settings = github_settings()
    runner = RecordingRunner(outputs=[{"sha": "a" * 40}, {"bounded": True}])
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    response = client.api_json(
        "POST",
        "repos/Leeroo-AI/kapso-knowledge/git/blobs",
        {"encoding": "base64", "content": "YQ=="},
    )

    request = runner.requests[0]
    assert response == {"sha": "a" * 40}
    assert request.argv == (
        "gh",
        "api",
        "--method",
        "POST",
        "--header",
        f"X-GitHub-Api-Version:{settings.api_version}",
        "repos/Leeroo-AI/kapso-knowledge/git/blobs",
        "--input",
        "-",
    )
    assert request.stdin == b'{"content":"YQ==","encoding":"base64"}'
    assert request.cwd == tmp_path
    assert request.timeout_seconds == settings.command_timeout_seconds
    assert request.maximum_output_bytes == settings.control_blob_size_bytes
    assert client.api_json_bounded(
        "GET",
        "repos/Leeroo-AI/kapso-knowledge/git/trees/" + "b" * 40,
        settings.git_tree_metadata_size_bytes,
    ) == BoundedJsonResponse(
        value={"bounded": True},
        size_bytes=len(b'{"bounded":true}'),
    )
    assert runner.requests[1].output_kind is CommandOutputKind.JSON
    assert (
        runner.requests[1].maximum_output_bytes == settings.git_tree_metadata_size_bytes
    )


def test_release_commands_never_embed_credentials_or_use_shell_strings(tmp_path):
    settings = github_settings()
    asset = tmp_path / "snapshot.tar.zst"
    asset.write_bytes(b"asset")
    commit_sha = "a" * 40
    asset_digests = {asset.name: tree_or_blob_digest(asset.read_bytes())}
    attestation = release_attestation(
        "Leeroo-AI/kapso-knowledge",
        "knowledge/S000001",
        commit_sha,
        asset_digests,
    )
    runner = RecordingRunner(
        outputs=[
            {
                "name": "declared-snapshot.tar.zst",
                "content_type": "application/zstd",
                "size": asset.stat().st_size,
            },
            f"gh version {settings.minimum_cli_version}\n",
            attestation,
            b"asset",
            b"source",
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    client.upload_release_asset(
        "Leeroo-AI/kapso-knowledge",
        7,
        asset,
        "declared-snapshot.tar.zst",
        "application/zstd",
        asset.stat().st_size,
    )
    assert (
        client.verify_release(
            "Leeroo-AI/kapso-knowledge",
            "knowledge/S000001",
            commit_sha,
            asset_digests,
        )
        == attestation
    )
    assert (
        attestation["verificationResult"]["statement"]["predicate"]["purl"]
        == "pkg:github/Leeroo-AI/kapso-knowledge@knowledge%2FS000001"
    )
    destination = tmp_path / "downloaded.tar.zst"
    assert (
        client.download_release_asset(
            "Leeroo-AI/kapso-knowledge",
            "42",
            destination,
            settings.release_asset_size_bytes,
        )
        == destination
    )
    assert destination.read_bytes() == b"asset"
    blob_sha = "b" * 40
    assert (
        client.read_git_blob(
            "Leeroo-AI/kapso-knowledge",
            blob_sha,
            settings.release_asset_size_bytes,
        )
        == b"source"
    )

    assert runner.requests[0].argv == (
        "gh",
        "api",
        "--method",
        "POST",
        "--header",
        "Accept:application/vnd.github+json",
        "--header",
        f"X-GitHub-Api-Version:{settings.api_version}",
        "--header",
        "Content-Type:application/zstd",
        "--input",
        str(asset),
        "https://uploads.github.com/repos/Leeroo-AI/kapso-knowledge/"
        "releases/7/assets?name=declared-snapshot.tar.zst",
    )
    assert runner.requests[1].argv == ("gh", "version")
    assert runner.requests[2].argv[:4] == (
        "gh",
        "release",
        "verify",
        "knowledge/S000001",
    )
    assert "Accept:application/octet-stream" in runner.requests[3].argv
    assert "Accept:application/vnd.github.raw+json" in runner.requests[4].argv
    assert runner.requests[4].argv[-1].endswith(f"/git/blobs/{blob_sha}")
    assert all(
        "token" not in argument.lower()
        for request in runner.requests
        for argument in request.argv
    )


@pytest.mark.parametrize(
    ("asset_name", "media_type"),
    [
        ("../escape.tar", "application/x-tar"),
        ("snapshot.tar", "application/x-tar\r\nX-Injected:true"),
    ],
)
def test_release_upload_rejects_unstable_or_injectable_metadata(
    tmp_path, asset_name, media_type
):
    asset = tmp_path / "snapshot.tar"
    asset.write_bytes(b"asset")
    runner = RecordingRunner(outputs=[])
    settings = github_settings()
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    with pytest.raises(GitHubCommandError):
        client.upload_release_asset(
            "Leeroo-AI/kapso-knowledge",
            7,
            asset,
            asset_name,
            media_type,
            asset.stat().st_size,
        )

    assert runner.requests == []


def test_release_asset_deletion_uses_exact_authenticated_api_endpoint(tmp_path):
    settings = github_settings()
    runner = RecordingRunner(outputs=[b""])
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    client.delete_release_asset("Leeroo-AI/kapso-knowledge", 11)

    assert runner.requests[0].argv == (
        "gh",
        "api",
        "--method",
        "DELETE",
        "--header",
        f"X-GitHub-Api-Version:{settings.api_version}",
        "repos/Leeroo-AI/kapso-knowledge/releases/assets/11",
    )


def test_subprocess_runner_parses_strict_json_and_treats_metacharacters_as_data(
    tmp_path,
):
    settings = github_settings()
    marker = tmp_path / "must-not-exist"
    argument = f"$(touch {marker})"
    request = CommandRequest(
        argv=(
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps({'argument': sys.argv[1]}))",
            argument,
        ),
        cwd=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        output_kind=CommandOutputKind.JSON,
        maximum_output_bytes=settings.control_blob_size_bytes,
    )

    result = SubprocessCommandRunner().run(request)

    assert result.output == {"argument": argument}
    assert not marker.exists()


def test_subprocess_runner_is_safe_from_threaded_callers(tmp_path):
    runner = SubprocessCommandRunner()

    def run_command(position):
        request = CommandRequest(
            argv=(sys.executable, "-c", f"print({position})"),
            cwd=tmp_path,
            timeout_seconds=5,
            output_kind=CommandOutputKind.TEXT,
            maximum_output_bytes=128,
        )
        return runner.run(request).output

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = tuple(executor.map(run_command, range(16)))

    assert outputs == tuple(f"{position}\n" for position in range(16))
    assert tuple(tmp_path.iterdir()) == ()


def test_command_boundary_fails_loud_without_secret_bearing_argv(tmp_path):
    settings = github_settings()
    with pytest.raises(GitHubCommandError):
        CommandRequest(
            argv=("gh", "api", "--header", "Authorization: Bearer secret"),
            cwd=tmp_path,
            timeout_seconds=settings.command_timeout_seconds,
            output_kind=CommandOutputKind.TEXT,
            maximum_output_bytes=settings.control_blob_size_bytes,
        )

    request = CommandRequest(
        argv=(
            sys.executable,
            "-c",
            "import os,sys; sys.stderr.write(sys.stdin.read()); "
            "sys.stderr.flush(); os.close(1); os.close(2); raise SystemExit(7)",
        ),
        cwd=Path(tmp_path),
        timeout_seconds=settings.command_timeout_seconds,
        output_kind=CommandOutputKind.TEXT,
        maximum_output_bytes=settings.control_blob_size_bytes,
        stdin=b"github_pat_secretvalue",
    )
    with pytest.raises(GitHubCommandError, match="exit 7") as failure:
        SubprocessCommandRunner().run(request)
    assert "github_pat_secretvalue" not in str(failure.value)
    assert "[REDACTED]" in str(failure.value)

    with pytest.raises(GitHubCommandError):
        CommandRequest(
            argv=("git", "fetch", "https://user:secret@github.com/org/repo.git"),
            cwd=tmp_path,
            timeout_seconds=settings.command_timeout_seconds,
            output_kind=CommandOutputKind.TEXT,
            maximum_output_bytes=settings.control_blob_size_bytes,
        )


def test_file_output_is_stopped_at_the_configured_bound(tmp_path):
    settings = github_settings()
    destination = tmp_path / "bounded-output"
    request = CommandRequest(
        argv=(
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x' * 4096)",
        ),
        cwd=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        output_kind=CommandOutputKind.FILE,
        output_path=destination,
        maximum_output_bytes=1024,
    )

    with pytest.raises(GitHubCommandError):
        SubprocessCommandRunner().run(request)

    assert destination.stat().st_size <= 1024


def test_text_stderr_is_stopped_before_it_can_be_captured_unbounded(tmp_path):
    settings = github_settings()
    request = CommandRequest(
        argv=(
            sys.executable,
            "-c",
            "import sys; sys.stderr.buffer.write(b'x' * 4096)",
        ),
        cwd=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        output_kind=CommandOutputKind.TEXT,
        maximum_output_bytes=1024,
    )

    with pytest.raises(GitHubCommandError, match="stderr exceeds"):
        SubprocessCommandRunner().run(request)


@pytest.mark.parametrize(
    ("observed_sha", "expected_exception"),
    [
        ("b" * 40, None),
        ("c" * 40, GitHubCompareAndSwapError),
        ("a" * 40, GitHubCommandError),
    ],
)
def test_ref_update_reconciles_uncertain_failure_by_observing_remote_state(
    tmp_path, observed_sha, expected_exception
):
    settings = github_settings()
    runner = ScriptedRunner(
        responses=[
            (1, None, b"remote rejected github_pat_secretvalue"),
            (0, {"object": {"sha": observed_sha}}, b""),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    if expected_exception is None:
        result = client.update_ref_compare_and_swap(
            "Leeroo-AI/kapso-knowledge",
            "repository-node",
            "main",
            "a" * 40,
            "b" * 40,
        )
        assert result == {"object": {"sha": "b" * 40}}
    else:
        with pytest.raises(expected_exception) as failure:
            client.update_ref_compare_and_swap(
                "Leeroo-AI/kapso-knowledge",
                "repository-node",
                "main",
                "a" * 40,
                "b" * 40,
            )
        assert "github_pat_secretvalue" not in str(failure.value)


def test_atomic_ref_update_sends_expected_parent_and_observes_requested_commit(
    tmp_path,
):
    settings = github_settings()
    runner = ScriptedRunner(
        responses=[
            (0, {"data": {"updateRefs": {"clientMutationId": None}}}, b""),
            (0, {"object": {"sha": "b" * 40}}, b""),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    result = client.update_ref_compare_and_swap(
        "Leeroo-AI/kapso-knowledge",
        "repository-node",
        "main",
        "a" * 40,
        "b" * 40,
    )

    assert result == {"object": {"sha": "b" * 40}}
    request = parse_json_bytes(runner.requests[0].stdin)
    assert request["variables"]["input"] == {
        "refUpdates": [
            {
                "afterOid": "b" * 40,
                "beforeOid": "a" * 40,
                "force": False,
                "name": "refs/heads/main",
            }
        ],
        "repositoryId": "repository-node",
    }


def test_successful_atomic_ref_update_rejects_immediate_supersession(tmp_path):
    settings = github_settings()
    runner = ScriptedRunner(
        responses=[
            (0, {"data": {"updateRefs": {"clientMutationId": None}}}, b""),
            (0, {"object": {"sha": "c" * 40}}, b""),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    with pytest.raises(GitHubCompareAndSwapError):
        client.update_ref_compare_and_swap(
            "Leeroo-AI/kapso-knowledge",
            "repository-node",
            "main",
            "a" * 40,
            "b" * 40,
        )


def test_artifact_identity_ref_creation_is_write_once_and_reconciles_failure(
    tmp_path,
):
    settings = github_settings()
    qualified_ref = "refs/kapso-artifacts/knowledge_snapshot/" + "a" * 64
    commit_sha = "b" * 40
    existing = {"ref": qualified_ref, "object": {"sha": commit_sha}}
    runner = ScriptedRunner(
        responses=[
            (1, None, b"uncertain network result"),
            (0, existing, b""),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    assert (
        client.create_ref_if_absent(
            "Leeroo-AI/kapso-knowledge", qualified_ref, commit_sha
        )
        == existing
    )
    assert runner.requests[0].stdin == canonical_json_bytes(
        {"ref": qualified_ref, "sha": commit_sha}
    )
    assert runner.requests[1].argv[-1].endswith(qualified_ref.removeprefix("refs/"))

    conflict_runner = ScriptedRunner(
        responses=[
            (1, None, b"already exists"),
            (
                0,
                {"ref": qualified_ref, "object": {"sha": "c" * 40}},
                b"",
            ),
        ]
    )
    conflict_client = GitHubCommandClient(
        conflict_runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )
    with pytest.raises(GitHubCompareAndSwapError):
        conflict_client.create_ref_if_absent(
            "Leeroo-AI/kapso-knowledge", qualified_ref, commit_sha
        )


def test_custom_ref_reader_uses_rest_and_distinguishes_only_exact_404(tmp_path):
    settings = github_settings()
    qualified_ref = "refs/kapso-artifacts/knowledge_snapshot/" + "a" * 64
    commit_sha = "b" * 40
    runner = ScriptedRunner(
        responses=[
            (
                0,
                {
                    "ref": qualified_ref,
                    "object": {"type": "commit", "sha": commit_sha},
                },
                b"",
            ),
            (1, None, b"gh: Reference does not exist (HTTP 404)\n"),
            (1, None, b"gh: transport unavailable (HTTP 503)\n"),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    assert (
        client.read_ref_commit(
            "Leeroo-AI/kapso-knowledge",
            qualified_ref,
            allow_missing=True,
        )
        == commit_sha
    )
    assert (
        client.read_ref_commit(
            "Leeroo-AI/kapso-knowledge",
            qualified_ref,
            allow_missing=True,
        )
        is None
    )
    with pytest.raises(GitHubCommandError, match="503"):
        client.read_ref_commit(
            "Leeroo-AI/kapso-knowledge",
            qualified_ref,
            allow_missing=True,
        )
    assert runner.requests[0].argv[-1].endswith(qualified_ref.removeprefix("refs/"))


def test_release_verification_rejects_cli_without_secure_attestation_support(tmp_path):
    settings = github_settings()
    runner = RecordingRunner(outputs=["gh version 2.92.0\n"])
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    with pytest.raises(GitHubCommandError, match=settings.minimum_cli_version):
        client.verify_release(
            "Leeroo-AI/kapso-knowledge",
            "knowledge/S000001",
            "a" * 40,
            {"snapshot.tar": tree_or_blob_digest(b"asset")},
        )

    assert tuple(request.argv for request in runner.requests) == (("gh", "version"),)


def test_release_verification_waits_for_github_attestation_visibility(
    tmp_path,
    monkeypatch,
):
    settings = github_settings()
    repository = "Leeroo-AI/kapso-knowledge"
    tag = "knowledge/S000001"
    commit_sha = "a" * 40
    asset_digests = {"snapshot.tar": tree_or_blob_digest(b"asset")}
    attestation = release_attestation(
        repository,
        tag,
        commit_sha,
        asset_digests,
    )
    runner = ScriptedRunner(
        responses=[
            (1, None, b"no attestations for tag knowledge/S000001\n"),
            (0, attestation, b""),
        ]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )
    client.release_verifier_ready = True
    monkeypatch.setattr(command_module.time, "sleep", lambda _seconds: None)

    assert (
        client.verify_release(
            repository,
            tag,
            commit_sha,
            asset_digests,
        )
        == attestation
    )
    assert len(runner.requests) == 2


def test_release_verification_rejects_unbound_or_malformed_success_output(tmp_path):
    settings = github_settings()
    repository = "Leeroo-AI/kapso-knowledge"
    tag = "knowledge/S000001"
    commit_sha = "a" * 40
    assets = {"snapshot.tar": tree_or_blob_digest(b"asset")}
    runner = RecordingRunner(
        outputs=[f"gh version {settings.minimum_cli_version}\n", {}]
    )
    client = GitHubCommandClient(
        runner,
        working_directory=tmp_path,
        timeout_seconds=settings.command_timeout_seconds,
        api_version=settings.api_version,
        minimum_cli_version=settings.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.control_blob_size_bytes,
    )

    with pytest.raises(GitHubCommandError):
        client.verify_release(repository, tag, commit_sha, assets)

    wrong_assets = release_attestation(
        repository,
        tag,
        commit_sha,
        {"other.tar": tree_or_blob_digest(b"asset")},
    )
    runner.outputs.append(wrong_assets)
    with pytest.raises(GitHubCommandError, match="asset closure"):
        client.verify_release(repository, tag, commit_sha, assets)
