"""Safe, fakeable command boundary for GitHub transport."""

from __future__ import annotations

import base64
import io
import re
import selectors
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Thread
from typing import Any, Mapping, Protocol
from urllib.parse import quote, urlencode

from kapso.cross_run.canonical import canonical_json_bytes, parse_json_bytes
from kapso.cross_run.git_refs import require_git_ref_name


class GitHubCommandError(RuntimeError):
    """A framework-owned GitHub command was invalid or failed."""


class GitHubCompareAndSwapError(GitHubCommandError):
    """A remote ref no longer has the caller's expected parent."""


_SECRET_DIAGNOSTIC_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]+", re.IGNORECASE),
    re.compile(r"gh[opusr]_[A-Za-z0-9_]+", re.IGNORECASE),
    re.compile(r"Authorization:\s*[^\r\n]+", re.IGNORECASE),
    re.compile(r"(https?://)[^/@\s]+:[^/@\s]+@", re.IGNORECASE),
)
_CLI_VERSION_PATTERN = re.compile(r"^gh version ([0-9]+)\.([0-9]+)\.([0-9]+)(?:\s|$)")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RELEASE_ASSET_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+$"
)
_RELEASE_PREDICATE_TYPE = "https://in-toto.io/attestation/release/v0.2"
_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
_STATEMENT_PAYLOAD_TYPE = "application/vnd.in-toto+json"
_UPDATE_REFS_MUTATION = """
mutation($input: UpdateRefsInput!) {
  updateRefs(input: $input) { clientMutationId }
}
"""


def _redact_diagnostics(value: str) -> str:
    redacted = value
    for pattern in _SECRET_DIAGNOSTIC_PATTERNS:
        redacted = pattern.sub(
            (
                r"\1[REDACTED]@"
                if pattern is _SECRET_DIAGNOSTIC_PATTERNS[-1]
                else "[REDACTED]"
            ),
            redacted,
        )
    return redacted


def _require_mapping(
    value: Any, name: str, error_type: type[RuntimeError] = GitHubCommandError
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{name} must be an object")
    return value


def validate_release_attestation(
    value: Any,
    *,
    repository: str,
    tag: str,
    commit_sha: str,
    asset_digests: Mapping[str, str],
    error_type: type[RuntimeError] = GitHubCommandError,
) -> Mapping[str, Any]:
    """Validate verified GitHub output and its exact release subject closure."""
    result = _require_mapping(value, "release verification result", error_type)
    attestation = _require_mapping(
        result.get("attestation"), "release attestation", error_type
    )
    bundle = _require_mapping(
        attestation.get("bundle"), "release attestation bundle", error_type
    )
    envelope = _require_mapping(
        bundle.get("dsseEnvelope"), "release attestation DSSE envelope", error_type
    )
    if envelope.get("payloadType") != _STATEMENT_PAYLOAD_TYPE:
        raise error_type("release attestation payload type is invalid")
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise error_type("release attestation has no signature")
    encoded_payload = envelope.get("payload")
    if not isinstance(encoded_payload, str) or not encoded_payload:
        raise error_type("release attestation payload is missing")
    statement = _require_mapping(
        parse_json_bytes(base64.b64decode(encoded_payload, validate=True)),
        "release attestation statement",
        error_type,
    )
    verification = _require_mapping(
        result.get("verificationResult"), "release verification metadata", error_type
    )
    verified_statement = _require_mapping(
        verification.get("statement"), "verified release statement", error_type
    )
    if canonical_json_bytes(statement) != canonical_json_bytes(verified_statement):
        raise error_type("verified release statement differs from its bundle")
    if statement.get("_type") != _STATEMENT_TYPE:
        raise error_type("release attestation statement type is invalid")
    if statement.get("predicateType") != _RELEASE_PREDICATE_TYPE:
        raise error_type("release attestation predicate type is invalid")
    predicate = _require_mapping(
        statement.get("predicate"), "release attestation predicate", error_type
    )
    package_uri = f"pkg:github/{repository}@{quote(tag, safe='')}"
    if (
        predicate.get("repository") != repository
        or predicate.get("tag") != tag
        or predicate.get("purl") != package_uri
    ):
        raise error_type("release attestation predicate target mismatch")
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or not subjects:
        raise error_type("release attestation subjects are missing")
    release_subjects = []
    attested_assets: dict[str, str] = {}
    for subject_value in subjects:
        subject = _require_mapping(
            subject_value, "release attestation subject", error_type
        )
        digest = _require_mapping(
            subject.get("digest"), "release subject digest", error_type
        )
        if subject.get("uri") == package_uri:
            release_subjects.append(digest)
            continue
        name = subject.get("name")
        if not isinstance(name, str) or not name or name in attested_assets:
            raise error_type("release attestation asset subject is invalid")
        sha256 = digest.get("sha256")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise error_type("release attestation asset digest is invalid")
        attested_assets[name] = f"sha256:{sha256}"
    if len(release_subjects) != 1 or release_subjects[0].get("sha1") != commit_sha:
        raise error_type("release attestation commit subject mismatch")
    if dict(sorted(attested_assets.items())) != dict(sorted(asset_digests.items())):
        raise error_type("release attestation asset closure mismatch")
    return result


class CommandOutputKind(str, Enum):
    BINARY = "binary"
    FILE = "file"
    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True)
class CommandRequest:
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: int
    output_kind: CommandOutputKind
    maximum_output_bytes: int
    stdin: bytes | None = None
    capture_failure: bool = False
    output_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.argv or any(
            not isinstance(argument, str) or not argument for argument in self.argv
        ):
            raise GitHubCommandError("command argv must contain non-empty strings")
        if not self.cwd.is_dir():
            raise GitHubCommandError("command cwd must be an existing directory")
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise GitHubCommandError("command timeout must be a positive integer")
        if type(self.maximum_output_bytes) is not int or self.maximum_output_bytes <= 0:
            raise GitHubCommandError("command output limit must be positive")
        if self.output_kind is CommandOutputKind.FILE:
            if self.output_path is None or self.output_path.exists():
                raise GitHubCommandError("file output path must be absent")
        elif self.output_path is not None:
            raise GitHubCommandError("output path requires file output")
        lowered = tuple(argument.lower() for argument in self.argv)
        forbidden = ("authorization:", "--token", "--api-token", "password=")
        if any(marker in argument for marker in forbidden for argument in lowered):
            raise GitHubCommandError("credentials are forbidden in command argv")
        if any(
            pattern.search(argument)
            for pattern in _SECRET_DIAGNOSTIC_PATTERNS
            for argument in self.argv
        ):
            raise GitHubCommandError("credentials are forbidden in command argv")


@dataclass(frozen=True)
class CommandResult:
    request: CommandRequest
    returncode: int
    stdout: bytes
    stderr: bytes
    output: Any


@dataclass(frozen=True)
class BoundedJsonResponse:
    """Parsed JSON plus the exact number of transport bytes consumed."""

    value: Any
    size_bytes: int


class CommandRunner(Protocol):
    def run(self, request: CommandRequest) -> CommandResult:
        """Execute one already-validated request."""


class SubprocessCommandRunner:
    """Execute argv directly while inheriting external Git/gh authentication."""

    def run(self, request: CommandRequest) -> CommandResult:
        stdout_handle = (
            request.output_path.open("xb")
            if request.output_kind is CommandOutputKind.FILE
            else io.BytesIO()
        )
        stderr_handle = io.BytesIO()
        with stdout_handle, stderr_handle, tempfile.TemporaryFile() as stdin_handle:
            if request.stdin is not None:
                stdin_handle.write(request.stdin)
                stdin_handle.seek(0)
            returncode = self._execute(
                request,
                stdin_handle,
                stdout_handle,
                stderr_handle,
            )
            stdout = (
                b""
                if request.output_kind is CommandOutputKind.FILE
                else stdout_handle.getvalue()
            )
            stderr_bytes = stderr_handle.getvalue()
        if returncode != 0:
            stderr = _redact_diagnostics(stderr_bytes.decode("utf-8", errors="replace"))
            if not request.capture_failure:
                raise GitHubCommandError(
                    f"command failed with exit {returncode}: "
                    f"program={request.argv[0]!r}; stderr={stderr}"
                )
            return CommandResult(
                request=request,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr.encode("utf-8"),
                output=None,
            )
        if request.output_kind is CommandOutputKind.JSON:
            output = parse_json_bytes(stdout)
        elif request.output_kind is CommandOutputKind.TEXT:
            output = stdout.decode("utf-8")
        elif request.output_kind is CommandOutputKind.FILE:
            output = request.output_path
        else:
            output = stdout
        return CommandResult(
            request=request,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr_bytes,
            output=output,
        )

    def _execute(
        self,
        request: CommandRequest,
        stdin: Any,
        stdout: Any,
        stderr: Any,
    ) -> int:
        process = subprocess.Popen(
            list(request.argv),
            cwd=request.cwd,
            stdin=stdin if request.stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        process_completed = Event()
        process_waiter = Thread(
            target=self._wait_for_process,
            args=(process, process_completed),
            daemon=True,
        )
        process_waiter.start()
        if process.stdout is None or process.stderr is None:
            self._terminate_process(process, process_completed, process_waiter)
            raise GitHubCommandError("command pipes were not created")
        streams = {
            process.stdout: (stdout, "stdout"),
            process.stderr: (stderr, "stderr"),
        }
        observed = {"stdout": 0, "stderr": 0}
        deadline = time.monotonic() + request.timeout_seconds
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    self._terminate_process(process, process_completed, process_waiter)
                    raise GitHubCommandError(
                        f"command timed out: program={request.argv[0]!r}"
                    )
                events = selector.select(remaining_seconds)
                if not events:
                    self._terminate_process(process, process_completed, process_waiter)
                    raise GitHubCommandError(
                        f"command timed out: program={request.argv[0]!r}"
                    )
                for key, _ in events:
                    stream = key.fileobj
                    destination, label = streams[stream]
                    chunk = stream.read1(io.DEFAULT_BUFFER_SIZE)
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    available = request.maximum_output_bytes - observed[label]
                    destination.write(chunk[:available])
                    observed[label] += len(chunk)
                    if observed[label] > request.maximum_output_bytes:
                        self._terminate_process(
                            process, process_completed, process_waiter
                        )
                        raise GitHubCommandError(
                            f"command {label} exceeds configured limit"
                        )
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0 or not process_completed.wait(remaining_seconds):
            self._terminate_process(process, process_completed, process_waiter)
            raise GitHubCommandError(f"command timed out: program={request.argv[0]!r}")
        process_waiter.join()
        if process.returncode is None:
            raise GitHubCommandError("command completion state is invalid")
        return process.returncode

    @staticmethod
    def _wait_for_process(process: subprocess.Popen, completed: Event) -> None:
        process.wait()
        completed.set()

    @staticmethod
    def _terminate_process(
        process: subprocess.Popen,
        completed: Event,
        waiter: Thread,
    ) -> None:
        process.kill()
        completed.wait()
        waiter.join()


class GitHubCommandClient:
    """Build the sole framework-owned GitHub CLI argv shapes."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        working_directory: Path,
        timeout_seconds: int,
        api_version: str,
        minimum_cli_version: str,
        control_blob_size_bytes: int,
    ) -> None:
        if not working_directory.is_dir():
            raise GitHubCommandError("GitHub working directory must exist")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise GitHubCommandError("GitHub command timeout must be positive")
        self.runner = runner
        self.working_directory = working_directory
        self.timeout_seconds = timeout_seconds
        self.api_version = api_version
        if type(control_blob_size_bytes) is not int or control_blob_size_bytes <= 0:
            raise GitHubCommandError("GitHub control output limit must be positive")
        self.control_blob_size_bytes = control_blob_size_bytes
        minimum_match = re.fullmatch(
            r"([0-9]+)\.([0-9]+)\.([0-9]+)", minimum_cli_version
        )
        if minimum_match is None:
            raise GitHubCommandError("minimum GitHub CLI version is invalid")
        self.minimum_cli_version = tuple(
            int(component) for component in minimum_match.groups()
        )
        self.release_verifier_ready = False

    def api_json(
        self,
        method: str,
        endpoint: str,
        body: Any | None = None,
    ) -> Any:
        argv = (
            "gh",
            "api",
            "--method",
            method,
            "--header",
            f"X-GitHub-Api-Version:{self.api_version}",
            endpoint,
        )
        stdin = None
        if body is not None:
            argv = (*argv, "--input", "-")
            stdin = canonical_json_bytes(body)
        request = CommandRequest(
            argv=argv,
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            stdin=stdin,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        return self.runner.run(request).output

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        return self.api_json(
            "POST",
            "graphql",
            {"query": query, "variables": variables},
        )

    def api_json_bounded(
        self,
        method: str,
        endpoint: str,
        maximum_bytes: int,
    ) -> BoundedJsonResponse:
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise GitHubCommandError("bounded API output limit must be positive")
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                method,
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                endpoint,
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            maximum_output_bytes=maximum_bytes,
        )
        result = self.runner.run(request)
        return BoundedJsonResponse(value=result.output, size_bytes=len(result.stdout))

    def upload_release_asset(
        self,
        repository: str,
        release_id: int,
        asset_path: Path,
        asset_name: str,
        media_type: str,
        asset_size: int,
    ) -> Mapping[str, Any]:
        if type(release_id) is not int or release_id < 1:
            raise GitHubCommandError("release asset upload ID must be positive")
        if _REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise GitHubCommandError("release asset upload repository is invalid")
        if _RELEASE_ASSET_NAME_PATTERN.fullmatch(asset_name) is None:
            raise GitHubCommandError("release asset upload name is invalid")
        if _MEDIA_TYPE_PATTERN.fullmatch(media_type) is None:
            raise GitHubCommandError("release asset upload media type is invalid")
        if not asset_path.is_file() or asset_path.is_symlink():
            raise GitHubCommandError("release asset upload path must be a regular file")
        if asset_size != asset_path.stat().st_size:
            raise GitHubCommandError("release asset upload size mismatch")
        upload_url = (
            f"https://uploads.github.com/repos/{repository}/releases/"
            f"{release_id}/assets?{urlencode({'name': asset_name})}"
        )
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "POST",
                "--header",
                "Accept:application/vnd.github+json",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                "--header",
                f"Content-Type:{media_type}",
                "--input",
                str(asset_path),
                upload_url,
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        uploaded = _require_mapping(
            self.runner.run(request).output,
            "uploaded release asset",
        )
        if (
            uploaded.get("name") != asset_name
            or uploaded.get("content_type") != media_type
            or uploaded.get("size") != asset_size
        ):
            raise GitHubCommandError("uploaded release asset metadata mismatch")
        return uploaded

    def delete_release_asset(self, repository: str, asset_id: int) -> None:
        """Delete one verified failed-upload placeholder from an owned draft."""
        if _REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise GitHubCommandError("release asset deletion repository is invalid")
        if type(asset_id) is not int or asset_id < 1:
            raise GitHubCommandError("release asset deletion ID must be positive")
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "DELETE",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                f"repos/{repository}/releases/assets/{asset_id}",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.BINARY,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        result = self.runner.run(request)
        if result.output != b"":
            raise GitHubCommandError("release asset deletion returned a body")

    def verify_release(
        self,
        repository: str,
        tag: str,
        commit_sha: str,
        asset_digests: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._require_release_verifier()
        request = CommandRequest(
            argv=(
                "gh",
                "release",
                "verify",
                tag,
                "--repo",
                repository,
                "--format",
                "json",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        return validate_release_attestation(
            self.runner.run(request).output,
            repository=repository,
            tag=tag,
            commit_sha=commit_sha,
            asset_digests=asset_digests,
        )

    def _require_release_verifier(self) -> None:
        if self.release_verifier_ready:
            return
        request = CommandRequest(
            argv=("gh", "version"),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.TEXT,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        output = self.runner.run(request).output
        if not isinstance(output, str):
            raise GitHubCommandError("GitHub CLI version output must be text")
        match = _CLI_VERSION_PATTERN.match(output)
        if match is None:
            raise GitHubCommandError("GitHub CLI version output is invalid")
        installed = tuple(int(component) for component in match.groups())
        if installed < self.minimum_cli_version:
            required = ".".join(
                str(component) for component in self.minimum_cli_version
            )
            raise GitHubCommandError(
                f"GitHub CLI {required} or newer is required for release verification"
            )
        self.release_verifier_ready = True

    def download_release_asset(
        self,
        repository: str,
        asset_id: str,
        destination: Path,
        maximum_bytes: int,
    ) -> Path:
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "GET",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                "--header",
                "Accept:application/octet-stream",
                f"repos/{repository}/releases/assets/{asset_id}",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.FILE,
            output_path=destination,
            maximum_output_bytes=maximum_bytes,
        )
        output = self.runner.run(request).output
        if not isinstance(output, Path) or output != destination:
            raise GitHubCommandError("GitHub asset download did not produce its path")
        return output

    def read_git_blob(
        self,
        repository: str,
        blob_sha: str,
        maximum_bytes: int,
    ) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{40}", blob_sha):
            raise GitHubCommandError("Git blob SHA must be 40 lowercase hex")
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise GitHubCommandError("Git blob output limit must be positive")
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "GET",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                "--header",
                "Accept:application/vnd.github.raw+json",
                f"repos/{repository}/git/blobs/{blob_sha}",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.BINARY,
            maximum_output_bytes=maximum_bytes,
        )
        output = self.runner.run(request).output
        if not isinstance(output, bytes):
            raise GitHubCommandError("GitHub blob read did not produce bytes")
        return output

    def create_ref_if_absent(
        self, repository: str, qualified_ref: str, commit_sha: str
    ) -> Any:
        require_git_ref_name(
            qualified_ref,
            "qualified GitHub ref",
            qualified=True,
            error_type=GitHubCommandError,
        )
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise GitHubCommandError("GitHub ref commit must be 40 lowercase hex")
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "POST",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                f"repos/{repository}/git/refs",
                "--input",
                "-",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            stdin=canonical_json_bytes({"ref": qualified_ref, "sha": commit_sha}),
            capture_failure=True,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        result = self.runner.run(request)
        if result.returncode != 0:
            current = self.api_json(
                "GET",
                f"repos/{repository}/git/ref/{qualified_ref.removeprefix('refs/')}",
            )
            if not isinstance(current, dict) or not isinstance(
                current.get("object"), dict
            ):
                raise GitHubCommandError("GitHub ref response must contain an object")
            if (
                current.get("ref") == qualified_ref
                and current["object"].get("sha") == commit_sha
            ):
                return current
            raise GitHubCompareAndSwapError(
                "GitHub write-once ref already targets another commit"
            )
        response = result.output
        if not isinstance(response, dict) or not isinstance(
            response.get("object"), dict
        ):
            raise GitHubCommandError("GitHub ref creation response is invalid")
        if (
            response.get("ref") != qualified_ref
            or response["object"].get("sha") != commit_sha
        ):
            raise GitHubCommandError("GitHub ref creation returned another ref")
        return response

    def update_ref_compare_and_swap(
        self,
        repository: str,
        repository_node_id: str,
        branch: str,
        expected_sha: str,
        commit_sha: str,
    ) -> Any:
        if not isinstance(repository_node_id, str) or not repository_node_id:
            raise GitHubCommandError("GitHub repository node ID is required")
        if not re.fullmatch(r"[0-9a-f]{40}", expected_sha) or not re.fullmatch(
            r"[0-9a-f]{40}", commit_sha
        ):
            raise GitHubCommandError("GitHub ref commits must be 40 lowercase hex")
        qualified_ref = f"refs/heads/{branch}"
        require_git_ref_name(
            qualified_ref,
            "GitHub branch ref",
            qualified=True,
            error_type=GitHubCommandError,
        )
        request = CommandRequest(
            argv=(
                "gh",
                "api",
                "--method",
                "POST",
                "--header",
                f"X-GitHub-Api-Version:{self.api_version}",
                "graphql",
                "--input",
                "-",
            ),
            cwd=self.working_directory,
            timeout_seconds=self.timeout_seconds,
            output_kind=CommandOutputKind.JSON,
            stdin=canonical_json_bytes(
                {
                    "query": _UPDATE_REFS_MUTATION,
                    "variables": {
                        "input": {
                            "refUpdates": [
                                {
                                    "afterOid": commit_sha,
                                    "beforeOid": expected_sha,
                                    "force": False,
                                    "name": qualified_ref,
                                }
                            ],
                            "repositoryId": repository_node_id,
                        }
                    },
                }
            ),
            capture_failure=True,
            maximum_output_bytes=self.control_blob_size_bytes,
        )
        result = self.runner.run(request)
        response = result.output
        mutation_succeeded = (
            result.returncode == 0
            and isinstance(response, Mapping)
            and response.get("errors") in (None, [])
            and isinstance(response.get("data"), Mapping)
            and isinstance(response["data"].get("updateRefs"), Mapping)
        )
        current = self.api_json("GET", f"repos/{repository}/git/ref/heads/{branch}")
        if not isinstance(current, dict) or not isinstance(current.get("object"), dict):
            raise GitHubCommandError("GitHub ref response must contain an object")
        current_sha = current["object"].get("sha")
        if current_sha == commit_sha:
            return current
        if current_sha != expected_sha:
            raise GitHubCompareAndSwapError(
                "remote ref no longer has the expected parent"
            )
        if not mutation_succeeded:
            stderr = _redact_diagnostics(
                result.stderr.decode("utf-8", errors="replace")
            )
            raise GitHubCommandError(
                f"GitHub ref update failed with exit {result.returncode}: "
                f"stderr={stderr}"
            )
        raise GitHubCommandError("GitHub ref changed after the atomic update")
