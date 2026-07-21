from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.validation import ExpertValidationError


class _Resolver:
    def __init__(self, pointer=None, resolved_pointer=None):
        self.pointer = pointer
        self.resolved_pointer = resolved_pointer
        self.calls = []

    def diagnose_repository(self, scope_id, artifact_kind):
        self.calls.append(("diagnose", scope_id, artifact_kind))

    def read_current_pointer_state(
        self,
        scope_id,
        artifact_kind,
        *,
        allow_missing=False,
    ):
        self.calls.append(("current", scope_id, artifact_kind, allow_missing))
        return SimpleNamespace(pointer=self.pointer)

    def resolve_artifact(self, scope_id, artifact_kind, artifact_id):
        self.calls.append(("artifact", scope_id, artifact_kind, artifact_id))
        return SimpleNamespace(pointer=self.resolved_pointer)


def _pointer(release_id):
    return SimpleNamespace(publication_record=SimpleNamespace(artifact_id=release_id))


def test_clean_verified_repository_without_current_is_bootstrap():
    resolver = _Resolver()

    release_id = GitHubExpertCurrentReleaseProvider(resolver).current_release_id(
        "ml_ai"
    )

    assert release_id is None
    assert resolver.calls == [
        (
            "diagnose",
            "ml_ai",
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        ),
        (
            "current",
            "ml_ai",
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            True,
        ),
    ]


def test_current_release_is_reopened_through_its_immutable_identity():
    release_id = content_id("expert-release", {"generation": 1})
    pointer = _pointer(release_id)
    resolver = _Resolver(pointer=pointer, resolved_pointer=pointer)

    resolved_id = GitHubExpertCurrentReleaseProvider(resolver).current_release_id(
        "ml_ai"
    )

    assert resolved_id == release_id
    assert resolver.calls[-1] == (
        "artifact",
        "ml_ai",
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
        release_id,
    )


def test_current_release_fails_if_immutable_resolution_switches_pointer():
    release_id = content_id("expert-release", {"generation": 1})
    resolver = _Resolver(
        pointer=_pointer(release_id),
        resolved_pointer=_pointer(content_id("expert-release", {"generation": 2})),
    )

    with pytest.raises(ExpertValidationError, match="observed CURRENT"):
        GitHubExpertCurrentReleaseProvider(resolver).current_release_id("ml_ai")
