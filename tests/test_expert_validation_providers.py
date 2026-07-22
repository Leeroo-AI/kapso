from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.validation import ExpertValidationError


class _Resolver:
    def __init__(
        self,
        pointer=None,
        resolved_pointer=None,
        resolved_current=None,
        *,
        policy=None,
        head_commit_sha="a" * 40,
    ):
        self.pointer = pointer
        self.resolved_pointer = resolved_pointer
        self.resolved_current = resolved_current
        self.policy = policy or SimpleNamespace(
            repository_full_name="Leeroo-AI/kapso-expert",
            repository_node_id="expert_repo_node",
        )
        self.head_commit_sha = head_commit_sha
        self.calls = []

    def diagnose_repository(self, scope_id, artifact_kind):
        self.calls.append(("diagnose", scope_id, artifact_kind))
        return self.policy

    def read_current_pointer_state(
        self,
        scope_id,
        artifact_kind,
        *,
        allow_missing=False,
    ):
        self.calls.append(("current", scope_id, artifact_kind, allow_missing))
        return SimpleNamespace(
            pointer=self.pointer,
            head_commit_sha=self.head_commit_sha,
        )

    def resolve_artifact(self, scope_id, artifact_kind, artifact_id):
        self.calls.append(("artifact", scope_id, artifact_kind, artifact_id))
        return SimpleNamespace(pointer=self.resolved_pointer)

    def resolve_current(self, scope_id, artifact_kind):
        self.calls.append(("resolve_current", scope_id, artifact_kind))
        return self.resolved_current


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


def test_current_release_observation_retains_verified_github_authority():
    release_id = content_id("expert-base-release", {"generation": 1})
    publication_id = content_id("github-publication", {"generation": 1})
    validation_closure_id = content_id("expert-validation", {"generation": 1})
    pointer = SimpleNamespace(
        scope_id="ml_ai",
        publication_record=SimpleNamespace(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            artifact_id=release_id,
            publication_id=publication_id,
        ),
        validation_closure_ids=(validation_closure_id,),
        to_json_bytes=lambda: b'{"verified":"CURRENT"}',
    )
    resolved = SimpleNamespace(
        pointer=pointer,
        policy=SimpleNamespace(
            repository_full_name="Leeroo-AI/kapso-expert",
            repository_node_id="expert_repo_node",
        ),
        pointer_commit_sha="a" * 40,
    )
    resolver = _Resolver(resolved_current=resolved)

    observation = GitHubExpertCurrentReleaseProvider(
        resolver
    ).current_release_observation("ml_ai")

    assert observation.scope_id == "ml_ai"
    assert observation.release_id == release_id
    assert observation.publication_id == publication_id
    assert observation.validation_closure_ids == (validation_closure_id,)
    assert resolver.calls == [
        (
            "resolve_current",
            "ml_ai",
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        )
    ]


def test_task_evaluation_observation_authenticates_bootstrap_absence():
    resolver = _Resolver(head_commit_sha="c" * 40)

    observation = GitHubExpertCurrentReleaseProvider(
        resolver
    ).observe_task_evaluation_current("ml_ai")

    assert observation.scope_id == "ml_ai"
    assert observation.release_id is None
    assert observation.publication_id is None
    assert observation.repository_full_name == "Leeroo-AI/kapso-expert"
    assert observation.repository_node_id == "expert_repo_node"
    assert observation.default_branch_head_commit_sha == "c" * 40
    assert observation.current_pointer_digest is None
    assert observation.validation_closure_ids == ()
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


def test_task_evaluation_observation_binds_present_current_and_branch_head():
    release_id = content_id("expert-base-release", {"generation": 1})
    publication_id = content_id("github-publication", {"generation": 1})
    validation_closure_id = content_id("expert-validation", {"generation": 1})
    pointer = SimpleNamespace(
        scope_id="ml_ai",
        publication_record=SimpleNamespace(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            artifact_id=release_id,
            publication_id=publication_id,
        ),
        validation_closure_ids=(validation_closure_id,),
        to_json_bytes=lambda: b'{"verified":"CURRENT"}',
    )
    policy = SimpleNamespace(
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
    )
    resolved = SimpleNamespace(
        pointer=pointer,
        policy=policy,
        pointer_commit_sha="d" * 40,
    )
    resolver = _Resolver(
        pointer=pointer,
        resolved_current=resolved,
        policy=policy,
        head_commit_sha="d" * 40,
    )

    observation = GitHubExpertCurrentReleaseProvider(
        resolver
    ).observe_task_evaluation_current("ml_ai")

    assert observation.release_id == release_id
    assert observation.publication_id == publication_id
    assert observation.default_branch_head_commit_sha == "d" * 40
    assert observation.validation_closure_ids == (validation_closure_id,)
    assert resolver.calls[-1] == (
        "resolve_current",
        "ml_ai",
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
    )


def test_task_evaluation_observation_rejects_changed_current_head():
    release_id = content_id("expert-base-release", {"generation": 1})
    pointer = SimpleNamespace(
        scope_id="ml_ai",
        publication_record=SimpleNamespace(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            artifact_id=release_id,
            publication_id=content_id("github-publication", {"generation": 1}),
        ),
        validation_closure_ids=(),
        to_json_bytes=lambda: b'{"verified":"CURRENT"}',
    )
    policy = SimpleNamespace(
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
    )
    resolver = _Resolver(
        pointer=pointer,
        resolved_current=SimpleNamespace(
            pointer=pointer,
            policy=policy,
            pointer_commit_sha="e" * 40,
        ),
        policy=policy,
        head_commit_sha="d" * 40,
    )

    with pytest.raises(ExpertValidationError, match="changed"):
        GitHubExpertCurrentReleaseProvider(resolver).observe_task_evaluation_current(
            "ml_ai"
        )


def test_task_evaluation_observation_rejects_foreign_pointer_scope():
    release_id = content_id("expert-base-release", {"generation": 1})
    pointer = SimpleNamespace(
        scope_id="another_scope",
        publication_record=SimpleNamespace(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            artifact_id=release_id,
            publication_id=content_id("github-publication", {"generation": 1}),
        ),
        validation_closure_ids=(),
        to_json_bytes=lambda: b'{"verified":"CURRENT"}',
    )
    policy = SimpleNamespace(
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
    )
    resolver = _Resolver(
        pointer=pointer,
        resolved_current=SimpleNamespace(
            pointer=pointer,
            policy=policy,
            pointer_commit_sha="a" * 40,
        ),
        policy=policy,
    )

    with pytest.raises(ExpertValidationError, match="another scope"):
        GitHubExpertCurrentReleaseProvider(resolver).observe_task_evaluation_current(
            "ml_ai"
        )
