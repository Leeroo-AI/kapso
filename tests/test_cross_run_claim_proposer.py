import json
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.catalog.claims import (
    ClaimProposalError,
    ClaimProposalPacket,
    ClaimProposer,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    CodingAgentOperationReceipt,
    ExpertScopeContract,
    KnowledgeClaim,
    PriorIdea,
    ReviewAssertion,
    TransferEpisode,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_RETURNED_ARTIFACT_FILENAMES as ARTIFACT_FILENAMES,
)
from kapso.execution.coding_agents.structured_call import CodingAgentCallResult
from test_cross_run_contracts import build_records, fixture_id

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class ArtifactFakeRunner:
    def __init__(self, artifact_root: Path, output):
        self.artifact_root = artifact_root
        self.output = output
        self.requests = []
        self.schemas = []

    def run(self, request, response_schema):
        self.requests.append(request)
        self.schemas.append(response_schema)
        operation = self.artifact_root / request.operation_id
        operation.mkdir(parents=True, exist_ok=True)
        output_text = json.dumps(self.output, sort_keys=True)
        contents = {
            "prompt.txt": request.prompt,
            "response_schema.json": json.dumps(response_schema, sort_keys=True),
            "invocation.json": json.dumps(request.to_dict(), sort_keys=True),
            "prior_knowledge.json": "null\n",
            "mcp_config.json": "{}\n",
            "stdout.txt": "structured event stream\n",
            "stderr.txt": "",
            "final.json": output_text,
            "mcp_audit.jsonl": "",
        }
        for filename in ARTIFACT_FILENAMES:
            (operation / filename).write_text(contents[filename], encoding="utf-8")
        result = CodingAgentCallResult(
            output=output_text,
            duration_seconds=1.0,
            cost_usd=None,
            input_tokens=100,
            output_tokens=50,
            artifacts=tuple(str(operation / name) for name in ARTIFACT_FILENAMES),
        )
        (operation / "result.json").write_text(
            json.dumps(result.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result


class FinalArtifactTamperingRunner(ArtifactFakeRunner):
    def run(self, request, response_schema):
        result = super().run(request, response_schema)
        final_path = next(
            Path(path) for path in result.artifacts if Path(path).name == "final.json"
        )
        final_path.write_text('{"claims": []}', encoding="utf-8")
        return result


class FinalArtifactWhitespaceRunner(ArtifactFakeRunner):
    def run(self, request, response_schema):
        result = super().run(request, response_schema)
        final_path = next(
            Path(path) for path in result.artifacts if Path(path).name == "final.json"
        )
        final_path.write_text(result.output + "\n", encoding="utf-8")
        return result


def catalog_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).catalog


def packet():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    episodes = tuple(
        sorted(
            (record for record in records if isinstance(record, TransferEpisode)),
            key=lambda record: record.episode_id,
        )
    )
    prior_ideas = tuple(
        sorted(
            (record for record in records if isinstance(record, PriorIdea)),
            key=lambda record: record.prior_idea_id,
        )
    )
    claims = tuple(
        sorted(
            (record for record in records if isinstance(record, KnowledgeClaim)),
            key=lambda record: record.revision_id,
        )
    )
    states = tuple(
        sorted(
            (record for record in records if isinstance(record, CatalogEntryState)),
            key=lambda record: record.catalog_entry_state_id,
        )
    )
    assertions = tuple(
        sorted(
            (
                record
                for record in records
                if isinstance(record, ReviewAssertion)
                and record.subject_id in {episode.episode_id for episode in episodes}
            ),
            key=lambda record: record.assertion_id,
        )
    )
    receipt_ids = {assertion.review_operation_ref for assertion in assertions}
    receipts = tuple(
        sorted(
            (
                record
                for record in records
                if isinstance(record, CodingAgentOperationReceipt)
                and record.operation_receipt_id in receipt_ids
            ),
            key=lambda record: record.operation_receipt_id,
        )
    )
    proof_ids = {
        reference
        for episode in episodes
        for reference in (
            episode.source_bundle_id,
            episode.sanitation_report_id,
            *episode.derivation_refs,
        )
    }
    proof_ids.update(
        reference
        for prior_idea in prior_ideas
        for reference in (
            prior_idea.source_bundle_id,
            prior_idea.sanitation_report_id,
        )
    )
    proof_ids.update(
        reference
        for assertion in assertions
        for reference in assertion.exact_evidence_refs
    )
    return ClaimProposalPacket(
        catalog_generation_id=fixture_id("catalog-generation"),
        catalog_generation=1,
        scope_contract=scope,
        episodes=episodes,
        prior_ideas=prior_ideas,
        existing_claims=claims,
        entry_states=states,
        review_assertions=assertions,
        operation_receipts=receipts,
        proof_reference_ids=tuple(sorted(proof_ids)),
    )


def valid_output(proposal_packet):
    episode_id = proposal_packet.episodes[0].episode_id
    return {
        "claims": [
            {
                "statement": "Parity checks improve reliable training setup.",
                "mechanism": "They expose representation mismatches before fitting.",
                "applicability_predicates": {"dataset_family": "instruction"},
                "explicit_exclusions": ["Unmeasured baseline-only runs."],
                "evidence_assessments": [
                    {
                        "episode_id": episode_id,
                        "relationship": "support",
                        "rationale": "The isolated comparable attempt improved quality.",
                    }
                ],
                "supersedes_revision_ids": [],
            }
        ]
    }


def test_claim_proposer_uses_complete_packet_and_framework_owned_identity(tmp_path):
    proposal_packet = packet()
    output = valid_output(proposal_packet)
    runner = ArtifactFakeRunner(tmp_path / "agent-artifacts", output)
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()

    result = ClaimProposer(catalog_settings(), runner).propose(
        proposal_packet,
        workspace,
    )

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.supporting_episode_ids == (proposal_packet.episodes[0].episode_id,)
    assert claim.contradicting_episode_ids == ()
    assert len(result.claim_evidence_closures) == 1
    closure = result.claim_evidence_closures[0]
    assert closure.claim_revision_id == claim.revision_id
    assert closure.evaluated_episode_ids == tuple(
        episode.episode_id for episode in proposal_packet.episodes
    )
    assert closure.supporting_episode_ids == claim.supporting_episode_ids
    assert closure.proposer_operation_receipt_id == (
        result.operation_receipt.operation_receipt_id
    )
    assert claim.claim_id.startswith("claim_")
    assert set(claim.proposal_provenance) == {
        "operation_receipt_id",
        "packet_digest",
        "proposal_ordinal",
    }
    assert set(result.operation_receipt.artifact_checksums) == {
        *ARTIFACT_FILENAMES,
        "result.json",
    }
    request = runner.requests[0]
    assert request.cli == "codex"
    assert request.model == "gpt-5.6-sol"
    assert request.allowed_tools == ()
    assert proposal_packet.episodes[0].proposal in request.prompt
    assert proposal_packet.prior_ideas[0].proposal in request.prompt
    assert tuple(workspace.iterdir()) == ()
    assert (
        "claim_id"
        not in runner.schemas[0]["properties"]["claims"]["items"]["properties"]
    )


def test_exact_cached_result_mints_the_same_receipt_and_claim(tmp_path):
    proposal_packet = packet()
    runner = ArtifactFakeRunner(
        tmp_path / "agent-artifacts",
        valid_output(proposal_packet),
    )
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()
    proposer = ClaimProposer(catalog_settings(), runner)

    first = proposer.propose(proposal_packet, workspace)
    second = proposer.propose(proposal_packet, workspace)

    assert first.operation_receipt == second.operation_receipt
    assert first.claims == second.claims
    assert runner.requests[0].operation_id == runner.requests[1].operation_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output, proposal_packet: output["claims"][0].__setitem__(
            "claim_id", "model-owned-id"
        ),
        lambda output, proposal_packet: output["claims"][0].__setitem__(
            "explicit_exclusions", []
        ),
        lambda output, proposal_packet: output["claims"][0].__setitem__(
            "applicability_predicates", {"model_name": "forbidden"}
        ),
        lambda output, proposal_packet: output["claims"][0].__setitem__(
            "evidence_assessments", []
        ),
        lambda output, proposal_packet: output["claims"][0]["evidence_assessments"][
            0
        ].__setitem__("episode_id", proposal_packet.prior_ideas[0].prior_idea_id),
        lambda output, proposal_packet: output["claims"][0].__setitem__(
            "supersedes_revision_ids", [fixture_id("unknown-claim-revision")]
        ),
    ],
)
def test_claim_proposer_rejects_model_owned_or_incomplete_semantics(
    tmp_path,
    mutate,
):
    proposal_packet = packet()
    output = valid_output(proposal_packet)
    mutate(output, proposal_packet)
    runner = ArtifactFakeRunner(tmp_path / "agent-artifacts", output)
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()

    with pytest.raises((ClaimProposalError, ValueError)):
        ClaimProposer(catalog_settings(), runner).propose(proposal_packet, workspace)


def test_claim_packet_fails_instead_of_truncating_complete_records(tmp_path):
    proposal_packet = packet()
    settings = replace(catalog_settings(), claim_packet_record_limit=1)
    runner = ArtifactFakeRunner(
        tmp_path / "agent-artifacts",
        valid_output(proposal_packet),
    )
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()

    with pytest.raises(ClaimProposalError, match="complete-record"):
        ClaimProposer(settings, runner).propose(proposal_packet, workspace)
    assert runner.requests == []


def test_claim_proposer_rejects_nonempty_or_relative_workspace(tmp_path):
    proposal_packet = packet()
    runner = ArtifactFakeRunner(
        tmp_path / "agent-artifacts",
        valid_output(proposal_packet),
    )
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        ClaimProposer(catalog_settings(), runner).propose(proposal_packet, nonempty)
    with pytest.raises(ValueError, match="absolute"):
        ClaimProposer(catalog_settings(), runner).propose(
            proposal_packet,
            Path("relative-workspace"),
        )


def test_claim_proposer_rejects_artifact_result_divergence(tmp_path):
    proposal_packet = packet()
    runner = FinalArtifactTamperingRunner(
        tmp_path / "agent-artifacts",
        valid_output(proposal_packet),
    )
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="does not match"):
        ClaimProposer(catalog_settings(), runner).propose(proposal_packet, workspace)


def test_operation_record_authenticates_exact_final_artifact_bytes(tmp_path):
    proposal_packet = packet()
    runner = FinalArtifactWhitespaceRunner(
        tmp_path / "agent-artifacts",
        valid_output(proposal_packet),
    )
    workspace = tmp_path / "sanitized-workspace"
    workspace.mkdir()

    result = ClaimProposer(catalog_settings(), runner).propose(
        proposal_packet,
        workspace,
    )

    assert result.operation_record.final_output.endswith("\n")
    result.operation_record.validate_receipt(result.operation_receipt)
