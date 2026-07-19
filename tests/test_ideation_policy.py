"""Scenario tests for deterministic policy precedence."""

from kapso.execution.search_strategies.generic.ideation import (
    CampaignAction,
    CampaignEvidenceSnapshot,
    EvidenceSignal,
    IdeationMode,
    ObjectiveDirection,
    content_identifier,
)
from kapso.execution.search_strategies.generic.ideation.policy import choose_policy
from test_ideation_domain import NOW
from test_ideation_evidence import capacity

DIGEST = "6" * 64
SNAPSHOT_ID = content_identifier("evidence_snapshot", DIGEST)


def snapshot(*signals: EvidenceSignal) -> CampaignEvidenceSnapshot:
    return CampaignEvidenceSnapshot(
        snapshot_id=SNAPSHOT_ID,
        campaign_id="campaign-alpha",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        generated_at=NOW,
        content_hash=DIGEST,
        experiments=(),
        claims=(),
        gaps=(),
        relevant_idea_ids=(),
        incumbent_node_id=None,
        latest_node_id=None,
        noise_floor=None,
        signals=signals,
    )


def test_terminal_capacity_precedes_every_ideation_mode():
    decision = choose_policy(
        snapshot(
            EvidenceSignal.RECOVERABLE_TECHNICAL_FAILURE,
            EvidenceSignal.NO_COMPARABLE_EXPERIMENT,
        ),
        capacity(can_start_complete_action=False),
    )
    assert decision.action == CampaignAction.FINALIZE
    assert decision.mode is None
    assert decision.reasons[0].code == "terminal_capacity"


def test_unjustified_opportunity_probe_finalizes_a_banked_campaign():
    decision = choose_policy(
        snapshot(EvidenceSignal.DELIVERY_INCUMBENT),
        capacity(
            opportunity_probe_required=True,
            opportunity_probe_admissible=False,
        ),
    )
    assert decision.action == CampaignAction.FINALIZE


def test_recoverable_failure_precedes_bootstrap_and_explore():
    decision = choose_policy(
        snapshot(
            EvidenceSignal.RECOVERABLE_TECHNICAL_FAILURE,
            EvidenceSignal.NO_COMPARABLE_EXPERIMENT,
            EvidenceSignal.PLATEAU,
        ),
        capacity(),
    )
    assert decision.mode == IdeationMode.RECOVER
    assert decision.action == CampaignAction.RECOVER


def test_cold_start_chooses_bootstrap():
    decision = choose_policy(
        snapshot(EvidenceSignal.NO_COMPARABLE_EXPERIMENT),
        capacity(),
    )
    assert decision.mode == IdeationMode.BOOTSTRAP


def test_proxy_divergence_and_surprising_gain_choose_verify():
    for signal in (
        EvidenceSignal.PROXY_FULL_DIVERGENCE,
        EvidenceSignal.SURPRISING_GAIN,
        EvidenceSignal.FIDELITY_PROMOTION_REQUIRED,
    ):
        decision = choose_policy(snapshot(signal), capacity())
        assert decision.mode == IdeationMode.VERIFY


def test_credible_gain_with_supported_lever_chooses_exploit():
    decision = choose_policy(
        snapshot(
            EvidenceSignal.CREDIBLE_IMPROVEMENT,
            EvidenceSignal.SUPPORTED_LEVER,
        ),
        capacity(),
    )
    assert decision.mode == IdeationMode.EXPLOIT


def test_plateau_or_contradiction_choose_explore_with_auditable_reasons():
    decision = choose_policy(
        snapshot(
            EvidenceSignal.PLATEAU,
            EvidenceSignal.CONTRADICTED_LEVER,
        ),
        capacity(),
    )
    assert decision.mode == IdeationMode.EXPLORE
    assert {reason.code for reason in decision.reasons} == {
        "plateau",
        "contradicted_lever",
    }
    assert all(SNAPSHOT_ID in reason.evidence_refs for reason in decision.reasons)


def test_same_serialized_inputs_produce_identical_decisions():
    evidence = snapshot(
        EvidenceSignal.CREDIBLE_IMPROVEMENT,
        EvidenceSignal.SUPPORTED_LEVER,
    )
    available = capacity()
    assert choose_policy(evidence, available) == choose_policy(evidence, available)
