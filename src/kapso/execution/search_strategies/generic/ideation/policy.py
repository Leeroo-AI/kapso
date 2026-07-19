"""Pure evidence-to-policy state machine for ideation."""

from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignAction,
    CampaignEvidenceSnapshot,
    EvidenceSignal,
    IdeationCapacityView,
    IdeationMode,
    PolicyDecision,
    PolicyReason,
)


def _reason(
    code: str,
    statement: str,
    snapshot: CampaignEvidenceSnapshot,
    capacity: IdeationCapacityView,
    *references: str,
) -> PolicyReason:
    return PolicyReason(
        code=code,
        statement=statement,
        evidence_refs=(
            snapshot.snapshot_id,
            capacity.capacity_snapshot_id,
            *references,
        ),
    )


def choose_policy(
    snapshot: CampaignEvidenceSnapshot,
    capacity: IdeationCapacityView,
) -> PolicyDecision:
    """Apply the documented precedence without estimating capacity locally."""
    signals = set(snapshot.signals)
    if (
        not capacity.can_start_complete_action
        or not capacity.can_run_granted_evaluation
        or not capacity.preserves_finalization_reserve
    ):
        return PolicyDecision(
            action=CampaignAction.FINALIZE,
            mode=None,
            reasons=(
                _reason(
                    "terminal_capacity",
                    "The capacity authority admits no complete granted evaluation while preserving delivery reserve.",
                    snapshot,
                    capacity,
                ),
            ),
        )
    if EvidenceSignal.RECOVERABLE_TECHNICAL_FAILURE in signals:
        latest_ref = (
            ()
            if snapshot.latest_node_id is None
            else (f"experiment:{snapshot.latest_node_id}",)
        )
        return PolicyDecision(
            action=CampaignAction.RECOVER,
            mode=IdeationMode.RECOVER,
            reasons=(
                _reason(
                    "recover_unfinished_intervention",
                    "The latest selected intervention failed technically before a fair evaluation.",
                    snapshot,
                    capacity,
                    *latest_ref,
                ),
            ),
        )
    if EvidenceSignal.NO_COMPARABLE_EXPERIMENT in signals:
        return PolicyDecision(
            action=CampaignAction.IDEATE,
            mode=IdeationMode.BOOTSTRAP,
            reasons=(
                _reason(
                    "establish_comparable_baseline",
                    "No valid comparable experiment exists.",
                    snapshot,
                    capacity,
                ),
            ),
        )
    verification_signals = (
        EvidenceSignal.FIDELITY_PROMOTION_REQUIRED,
        EvidenceSignal.PROXY_FULL_DIVERGENCE,
        EvidenceSignal.SURPRISING_GAIN,
    )
    active_verification = tuple(
        signal for signal in verification_signals if signal in signals
    )
    if active_verification:
        references = tuple(f"signal:{signal.value}" for signal in active_verification)
        return PolicyDecision(
            action=CampaignAction.IDEATE,
            mode=IdeationMode.VERIFY,
            reasons=(
                _reason(
                    "verify_fragile_evidence",
                    "Decision-critical evidence requires replication or higher fidelity.",
                    snapshot,
                    capacity,
                    *references,
                ),
            ),
        )
    if (
        EvidenceSignal.CREDIBLE_IMPROVEMENT in signals
        and EvidenceSignal.SUPPORTED_LEVER in signals
    ):
        return PolicyDecision(
            action=CampaignAction.IDEATE,
            mode=IdeationMode.EXPLOIT,
            reasons=(
                _reason(
                    "refine_supported_lever",
                    "Comparable evidence shows a credible gain with a supported causal lever.",
                    snapshot,
                    capacity,
                    "signal:credible_improvement",
                    "signal:supported_lever",
                ),
            ),
        )
    exploration_reasons = []
    for signal, statement in (
        (EvidenceSignal.PLATEAU, "Comparable utility has plateaued."),
        (
            EvidenceSignal.CONTRADICTED_LEVER,
            "A causal lever in the active lineage is contradicted.",
        ),
        (EvidenceSignal.DIVERSITY_COLLAPSE, "Recent ideas share one mechanism family."),
        (EvidenceSignal.GAP_DEBT, "An actionable evaluation gap has accumulated debt."),
    ):
        if signal in signals:
            exploration_reasons.append(
                _reason(
                    signal.value,
                    statement,
                    snapshot,
                    capacity,
                    f"signal:{signal.value}",
                )
            )
    if not exploration_reasons:
        exploration_reasons.append(
            _reason(
                "no_supported_lever",
                "No supported causal lever justifies local refinement.",
                snapshot,
                capacity,
            )
        )
    return PolicyDecision(
        action=CampaignAction.IDEATE,
        mode=IdeationMode.EXPLORE,
        reasons=tuple(exploration_reasons),
    )
