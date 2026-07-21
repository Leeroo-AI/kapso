"""Capability-only façade for expert repository generalization."""

from __future__ import annotations

from kapso.cross_run.expert.proposal import (
    ExpertCandidateProposalEngine,
    ExpertCandidateProposalResult,
)
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)
from kapso.cross_run.github.materializer import MaterializedArtifact
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization


class ExpertCapabilityGeneralizer:
    """Propose, but never promote, one capability candidate."""

    def __init__(self, engine: ExpertCandidateProposalEngine) -> None:
        self.engine = engine

    def propose(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        materialized_parent: MaterializedArtifact,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None = None,
        ancestor_candidate_ids: tuple[str, ...] = (),
    ) -> ExpertCandidateProposalResult:
        return self.engine.propose_generalization(
            packet=packet,
            decision=decision,
            materialized_parent=materialized_parent,
            prior_knowledge=prior_knowledge,
            ancestor_candidate_ids=ancestor_candidate_ids,
        )
