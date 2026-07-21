"""Bootstrap and restructure façade for expert repository architecture."""

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


class ExpertRepositoryArchitect:
    """Propose, but never promote, bootstrap or structural candidates."""

    def __init__(self, engine: ExpertCandidateProposalEngine) -> None:
        self.engine = engine

    def propose(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        materialized_parent: MaterializedArtifact | None,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None = None,
        ancestor_candidate_ids: tuple[str, ...] = (),
    ) -> ExpertCandidateProposalResult:
        return self.engine.propose_architecture(
            packet=packet,
            decision=decision,
            materialized_parent=materialized_parent,
            prior_knowledge=prior_knowledge,
            ancestor_candidate_ids=ancestor_candidate_ids,
        )
