You are proposing one executable experiment for an evidence-directed search.

Treat the mandatory packet as immutable evidence and instruction data. Follow
the assigned operator and resolved parent exactly. Do not invent identifiers.
Copy `operator_brief.descriptor_target` exactly into `descriptor`. This
descriptor is a control-plane taxonomy, not a place to describe the proposal.
Put proposal-specific detail in `proposal`, `directive_rationale`,
`evaluation_method`, and `expected_observations`.

`evidence_refs` must be a non-empty subset of `allowed_evidence_refs`, copied
exactly. The capacity snapshot is planning context, not evidence. `claim_ids`
and `resolves_claim_ids` may contain only exact claim identifiers from the
evidence snapshot. A contradicted claim may be cited only when the experiment
explicitly tests its resolution and includes it in `resolves_claim_ids`.

The proposal must state one concrete intervention. The evaluation method and
expected observations must make the causal hypothesis falsifiable. Describe
the actual intervention in the narrative fields; do not use novelty adjectives
as evidence. Report the nearest prior idea and experiment when one exists.

Return only the schema-constrained object requested by the caller.

Mandatory packet:

{{MANDATORY_PACKET}}
