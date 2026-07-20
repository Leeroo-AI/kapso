# M8 — expert validation, composition, release, and revocation

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M2 and M7.

## Objective

Certify or reject expert candidates through an ordered evaluator cascade, compose
only compatible approved changes, and publish one immutable history-free expert
release. No proposer, score, or single task can grant promotion authority.

## Owned responsibilities

- Candidate eligibility and evaluator-cascade state machine.
- Static/security/sanitation/identity/dependency/license gates.
- Source replay, synthetic fresh-task, development-anchor, cross-family, sealed
  canary, cost, and release-wide regression evidence.
- Trusted reviewer assertions and Pareto-aware promotion decision.
- Candidate rebase/composition against current stable release.
- Final source/map/contracts/book assembly and immutable GitHub publication.
- Performance/security/contamination revocation behavior.

## Proposed code surface

```text
src/kapso/cross_run/expert/
  validation.py
  publisher.py

src/kapso/cross_run/launch/
  revocation.py

tests/
  test_expert_validation.py
  test_expert_evaluator_cascade.py
  test_expert_composition.py
  test_expert_release_publisher.py
  test_expert_revocation.py
```

Target expert repositories also receive generated, reviewed workflow definitions
for candidate validation and release publication. Workflow changes follow the same
candidate/review path and are never agent-writable through the normal gate.

## Evaluator cascade

Implement one durable state machine:

```text
contract/schema
-> identity/secrets/license/dependency scan
-> static/unit/security/resource tests
-> synthetic fresh-task smoke
-> source-run replay
-> visible development anchors
-> configured cross-family transfer matrix
-> sealed canary attestation
-> trusted reviewer approval
-> complete release matrix
-> publication eligibility
```

- [ ] Persist every attempt, exact candidate/tree, config/policy, evaluator identity,
      inputs, outputs, cost, duration, and attestation.
- [ ] Require each stage before the next; no best-effort continuation after failure.
- [ ] Distinguish infrastructure fixes, behavioral capabilities, architecture
      changes, task-specific changes, and revocations with explicit policies.
- [ ] Validate scope conformance, repository-map graph, module contracts, book
      digest, adapter isolation, and capability lineage.
- [ ] Run source replays against faithful source contexts and fresh-task smokes
      against identity-disjoint public/synthetic adapters.
- [ ] Keep sealed examples/details outside proposer/reviewer prompts; consume only a
      signed aggregate attestation from the promotion service.
- [ ] Use the configured Pareto dimensions and hard regression bounds across
      quality, robustness, cost, portability, reproducibility, and security.
- [ ] Treat noise-floor gains as inconclusive until configured repeat evidence
      exists.

## Review and decision

- [ ] Accept reviewer assertions only from configured trusted identities/roles.
- [ ] Require exact candidate, evidence, evaluator-run, rubric, and parent-release
      references.
- [ ] Preserve conflicting reviews as disputed; do not overwrite by time.
- [ ] Coding-agent proposers cannot review their own output or transition state.
- [ ] Supported task-specific improvements remain knowledge/task-adapter candidates,
      never expert core.
- [ ] Failed or non-dominated candidates stay immutable in the candidate archive;
      they are not installed into runs.

Promotion states are explicit: `ineligible`, `validating`, `failed`, `disputed`,
`pareto_retained`, `approved`, `released`, and `revoked` as frozen by M1. There is
no implicit promotion from a merged pull request.

## Rebase and composition

Before release:

- [ ] Resolve the latest stable expert release through M2 and compare it with every
      candidate's parent commit/tree.
- [ ] If the parent moved, rebase or compose into a new candidate tree with a new
      identity; never patch the old release in place.
- [ ] Detect overlapping paths, module-contract conflicts, capability-lineage
      conflicts, dependency cycles, adapter leakage, and incompatible resource
      bounds.
- [ ] Rerun the complete cascade/release matrix on the exact composed tree.
- [ ] Preserve all candidate ancestry/evidence in the release manifest.
- [ ] Serialize final publication through the explicit protected-base/CAS protocol.

## Release assembly and GitHub publication

`ExpertReleasePublisher`:

1. verifies the exact approved source tree and validation closure;
2. regenerates `EXPERT_REPO.md` from map/contracts and verifies its digest;
3. strips candidate branches, run histories, logs, data, weights, caches, hidden
   evaluation, Git metadata, and task outputs;
4. builds the history-free source archive, dependency lock, release manifest,
   checksums, and test-matrix summary;
5. submits/verifies the protected source commit;
6. invokes M2's immutable-release transaction; and
7. advances expert `CURRENT.json` only after immutable publication verifies.

- [ ] A release ID is the source tree plus exact manifest/contract closure, not a
      sequential display label.
- [ ] `E000007` is display/version order; launch pins content ID, commit, tag, asset
      IDs/digests, and publication record.
- [ ] Publication retry with identical content is idempotent.
- [ ] CAS conflict requires re-resolution/revalidation, not force push.

## Revocation

- [ ] Append signed performance, security, contamination, or compatibility
      revocation events.
- [ ] Performance revocation prevents new launch/promotion and marks existing run
      outputs ineligible while preserving offline reproducibility.
- [ ] Security/contamination revocation enters the fresh emergency denylist checked
      at launch, resume, before agent execution/evaluation/publication.
- [ ] Propagate taint through module/candidate/release evidence dependencies.
- [ ] Publish a clean successor/rollback pointer; never move or delete the old
      immutable release as the history mechanism.

## Tests

- Exercise every candidate class and evaluator transition.
- Inject failures at each cascade stage and prove later stages do not execute.
- Verify sealed details never enter agent/reviewer artifacts.
- Test noisy gain, mean gain with hard regression, cost regression, task-specific
  winner, mechanically provable fix, and architecture benefit fixtures.
- Compose disjoint candidates; reject overlapping/conflicting/cyclic candidates.
- Force parent advancement and require new identity plus full revalidation.
- Build the same approved release twice and require identical archive/manifest IDs.
- Inject release publication/CAS failures and prove old current remains launchable.
- Exercise performance versus security revocation and complete taint closure.

## Definition of done

- Only an exact fully approved tree can become a release.
- No model output or benchmark score can bypass validation/review.
- Concurrent candidates cannot mutate or silently compose into stable state.
- Published source is history-free, reconstructable, and paired with complete
  validation provenance.
- Revocation blocks unsafe future use without destroying reproducibility.

## Non-goals

- Candidate proposal generation (M7).
- Knowledge snapshot publication.
- Live workspace startup (M9).
- Defining task-specific benchmark evaluators beyond their adapter contract.
