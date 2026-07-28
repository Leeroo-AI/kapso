# M10 — system rollout, operations, and production validation

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M3–M9.

## Objective

Prove the complete system under failure and real provider boundaries, expose an
operable CLI/workflow, delete every superseded path, and activate GitHub-backed
cross-run learning as the sole supported evolve startup/publication design.

## Owned responsibilities

- Operational commands and status/diagnostic output.
- Minimal autonomous GitHub repository/authentication setup and validation.
- Cross-module deterministic, failure-injection, and scenario-replay suites.
- Credentialed Codex/Claude/OpenAI/GitHub production tests.
- Clean-machine and offline-resume tests.
- Metrics/telemetry, rollout gates, and incident/revocation runbooks.
- Final deletion of old schemas, startup paths, config, prompts, fixtures, and docs.

## Proposed code/document surface

```text
src/kapso/
  cli.py

tests/cross_run/
  test_system_e2e.py
  test_failure_injection.py
  test_concurrent_publication.py
  test_scenario_replay.py

tests/production/
  test_github_cross_run_smoke.py
  test_cross_run_coding_agents_smoke.py
  test_cross_run_openai_embeddings_smoke.py
  test_clean_machine_launch_smoke.py

docs/cross-run/
  operations.mdx
  github-setup.mdx
  security-and-revocation.mdx
  troubleshooting.mdx
```

## Operational command surface

Exact command names may be refined before implementation, but responsibilities are
fixed and no command bypasses module validation:

```text
kapso cross-run inspect
kapso cross-run capture
kapso cross-run publish-knowledge
kapso cross-run propose-expert
kapso cross-run validate-expert
kapso cross-run publish-expert
kapso cross-run resolve-launch
kapso cross-run verify
kapso cross-run revoke
```

- [x] Every command accepts `--config`; repository coordinates and operational
      values come from the config file.
- [x] Commands print IDs, commits, release tags, asset digests, validation states,
      and next actions—but never secrets or raw sensitive content.
- [x] Mutation commands require explicit complete artifact/candidate IDs and
      expected parent identities.
- [x] `inspect`/`verify` are read-only and work against local pins when applicable.
- [x] Provider diagnostics distinguish missing authentication, insufficient
      permission, stale-parent conflict, network failure, schema corruption, and
      digest failure without fallback.

## Deterministic integration suite

Build typed fixtures for:

- empty scope -> E0 + EMPTY -> first launch;
- one completed run -> B1 -> episodes/prior ideas -> S1;
- stopped/crashed run harvested at last reconciled frontier;
- S1 retrieved into a new task's ideation packet;
- repeated general mechanism -> expert candidate -> E1;
- new relational-prediction task family -> architecture candidate -> E2;
- contradiction narrows a claim without deleting prior evidence;
- concurrent knowledge publishers and concurrent expert candidates;
- late contamination causing claim/candidate/release taint and revocation; and
- clean successor release plus rollback/reproducibility of older runs.

- [ ] Use deterministic fake coding-agent, embedding, GitHub, automated-reviewer,
      evaluator, and sealed-service boundaries for the primary CI suite.
- [ ] Assert complete artifacts and side effects, not mock call counts alone.
- [ ] Verify content IDs/bytes under input reordering and independent process runs.
- [ ] Validate all fail-loud paths and absence of partial durable state.

## Failure-injection matrix

Inject process failure immediately before/after:

- execution journal append;
- checkpoint/archive/store/capture commits;
- quarantine sanitation and bundle publication;
- catalog generation and claim proposal artifact persistence;
- snapshot/index packaging;
- expected-parent direct commit;
- GitHub draft creation and each release-asset upload;
- immutable release publication and `CURRENT.json` CAS;
- expert workspace extraction/rename/bootstrap pin;
- ideation prior packet creation/generation/selection; and
- security-revocation refresh;
- retained run-action cleanup before/after Docker-daemon and host restart; and
- replacement/corruption of the pre-provisioned Docker mutation-lock inode.

For every seam, define the one valid restart action and prove idempotent retry or
typed irrecoverable failure. Never infer success from a partially visible remote
artifact.

### Implemented coverage ledger

The system scenario is `tests/test_cross_run_system_e2e.py`. It composes the
sealed M3–M9 artifacts once for both PostTrain-shaped and RelBench-shaped task
contexts: EMPTY/E0 launch, captured experiment, catalog generation, S1 publication
and retrieval, validated E1 candidate, later-task selection, and old-run resume.
The successor assertions are exact: the later launch pins E1 and S1, replays the
published E1 source bytes, and the original run remains pinned to E0 and EMPTY.
The lower-level failure seams stay owned by their focused tests:

| System seam | Existing evidence |
|---|---|
| Interrupted GitHub publication and retry | `test_cross_run_github_publisher.py::test_publication_failure_never_activates_current_early`, `::test_retry_resumes_partially_uploaded_draft_without_duplicate_asset`, and `::test_post_cas_witness_failure_leaves_recoverable_current` |
| Knowledge `CURRENT` CAS conflict | `test_knowledge_snapshot_publisher.py::test_m2_compare_and_swap_failure_propagates_without_fallback` |
| Concurrent knowledge/expert activation | `test_cross_run_github_publisher.py::test_two_publishers_from_one_head_produce_typed_compare_and_swap_conflict` and the sealed expert authorization tests in that module |
| Clean-directory materialization | `test_cross_run_materializer.py::test_materializer_atomically_commits_read_only_cache_and_reuses_it` and `test_launch_workspace.py::test_builder_materializes_private_read_only_copies` |
| Cross-module old-run resume | `test_cross_run_system_e2e.py::test_empty_launch_to_s1_e1_later_task_and_old_resume` and `test_launch_handoff.py::test_resume_handoff_maps_the_refreshed_checkpoint_head` |
| Daemon/host restart | `test_run_action_recovery.py::test_provider_termination_has_exact_crash_restart_semantics`, `::test_result_received_recovers_after_full_runtime_restart`, and `::test_result_decided_recovers_after_full_runtime_restart_without_implementation` |
| Final legacy cutover | `test_launch_workspace.py::test_published_envelope_rejects_legacy_run_action_lock`, `test_run_action_recovery.py::test_legacy_direct_spawn_interfaces_are_removed`, plus the final repository search and complete-suite gate |

This ledger is deliberately referential: a seam gets a new M10 test only when no
focused test already proves its restart or conflict contract.

## Scenario acceptance

### Domain neutrality

- Run one language-post-training-shaped fixture and one RelBench-shaped relational
  prediction fixture under the same broad scope.
- Prove their typed bindings resolve through `ml_ai` to the same configured expert
  and knowledge repositories without either benchmark config containing repo names.
- Prove core schemas/index/filter code contain no task-family-specific conditionals.
- Prove raw measurements do not compare across evaluator fingerprints while
  analogical records may still inspire a local idea.
- Prove the architect may restructure expert topology without changing framework
  folder assumptions.

### Knowledge value

- Compare matched new-task runs with explicit EMPTY versus pinned prior snapshot.
- Record cost/time to first valid evaluation, repeated-failure avoidance, retrieved
  record use, contradictions, and prompt/search overhead.
- Do not claim benefit without matched budgets and repeat evidence.

### Expert value

- Compare matched runs from parent versus promoted expert release.
- Record capability activation, transfer success, regressions, portability,
  reproducibility, and maintenance cost.
- A release that fails configured transfer/cost/robustness gates is not activated.

## Real production test sequence

Production tests are manually/explicitly enabled and never run in normal CI:

1. **Read-only GitHub smoke:** resolve private expert/knowledge EMPTY/E0 releases
   and the security generation-zero release, verify attestations/digests,
   materialize twice, and prove cache reuse plus live denylist refresh.
2. **Knowledge publication smoke:** publish one sanitized synthetic bundle, run
   automated admission, commit directly, publish immutable S1, and resolve it from
   a clean directory.
3. **Embedding smoke:** build/rebuild S1 search sidecars through the official OpenAI
   embeddings endpoint and verify stable space/input identities.
4. **Ideation CLI smoke:** run one Codex or Claude Code ideation batch with the
   packet-only MCP reader; verify packet/MCP provenance, configured GitHub write
   credentials are absent, raw Docker-socket and mutation-lock access are denied,
   and all secret bytes are absent from prompts/artifacts/logs.
   For Claude, first verify the exact installed build accepts the generated
   sandbox settings, has `bubblewrap` and `socat`, authenticates through the
   configured OAuth/provider mechanism, can read only the workspace and packet,
   and is denied `.env`, `/proc`, and provider/GitHub credential-store canaries.
   A response from an unknown-setting probe is a hard activation failure because
   print mode can silently ignore invalid settings.
5. **Expert bootstrap smoke:** let the configured coding-agent CLI propose E0 in
   the configured expert repository; validate through automated independent roles,
   publish directly, and verify the semantic book/repository map.
6. **Expert successor smoke:** create one mechanically general synthetic fix,
   validate through the configured non-sealed cascade, publish E1, and ensure a new
   launch pins E1 while the old run remains on E0.
7. **Concurrency smoke:** race two knowledge candidates and two expert candidates;
   verify merge/CAS conflict behavior without data loss or force updates.
8. **Clean-machine smoke:** with only configured provider authentication and task
   input, resolve/materialize/run from immutable releases without historical run
   directories.
9. **Docker authority smoke:** pre-provision the root-owned mutation lock, prove
    ordinary mutations serialize while closed containment is not delayed, prove
    coding agents and workloads cannot resolve the lock or socket, and prove the
    typed lost-installation path converges after daemon and host restart without
    guessing or human cleanup.
10. **Revocation smoke:** publish security generations zero and one through the
    focused lineage gate, revoke the synthetic smoke release, prove new
    launch/resume blocking, then prove rollback/fork rejection from persisted local
    state. Run this last so the revoked release cannot invalidate clean-launch and
    restart evidence.

Sealed benchmark/canary testing is a separate explicitly authorized production
stage and must not expose hidden examples to coding-agent processes or GitHub
artifacts.

### Production checkpoint (2026-07-27)

The credentialed smoke has passed GitHub authority bootstrap/read, knowledge S1
publication and clean retrieval, OpenAI embedding construction, Codex packet-only
ideation, task-adapter publication, expert E0 proposal, and exact E0 validation
enrollment. Enrollment deliberately stops at the `contract_schema` evaluator
transition: it proves the proposed candidate is the one durably entering the
validation state machine, but it does not forge an evaluator decision.

Production has therefore **not** passed signed generic evaluator validation, E0
publication, E1 proposal/publication, an E1/S1 production launch, concurrency,
revocation, clean-machine execution, or live daemon/host restart. Those stages
remain blocked by external production authorities described below. A sealed
canary is optional for E0 and mechanically classified E1 validation and is not the
current blocker.

All of those steps are now explicit selectable stages in the same durable smoke
receipt. Existing operational services perform validation advancement,
publication, launch, and revocation. S1 now carries one admitted lineage-tracked
transport `TransferEpisode`; the E1 proposal stage performs a real read-only Codex
inspection of E0 and seals an episode/path/capability-bound
`mechanically_general_fix` observation before invoking the normal proposer. A
missing signed evaluator result, second eligible concurrency child, pinned image,
or external restart controller stops at that exact boundary and writes no passing
stage receipt.

The final implementation gate executed the complete repository suite after
legacy deletion and the reviewer-requested E1 orchestration repair in
`70590a1c`: 4,015 passed, 25 skipped, and no tests failed in 2:04:50. The skips
are explicit optional/manual-provider boundaries;
production is never reported as passing through them. All 61 surviving Python
files changed since the M9 review boundary pass Black; the package, tests, and
benchmarks compile; the repository diff check is clean; and the superseded
schema/config/prompt-name search contains only documentation of their deletion.
Before that complete-suite gate, the E1 repair passed 81 affected
knowledge/expert/operations/system tests, including an actual stage-chain test
from an admitted snapshot episode through deterministic trigger selection and
the successor proposal operation.

## Production access checklist

Do not paste long-lived secrets into prompts, commits, config, test output, or this
repository. Before production smokes, the operator supplies authentication through
the provider-owned mechanisms:

### Provisioned GitHub resources

Verified on 2026-07-20:

- owner: `Leeroo-AI`;
- scope registry entry: `ml_ai`;
- expert repository: `Leeroo-AI/kapso-expert`, private, default branch `main`;
- knowledge repository: `Leeroo-AI/kapso-knowledge`, private, default branch `main`;
- security repository: `Leeroo-AI/kapso-security`, private, default branch `main`;
- autonomous actor: `leeroo-coder`, authenticated through the external SSH/`gh`
  credential stores with repository administrator authority;
- immutable releases: must be enabled and revalidated on all three repositories;
- branch/tag rulesets: none.

No credential value is stored in this repository. M2 preflight must revalidate
this external state rather than trusting the planning-time observation.

### Required

- Canonical `cross_run.scopes.ml_ai` mapping to the three provisioned repositories;
  benchmark configs contain only their scope/family/adapter bindings.
- GitHub organization/owner and the three private repository names.
- One authenticated Git/`gh` profile with full read/write authority for all three
  repositories, including direct commits, tags, releases, assets, and workflows.
- Direct default-branch writes enabled with no pull-request, reviewer, ruleset, or
  human approval requirement.
- Immutable releases enabled on all three repositories.
- An authenticated supported coding-agent CLI: Codex login/profile or Claude Code
  login/provider configuration. Only the CLI selected in `config.yaml` is required.
- For Claude activation, `bubblewrap` and `socat`, plus a successful authenticated
  allow-read/deny-read/MCP preflight against the exact installed CLI version.
- OpenAI embeddings authentication available to the trusted parent process through
  official SDK credential discovery; it must be absent from coding-agent/MCP child
  environments.
- Private GHCR package access with `write:packages` (and its implied read access),
  plus the content-pinned `ghcr.io/leeroo-ai/kapso-coding-agent` manifest and
  inspected image-config digest recorded in `config.yaml`.
- A root-provisioned Docker mutation lock matching the configured path, owned by
  `root:docker`, mode `0640`, in a root-owned directory with no group/world write;
  application code must not create or repair it.

### Required for full task evaluation, not transport smoke

- Task-adapter repository/artifact access and any public task data/runtime required
  by the chosen RelBench or language-post-training smoke.
- Compute runner credentials/capacity for configured training/evaluation jobs.
- Configured autonomous reviewer/evaluator identities or services for claim and
  expert admission; they execute without human intervention.

### Optional until sealed promotion testing

- Sealed-canary service endpoint and short-lived client identity.
- Hidden evaluator/data credentials held only by that service.
- Signing/trust-root material beyond GitHub's release attestation if organization
  policy requires a second signer.

The implementation must include a preflight command that reports which capabilities
are present without printing secret values.

## Legacy deletion

Completed before the final complete-suite gate:

- [x] Delete old IdeaArchive/Generic checkpoint readers and fixtures; retain only
      canonical IdeaArchive v4, GenericSearch v5, run-checkpoint v2, and
      ExperimentHistoryStore v5 authorities.
- [x] Delete active `initial_repo`/starter-selection cloning and config/docs.
- [x] Delete any prototype merged cross-run experiment store.
- [x] Delete duplicate embedding providers and old imports.
- [x] Delete any prototype GitHub App, candidate-PR, or human approval path; retain
      only the configured external Git/`gh` credential discovery.
- [x] Delete fallback retrieval/publication paths and legacy aliases.
- [x] Delete or move every Docker-SDK mutator outside the pinned cross-run daemon;
      repository search must show no shared-socket mutation bypass.
- [x] Run the complete suite after deletion and verify repository search finds no
      superseded schema/config/prompt names.

## Definition of done

- All deterministic and failure-injection suites pass.
- All required real-provider smokes pass in the configured private repositories.
- Provider credential bytes are absent from prompts, artifacts, config, and logs.
- Autonomous direct GitHub publication and immutable releases enforce the
  documented operating model.
- Matched domain-neutral scenarios produce auditable evidence without overclaiming
  transfer benefit.
- The GitHub-backed launch/publication path is the only supported path.
- Operational/security/revocation/troubleshooting docs are complete.

## Non-goals

- Automatically creating production secrets or organization policy.
- Using real hidden benchmark data in ordinary CI.
- Declaring cross-run benefit from one unreplicated smoke.
- Retaining old behavior as a rollback mechanism; immutable old artifacts provide
  reproducibility, while code tracks only the new design.
