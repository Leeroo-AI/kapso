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

- [ ] Every command accepts `--config`; repository coordinates and operational
      values come from the config file.
- [ ] Commands print IDs, commits, release tags, asset digests, validation states,
      and next actions—but never secrets or raw sensitive content.
- [ ] Mutation commands require explicit complete artifact/candidate IDs and
      expected parent identities.
- [ ] `inspect`/`verify` are read-only and work against local pins when applicable.
- [ ] Provider diagnostics distinguish missing authentication, insufficient
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
- security-revocation refresh.

For every seam, define the one valid restart action and prove idempotent retry or
typed irrecoverable failure. Never infer success from a partially visible remote
artifact.

## Scenario acceptance

### Domain neutrality

- Run one language-post-training-shaped fixture and one RelBench-shaped relational
  prediction fixture under the same broad scope.
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

1. **Read-only GitHub smoke:** resolve private expert/knowledge EMPTY/E0 releases,
   verify attestations/digests, materialize twice, and prove cache reuse.
2. **Knowledge publication smoke:** publish one sanitized synthetic bundle, run
   automated admission, commit directly, publish immutable S1, and resolve it from
   a clean directory.
3. **Embedding smoke:** build/rebuild S1 search sidecars through the official OpenAI
   embeddings endpoint and verify stable space/input identities.
4. **Ideation CLI smoke:** run one Codex or Claude Code ideation batch with the
   packet-only MCP reader; verify packet/MCP provenance, configured GitHub write
   access, and absence of secret bytes from prompts/artifacts/logs.
5. **Expert bootstrap smoke:** let the configured coding-agent CLI propose E0 in
   the configured expert repository; validate through automated independent roles,
   publish directly, and verify the semantic book/repository map.
6. **Expert successor smoke:** create one mechanically general synthetic fix,
   validate through the configured non-sealed cascade, publish E1, and ensure a new
   launch pins E1 while the old run remains on E0.
7. **Concurrency smoke:** race two knowledge candidates and two expert candidates;
   verify merge/CAS conflict behavior without data loss or force updates.
8. **Revocation smoke:** revoke the synthetic smoke release, refresh the denylist, and
   prove new launch/resume blocking under security policy.
9. **Clean-machine smoke:** with only configured provider authentication and task
   input, resolve/materialize/run from immutable releases without historical run
   directories.

Sealed benchmark/canary testing is a separate explicitly authorized production
stage and must not expose hidden examples to coding-agent processes or GitHub
artifacts.

## Production access checklist

Do not paste long-lived secrets into prompts, commits, config, test output, or this
repository. Before production smokes, the operator supplies authentication through
the provider-owned mechanisms:

### Provisioned GitHub resources

Verified on 2026-07-20:

- owner: `Leeroo-AI`;
- expert repository: `Leeroo-AI/kapso-expert`, private, default branch `main`;
- knowledge repository: `Leeroo-AI/kapso-knowledge`, private, default branch `main`;
- autonomous actor: `leeroo-coder`, authenticated through the external SSH/`gh`
  credential stores with repository administrator authority;
- immutable releases: enabled on both repositories; and
- branch/tag rulesets: none.

No credential value is stored in this repository. M2 preflight must revalidate
this external state rather than trusting the planning-time observation.

### Required

- GitHub organization/owner and the two private repository names.
- One authenticated Git/`gh` profile with full read/write authority for both
  repositories, including direct commits, tags, releases, assets, and workflows.
- Direct default-branch writes enabled with no pull-request, reviewer, ruleset, or
  human approval requirement.
- Immutable releases enabled on both repositories.
- An authenticated supported coding-agent CLI: Codex login/profile or Claude Code
  login/provider configuration. Only the CLI selected in `config.yaml` is required.
- OpenAI embeddings authentication available to the trusted parent process through
  official SDK credential discovery; it must be absent from coding-agent/MCP child
  environments.

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

After the credentialed path passes:

- [ ] Delete old IdeaArchive/Generic checkpoint readers and fixtures.
- [ ] Delete active `initial_repo`/starter-selection cloning and config/docs.
- [ ] Delete any prototype merged cross-run experiment store.
- [ ] Delete duplicate embedding providers and old imports.
- [ ] Delete any prototype GitHub App, candidate-PR, or human approval path; retain
      only the configured external Git/`gh` credential discovery.
- [ ] Delete fallback retrieval/publication paths and legacy aliases.
- [ ] Run the complete suite after deletion and verify repository search finds no
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
