# M2 — autonomous GitHub publication and verified materialization

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1.

## Objective

Implement the Git/GitHub transport used by autonomous coding agents and trusted
Kapso processes. One externally authenticated GitHub identity has full read/write
authority over the expert and knowledge repositories. Publication writes directly
to the default branch and immutable releases without pull requests, rulesets, or
human approval gates.

## Owned responsibilities

- Git and `gh` command execution behind a fakeable typed runner.
- Repository/ref/release/asset discovery and strict response validation.
- Direct expected-parent commits for validated expert and knowledge artifacts.
- `GitHubPublicationRecord` creation and verification.
- Draft-upload-verify-publish-CAS release protocol.
- Content-addressed, read-only local artifact cache.
- GitHub authentication diagnostics without reading or persisting secrets.
- Minimal repository-policy verification used by production validation.

## Proposed code surface

```text
src/kapso/cross_run/github/
  __init__.py
  command.py
  publisher.py
  resolver.py
  materializer.py

tests/
  test_cross_run_github_command.py
  test_cross_run_github_publisher.py
  test_cross_run_github_resolver.py
  test_cross_run_materializer.py
```

## Authentication and command boundary

- [ ] Define a typed command request/result interface accepting argv arrays,
      explicit cwd, timeout from config, and expected output schema.
- [ ] Invoke trusted framework commands without a shell and never build command
      strings from model output.
- [ ] Let command failures propagate with complete non-secret diagnostics; do not
      select a fallback transport or retry policy outside configured behavior.
- [ ] Never read authentication environment variables in Kapso code. `git`, `gh`,
      and their credential stores own credential discovery.
- [ ] Allow configured Codex or Claude Code processes to inherit the same Git/`gh`
      authentication and modify either repository autonomously.
- [ ] Keep credentials out of `config.yaml`, prompts, invocation artifacts, logs,
      commits, and release assets.

This is an explicit trusted-agent operating model. GitHub credentials are not a
security boundary between the coding agent and the repositories. Artifact
validation, immutable releases, content hashes, and expected-parent commits remain
correctness and reproducibility boundaries.

Tests use a deterministic fake runner. Unit tests never require network access or
the developer's GitHub authentication.

## Direct autonomous publication

`AutonomousGitHubPublisher` accepts a validated publication envelope:

```text
artifact ID and kind
target repository
expected default-branch commit
candidate manifest
validated local tree/assets
publication provenance
```

Tasks:

- [ ] Verify target repository/node identity against config.
- [ ] Verify the local tree, manifest, total bytes, symlinks, submodules, and parent
      commit before publication.
- [ ] Build a deterministic commit from the exact validated tree.
- [ ] Update the default branch only when it still equals the expected parent; no
      force push and no implicit conflict resolution.
- [ ] Treat replay of identical content as idempotent; reject the same artifact ID
      with different bytes.
- [ ] Record repository, commit, parent, actor, artifact ID, and validation closure
      in publication telemetry; never record credentials.

There is no remote candidate branch or pull request. Proposal, automated review,
validation, and admission occur in the local durable state machine. Once eligible,
the same autonomous process commits and publishes it.

## Immutable release transaction

The publisher exposes one artifact-neutral protocol used by M5 and M8:

1. validate the artifact and confirm the default branch has the expected parent;
2. commit the exact source/manifest tree directly to the default branch;
3. create a draft release/tag at that exact commit;
4. upload every precomputed asset and checksum file;
5. query the release and verify asset names, IDs, sizes, and SHA-256 digests;
6. publish the release so immutable-release controls lock the tag/assets;
7. verify the resulting release attestation/provenance;
8. create `GitHubPublicationRecord`;
9. update `CURRENT.json` in a separate expected-parent commit without force.

- [ ] Never update `CURRENT.json` before the immutable release verifies.
- [ ] A failure before step 9 leaves an inactive source commit or orphan release
      for audit; readers continue using the prior pointer.
- [ ] A compare-and-swap conflict returns a typed conflict to the domain publisher,
      which reloads and rebuilds; transport code never guesses how to merge
      scientific data or expert code.
- [ ] Never mutate/delete a published release or move its tag; publish a successor.
- [ ] Validate GitHub's returned digest against the locally computed digest.
- [ ] Pin immutable release ID and asset IDs as well as human-readable tag/name.

## Resolution and materialization

`GitHubArtifactResolver`:

- [ ] Resolves the default-branch head once and reads `CURRENT.json` at that commit.
- [ ] Validates repository identity, publication record, tag, source commit,
      immutable-release state, asset set, digests, and configured publisher.
- [ ] Returns immutable locators only; callers cannot request `latest` after
      resolution.
- [ ] Downloads assets to a same-filesystem staging directory.
- [ ] Verifies archive path safety, manifest schema, all content hashes, and the
      expected artifact ID before extraction becomes visible.
- [ ] Flushes and atomically renames to a cache path keyed by content identity.
- [ ] Marks the completed cache read-only and writes a verification receipt.
- [ ] Reuses a cache entry only when its receipt and complete tree/package digest
      verify.
- [ ] Exposes explicit cache inspection/pruning; never removes a pinned entry.

Partial downloads/extractions have no committed cache marker and are never
returned. Missing or corrupt cache content raises rather than triggering an
unrecorded alternate source.

## Repository policy contract

Production repositories require only:

- private visibility unless an explicit scope policy permits public artifacts;
- one configured autonomous GitHub identity with read/write authority;
- a default branch that permits direct writes by that identity; and
- immutable releases.

Branch protection, tag rulesets, pull requests, required reviewers, and separate
reader/candidate/stable identities are deliberately absent. Automated scientific
admission and expert validation remain application-level state machines, not human
GitHub gates.

M2 supplies a read-only diagnostic that reports repository identity, visibility,
default branch, authenticated actor, write access, and immutable-release status.

## Tests

- Assert exact safe argv for every framework-owned Git/`gh` call.
- Reject invalid repository/ref names, unexpected artifact shapes, symlinks,
  submodules, oversized publications, and stale parents.
- Prove identical replay is idempotent and conflicting replay fails.
- Inject failure after direct commit, draft creation, each asset upload, release
  publication, and before/after pointer commit.
- Prove `CURRENT.json` cannot reference a draft or incomplete release.
- Simulate two publishers from one base and require one typed CAS conflict.
- Reject replaced tags, missing assets, digest mismatch, forged publication
  records, and inconsistent repository identity.
- Inject partial download/extraction and prove no cache entry becomes visible.
- Verify cache reuse, corruption detection, read-only permissions, and pinned-entry
  retention.
- Prove credentials never appear in framework logs, manifests, or artifacts.

## Definition of done

- M3/M5/M8/M9 can use typed GitHub resolution/publication operations.
- A validated artifact is committed and released without a PR or human action.
- Publication has the documented crash, idempotency, and CAS semantics.
- A release can be resolved and materialized from only its publication record and
  configured repository identity.
- Offline unit tests cover all failure boundaries; real GitHub tests run in M10.

## Non-goals

- Scientific admission or expert promotion decisions.
- Building snapshot or expert release contents.
- Creating, reading, or storing GitHub secrets in Kapso.
- GitHub Apps, pull-request workflows, branch protection, or organization rulesets.
