# M2 — autonomous GitHub publication and verified materialization

Status: **complete; independently reviewed**

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
- Transport over M1-resolved `ScopeRepositorySettings`; no task-supplied repository
  coordinates or repository-name inference.
- Direct expected-parent commits for validated expert and knowledge artifacts.
- Write-once pre-release intents, exact tag refs, and final artifact identity refs.
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

src/kapso/cross_run/git_refs.py

tests/
  test_cross_run_github_command.py
  test_cross_run_github_publisher.py
  test_cross_run_github_resolver.py
  test_cross_run_materializer.py
```

## Authentication and command boundary

- [x] Define a typed command request/result interface accepting argv arrays,
      explicit cwd, timeout from config, and expected output schema.
- [x] Invoke trusted framework commands without a shell and never build command
      strings from model output.
- [x] Let command failures propagate with complete non-secret diagnostics; do not
      select a fallback transport or retry policy outside configured behavior.
- [x] Bound every stdout/stderr stream and stream source/cache traversal under
      configured entry and byte limits.
- [x] Never read authentication environment variables in Kapso code. `git`, `gh`,
      and their credential stores own credential discovery.
- [x] Allow configured Codex or Claude Code processes to inherit the same Git/`gh`
      authentication and modify either repository autonomously.
- [x] Keep credentials out of `config.yaml`, prompts, invocation artifacts, logs,
      commits, and release assets.

This is an explicit trusted-agent operating model. GitHub credentials are not a
security boundary between the coding agent and the repositories. Artifact
validation, immutable releases, content hashes, and expected-parent commits remain
correctness and reproducibility boundaries.

Tests use a deterministic fake runner. Unit tests never require network access or
the developer's GitHub authentication.
The subprocess runner drains stdout/stderr through parent-side bounded pipes, so
it is safe in threaded evolve processes and does not rely on `preexec_fn`.
An independent completion waiter preserves the configured timeout after both
output pipes close, without selecting on an empty descriptor set or masking the
real exit code.
Command stdin uses anonymous temporary storage; no named command-output or Git-blob
scratch survives a hard crash.

## Direct autonomous publication

`AutonomousGitHubPublisher` accepts a validated publication envelope:

```text
artifact ID and kind
resolved scope repository settings
expected default-branch commit
candidate manifest
validated local tree/assets
publication provenance
```

Tasks:

- [x] Resolve the configured repository coordinate, then pin and verify its live
      immutable node identity in the intent and publication record.
- [x] Require the artifact's `scope_id` to match the resolved registry entry and
      reject expert/knowledge publication to the opposite or an unregistered repo.
- [x] Verify the local tree, manifest, total bytes, symlinks, submodules, and parent
      commit before publication.
- [x] Build a deterministic commit from the exact validated tree.
- [x] Submit the bounded UTF-8 source closure in one nested Git-tree request,
      recompute tree identities in linear time, and reject configurations whose
      worst-case read/write fan-out exceeds the configured GitHub rate budgets.
- [x] Update the default branch only when it still equals the expected parent; no
      force push and no implicit conflict resolution.
- [x] Treat replay of an already-active identical artifact as idempotent; reject
      conflicting bytes and return a typed conflict when an immutable artifact is
      published but lost activation.
- [x] Bind each published artifact ID to a write-once, content-derived identity
      ref so replay remains globally checkable after `CURRENT.json` advances.
- [x] Record repository, commit, parent, actor, artifact ID, and validation closure
      in publication telemetry; never record credentials.

There is no remote candidate branch or pull request. Proposal, automated review,
validation, and admission occur in the local durable state machine. Once eligible,
the same autonomous process commits and publishes it.

## Immutable release transaction

The publisher exposes one artifact-neutral release protocol used by M5 and M8:

1. copy every declared local asset to private staging, validate and unpack those
   staged bytes, and prove the resulting package recreates the exact Git source
   subset before any remote write;
2. create the deterministic source/manifest tree in one bounded request, verify
   its exact Git identity, and commit it by exact GraphQL `updateRefs.beforeOid`
   compare-and-swap;
3. create a write-once `ArtifactPublicationIntent` binding parent, source
   descriptors, Git tree, package descriptor, assets, tag, actor, and validation;
4. create or verify the write-once tag ref at the exact source commit;
5. create/resume a draft release and stream every precomputed asset through the
   raw GitHub upload endpoint with its validated name and media type; a retry may
   delete only a matching empty GitHub `starter` asset from an owned draft;
6. query and verify the exact asset name/media/size/SHA-256 closure;
7. publish under immutable-release controls and verify the DSSE/Sigstore bundle;
8. create `GitHubPublicationRecord` and the final pointer;
9. create a global write-once artifact-identity ref whose pointer digest binds the
   complete pre-release intent;
10. update `CURRENT.json` in a separate exact-parent commit without force.

- [x] Never update `CURRENT.json` before the immutable release verifies.
- [x] A failure before step 10 leaves an inactive source commit or orphan release
      for audit; readers continue using the prior pointer.
- [x] A compare-and-swap conflict returns a typed conflict to the domain publisher,
      which reloads and rebuilds; transport code never guesses how to merge
      scientific data or expert code.
- [x] Never mutate/delete a published release or move its tag; publish a successor.
- [x] Validate GitHub's returned digest against the locally computed digest.
- [x] Pin immutable release ID and asset IDs as well as human-readable tag/name.

The Git source descriptor and materialized package descriptor are intentionally
separate. M5 may add manifest-bound search/index assets that are not Git files, and
M8 may split control metadata, expert source, and test summaries across assets.
Source Git is limited to 512 UTF-8 files/directories and 8 MiB, with a 32 MiB
encoded request ceiling; release assets are limited to 16. The strict settings
contract rejects any override whose worst-case protocol fan-out exceeds the
configured 80 content writes or 900 request points per minute. Each publication
or resolution performs one complete remote source verification; this worst-case
fan-out is the larger of the normal transaction and a retry that deletes and
reuploads every configured asset.
The publisher does not repeat that immutable-commit traversal after the release
verifies. Release attestation package URLs percent-encode the slash-bearing tag,
so the verified PURL and raw Git ref name remain unambiguous representations of
the same release.

## Resolution and materialization

`GitHubArtifactResolver`:

- [x] Accepts only an M1-resolved scope repository pair, never raw repository
      coordinates from a launch request or task adapter.
- [x] Resolves either the default-branch `CURRENT.json` or one global immutable
      artifact-identity ref; inactive CAS losers remain reproducible by ID.
- [x] Validates repository identity, complete intent, publication record, source
      parent, exact globally bounded non-recursive Git tree/file closure and blob
      SHA-256, UTF-8 source bytes, tag, source commit,
      immutable-release state, asset set, digests, and configured publisher.
- [x] Returns immutable locators only; callers cannot request `latest` after
      resolution.
- [x] Downloads assets to a same-filesystem staging directory.
- [x] Accepts exactly one zstd frame and rejects concatenated, skippable, or
      trailing frames before decompression.
- [x] Rejects PAX/GNU extension headers, hidden regular-file members, and all tar
      special files in a bounded physical-header scan, then verifies paths,
      manifest schema, all content hashes, and the expected artifact ID before
      extraction becomes visible. The entry budget charges both physical headers
      and newly created implicit parent directories.
- [x] Flushes and atomically renames to a cache path keyed by content identity.
- [x] Marks the completed cache read-only and writes a verification receipt.
- [x] Reuses a cache entry only when its receipt and complete tree/package digest
      verify.
- [x] Exposes explicit cache inspection/pruning; never removes a pinned entry.

Cache roots reject symlinked ancestors and stream configured entry bounds before
hash traversal. An advisory lease serializes cooperating Kapso cache operations.
Descriptor anchoring prevents symlink-following and preserves object identity;
canonical placement is revalidated after installation and after a pruning rename,
before recursive deletion. Pruning first atomically renames a canonical entry to
a hidden tombstone; a crash can leave reclaimable garbage, never a writable
partial entry at the content-addressed path. Materialization, package-validation,
and cache-reverification staging left by a hard crash, plus pruning tombstones,
are reclaimed under the lease before cache capacity is enforced. Receipts contain
only content identity and package closure, so an
authorized byte-identical repository relocation reuses the cache.

The Kapso OS account is trusted. Cache containment covers untrusted artifact
bytes, pre-existing symlinks/corruption, and cooperating processes that honor the
lease. It cannot sandbox arbitrary code with the same UID, which can rename
user-owned ancestors, modify Kapso, or ptrace the verifier. A deployment that
treats local agent code as hostile must put materialization behind a narrow
separate-UID service with root-owned, agent-read-only cache ancestors.

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
- Reject unregistered repositories, wrong-scope publication records, swapped
  expert/knowledge repositories, and task-supplied location overrides.
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

Current validation candidate: 203 focused M1/M2 tests plus 4 affected post-train
configuration tests, compile checks, Black, and `git diff --check`. Production
GitHub writes are deliberately deferred to M10.
The production host must upgrade `gh` from 2.45 to at least the configured 2.93
before secure release verification can run.

## Definition of done

- M5/M8/M9 can use typed GitHub resolution/publication operations.
- A validated artifact is committed and released without a PR or human action.
- Publication has the documented crash, idempotency, and CAS semantics.
- A release can be resolved and materialized from only its publication record and
  configured repository identity.
- Offline unit tests cover all failure boundaries; real GitHub tests run in M10.

## Non-goals

- Scientific admission or expert promotion decisions.
- Building snapshot or expert release contents.
- Creating, reading, or storing GitHub secrets in Kapso.
- Publishing pre-admission `RunBundle` artifacts; M3 stores them locally and M5
  carries admitted audit closure in the next knowledge release.
- GitHub Apps, pull-request workflows, branch protection, or organization rulesets.
