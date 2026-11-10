---
name: warehouse-release
description: "The warehouse-release deployment target: releases what is merged on the repository's base branch into the production warehouse through the platform's own release command, rehearsed first, audited with every released ticket's checks before anything is published, checked again on production afterwards, and rolled back, with production compared to its state before the release, by the target itself on any failure and by the deployment's stop. Use when the task asks for the fixes to be applied or released in production after their pull requests are merged, and the task statement names a release command; never to write to production directly."
metadata:
  run_kinds: [script]
---
# Warehouse release

A build session reads production and never writes it. What reaches production is what a
person merged, built by the platform, and this target is the audit around that: it
decides from the tickets' own checks whether a rehearsed release is published and whether
a published one holds.

## What the deployment session has

The solution's repository at its delivered commit in the workspace, with the proofs and
the evidence the build wrote (results/proofs/<ticket>.json, results/<ticket>/evidence.json
and the specs it names), prepared/environment.json with `release_command` (the platform's
command), `rehearsal_target` (the dbt target that reads the database a release is
rehearsed in) and `reference_target` (the target that reads production),
`KAPSO_SOURCE_REMOTE` (the repository the released commit is fetched from), and the
options `KAPSO_OPTION_COMMIT` (the commit on the base branch to release; its tip when
unset), `KAPSO_OPTION_REQUEST_ARGUMENTS` (extra arguments for the release request, as
the repository's documentation gives them, for instance to build a dataset whose own
file did not change) and `KAPSO_OPTION_ALLOW_UNAUDITED` (`yes` lets merged changes no
ticket of this campaign covers ride along; they are then named in the record).

## What the deadlines must cover

A release runs for the length of its audits: every released ticket's checks on the
rehearsal, then again on production once published, each check a query against a
warehouse. On a backlog that is tens of minutes, past the engine's packaged thirty for a
deployment session and ten for a lifecycle script, and a start cut short leaves a
rehearsal staged and nothing published, or, cut during the publication, a record at
`publishing` for the stop to settle. The repository's Kapso config sets
`deployment.timeout` and `deployment.script_timeout` (seconds) to cover the slowest
release it expects: the ticket-check service's own timing, measured by the build's
proofs, is the estimate, with room for the publication and the production checks on top.

## The release command's contract

Platforms release in different ways: a job a merge triggers, a scheduler's deploy, a
command a release manager runs. This target drives one small command-line contract, and
the repository's documentation says which command speaks it. Where the platform's own
interface differs, a wrapper script in the repository absorbs the difference; this
skill's scripts are never edited for one platform.

| Verb | What the platform does | The JSON record it prints |
| --- | --- | --- |
| `request [--commit C] --json` | rehearses a release of what the base branch holds beyond production, somewhere production is not touched | `release` (the id), `commit`, `state` (`staged` when everything built) and `staged`, one entry per resource with `project` (the project's directory in the repository), `resource` (its name), `status` (`success` or not) and `errors` |
| `publish <id> --json` | builds the rehearsed release in production | `state` (`published` when it holds), `published`, `backed_up` |
| `rollback <id> --json` | puts production back to before that release, whatever part of it was published | `state` (`rolled_back` when production is as before) |
| `production --json` | nothing | `released_commit`, the commit production stands at |
| `status <id> --json` | nothing | the platform's record of the release |

A verb that could not do what was asked exits non-zero and says why in the record's
`error`. A repository that names no such command has no release path this target can
drive: say so and stop.

## Steps

1. Read which pull requests are merged (`gh pr list --state merged`, or the
   pull-request deployment's proof call): only merged changes can be released, and a
   ticket whose pull request is still open is left out by the platform itself. When
   nothing is merged, stop and say so; merging is the reviewer's act and never this
   session's.
2. Run `bash "$KAPSO_TARGET_DIR/scripts/deploy.sh"`. It asks the platform to rehearse
   the release, fetches the released commit and exports it beside the workspace: every
   audit runs in that export, so a ticket's checks read the relations the released code
   names, whatever unmerged change the workspace still holds. A ticket is in the release
   when that commit holds every file its proof names exactly as the workspace does and
   the platform built each of its datasets; a ticket whose files differ there is left
   out and named. Two things stop the release before anything is published: a ticket
   whose files are merged while the platform did not build one of its datasets
   (production would carry the merge without the fix; `request_arguments` asks the
   platform for it, as the repository's documentation says), and a dataset the platform
   built that no released ticket covers (pass `allow_unaudited=yes` only when the user
   says those changes may ride along). The script runs every released ticket's specs on
   the rehearsal and goes on only when all pass. It then fingerprints on production
   every dataset the release builds (data-diff), or shows it absent there, and refuses
   to publish what it could not record, or when production moved to another commit in
   the meantime. The publication and the same specs on production run as a process of
   their own. When that ends in any way short of "published and every check passes",
   the script settles the release and exits 1: when production still stands at the
   earlier commit with every dataset reading as recorded, nothing was published and the
   record says `not_published`; otherwise it asks the platform for the rollback, which
   counts only with the platform's confirmation, the earlier commit and equal
   fingerprints. deploy/release/released.json holds the release's state at every step
   (`requested`, `rehearsed`, `publishing`, `published`, `not_published`,
   `rolling_back`, `rolled_back`, `rollback_failed`), written whole or not at all, and
   the releases this deployment made before it.
3. Prove one call: `echo '{}' | "$KAPSO_INTERPRETER" "$KAPSO_TARGET_DIR/scripts/run.py"`
   prints, per released ticket, how many checks pass on production as it stands.
4. Write the result file: `endpoint` the path deploy.sh printed as `KAPSO_ENDPOINT`,
   `run` as `{"kind": "script"}`, `values` with `release`, `commit` and `tickets`
   exactly as deploy.sh printed them, the artifacts (deploy/release/released.json and the
   check results and fingerprints beside it), and the proof call's inputs (`{}`) and
   output.
5. Say in the summary what production now holds that it did not before, which tickets
   were left out and why, which datasets read a released one and were not rebuilt (the
   platform builds what changed, not what reads it: each proof's downstream list names
   them), and how to undo it: the deployment's stop rolls the release back and compares
   production with its record from before the publication; its start releases the same
   commit again, never a newer one.

## When a release does not hold

A record that says `publishing`, `rolling_back` or `rollback_failed` means production
may hold part of a release: the deployment's stop is run first, and nothing new is
released until it has. The script settles a failed publication itself, but it cannot
outlive its own end: killed together with the publication (a deployment's time limit),
it leaves the record at `publishing`, and the stop is what settles it then. A rollback
is refused, and left to the platform's owner, when production stands at neither this
release's commit nor the one before it: a later release holds, and undoing this one
would undo that. The fingerprints that check a rollback are a screen (counts, sums,
distinct counts, a relation's absence), not a row comparison, which is why the
platform's own confirmation is required beside them; `rollback_failed` is an incident
for the platform's owner, with what the platform said and what differs in the record.
The start releases the commit the deployment's record holds: a rehearsal of any other
commit is refused before it is recorded.

## Never

Never write to production with the session's own connection, never merge a pull request
to make a release possible, never publish past a failing check, and never release a
commit that is not on the base branch. A release whose production checks fail is rolled
back, not patched.
