---
name: pull-request
description: "The pull-request deployment target: delivers the build as pull requests on the repository's host through the GitHub CLI, one for a single ticket or one per closed ticket of a backlog, each changing only the files its proof names, stacked on the change it needs to build, checked on its own before it is opened, with its title and body rendered at publication from the proof, its evidence and the verification report, and the verdict stated in the body. Use when deploy() names the pull-request target, when the task's delivery channel is a pull request, or when the user asks for the result as pull requests; never to push to the base branch."
metadata:
  run_kinds: [script]
---
# Pull requests

## What the deployment session has

The solution's repository at its delivered commit in the workspace, with the proofs the
build wrote (results/proof.json, or results/proofs/<ticket>.json and the ledger
results/backlog.json for a backlog) and the evidence each proof rests on
(results/<ticket>/evidence.json), `gh` logged in on the machine, and the options of
the `deploy()` call: `KAPSO_OPTION_REPO` (owner/name of the repository to open against;
when unset, the repository the solution's code came from, `KAPSO_SOURCE_REMOTE`) and
`KAPSO_OPTION_BASE` (the branch to open against; `main` when unset). When
`KAPSO_VERIFICATION_DIR` is set it holds what verification wrote about the delivered
attempt; its `evidence_check.json` is the re-measurement of every ticket, and the
target reads it by itself.

The target drives GitHub only. On another host do not run it and do not imitate it by
hand: say so in the summary, with each unit's files and its rendered title and body
(pull-request-proof) for a person to open.

## Steps

1. Decide the units. A single ticket is one unit (`change`). For a backlog, every ticket
   the ledger calls `done` is a unit; a `no_change` ticket ships no pull request (its
   evidence goes in the summary), and a `parked` or `open` ticket ships nothing and is
   named in the summary with its note, so the reader sees what the delivery does not
   cover. When no ticket is `done`, there is nothing to open: say so in the summary
   with each decision's evidence, and do not run the target. Two tickets whose whole
   change is one shared file cannot be opened apart: list only the one the ledger
   closed first, whose pull request then carries the file whole, and say in the summary
   that the other ships with it.
2. Write deploy/pull_requests/manifest.json, one entry per unit, and nothing else:

   ```json
   {"units": [{"unit": "<ticket>", "proof": "results/proofs/<ticket>.json"}]}
   ```

   A unit is named after its ticket and ships under that ticket's own proof, whose
   `changed_files` are its files and whose `needs` is the one unit without whose change
   its code does not build (it is then stacked on that unit). No title, body or label
   is written by the session.
3. Run `bash "$KAPSO_TARGET_DIR/scripts/deploy.sh"`. It makes the branch
   `kapso/$KAPSO_DEPLOYMENT_NAME/<unit>` of every unit in a checkout of its own, from
   the base branch (or from the branch of the unit it needs) plus the unit's files as
   the workspace holds them; the workspace itself is never switched. Then it runs the
   task's acceptance check on every branch alone (the `acceptance_command` of
   prepared/environment.json, in a checkout that holds the repository's tracked files
   and nothing else) before anything is pushed. A branch that fails alone stops the
   deployment with its name: the unit lacks a `needs` in its proof (correct the proof,
   compute its label again, run the script again), or it is not ready and the summary
   says why. Where no check is pinned, or the pinned one reads the build namespace (no
   `acceptance_at_a_time`), no branch can be checked alone and every pull request
   carries that as an open point. A file is delivered whole, so a file several units
   changed ships in the one the others are stacked on, both pull requests say so, and
   units that share a file without being stacked that way are refused. Then the script
   renders each title and body (pull-request-proof) from verification's measurement of
   the delivered code. The title is one plain line; the body opens with the verdict that
   measurement, the verification report and the branch's check earn, as a sentence:
   verified, or each point still open. It pushes each branch, opens the pull
   requests labelled `kapso-$KAPSO_DEPLOYMENT_NAME`, each saying what it is stacked on
   (or what is stacked on it), how to merge without losing the stack and what its
   branch check returned, and keeps deploy/pull_requests/opened.json current after
   every one, with each unit's `label` and `open_points`. The base branch is never
   pushed. A second run updates the pull requests
   the first one opened and keeps their record.
4. Prove one call: `echo '{}' | "$KAPSO_INTERPRETER" "$KAPSO_TARGET_DIR/scripts/run.py"`
   prints each pull request's number, state, target branch, mergeability and checks.
5. Write the result file: `endpoint` the URL deploy.sh printed as `KAPSO_ENDPOINT` (the
   one pull request, or the repository's list filtered by the label), `run` as
   `{"kind": "script"}`, `values` with `repo`, `base`, `label` and `pr_numbers` (comma
   separated) exactly as deploy.sh printed them, the artifacts (the manifest,
   prepared.json, branch_checks.json, opened.json and the rendered titles and bodies),
   and the proof call's inputs (`{}`) and output. The summary names the merge order (the
   unit others are stacked on first), repeats the stack's merge rule below in one line,
   and names every pull request that is not verified with its open points, read from
   opened.json (a `label` other than `safe to merge`, and its `open_points`).

## Merging a stack

Written for whoever merges, and repeated in the pull requests themselves. The pull
request others are stacked on is merged first with a merge commit (the merge button's
"Create a merge commit", or `gh pr merge <number> --merge`); where the repository allows
only squash or rebase merges, the summary says that each stacked pull request needs a
rebase onto the base branch once the one under it is merged. Its branch is left to the host. When GitHub deletes
a merged branch itself (the repository's setting, or the Delete branch button) it moves
every pull request stacked on it onto the base branch. A branch deleted from the command
line (`gh pr merge --delete-branch`, `git push --delete`) closes them instead, and a
closed one can be neither moved nor reopened while that branch is gone. The deployment's
start (`scripts/start.sh`) undoes that: it puts the branch back at the recorded commit,
reopens them, moves each onto the branch of the nearest unit under it that is not merged
yet (the base branch when all are merged), and removes the branch again once no open
pull request of the repository targets it. In a repository that keeps merged branches,
each stacked pull request is moved by hand once the one under it is merged, or by that
same start.

## Never

Never push to the base branch, push over a branch this deployment did not create, merge
a pull request, or open one for a ticket without a proof. Never write a title, a body or
a label by hand: what is published is what the scripts render from the measurements.
