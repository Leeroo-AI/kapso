---
name: ticket-backlog
description: How a senior data engineer works a backlog of tickets on one warehouse rather than one ticket, with the inventory that makes the backlog legible (ticket, dataset, layer, kind of change, acceptance), the ordering by lineage and by how decidable a ticket is, one ticket as one unit of work with its own baseline, build, acceptance check, proof and commit, shared causes fixed once where the lineage shares them, the regression check across tickets already closed, an acceptance check someone else owns treated as an oracle and never as a target, the parking rule for a ticket that resists, and the per-ticket proof layout with its ledger. Use whenever the task statement holds more than one ticket, in setup for the inventory, in ideation to choose and order a batch, in implementation for the loop, in verification and feedback to judge per ticket.
---
# A backlog of tickets

## The inventory

One row per ticket, written once in setup (prepared/backlog/inventory.json) and read by
every session:

| Field | What it holds |
| --- | --- |
| `ticket` | the ticket's key, as the task names it (the key its acceptance check takes, where it has one): letters, digits, `.`, `_` and `-`, since it names a file, a directory and a branch |
| `datasets` | the model, snapshot or table the ticket names, or `unknown` until the investigation resolves it, with file, layer and materialization as the project resolves them today (`exists: false` for one to create) |
| `output` | the relation the ticket expects, schema included, when the ticket states it |
| `change_kind` | fix, new, refactor, performance or migration (validation-report); `unclear` when only the investigation will tell |
| `acceptance` | where "done" is decided: the task's check for this ticket when there is one, and the ticket's own stated criteria |
| `upstream_of`, `shares_parent_with` | other tickets whose datasets consume this one's, or read the same parent |

The inventory is read from the tickets and the project's graph; it runs no build and
reproduces no number. It is mechanical on purpose: a setup that investigates every ticket
of a backlog investigates none of them well.

## The order

- Lineage first: a ticket whose dataset feeds another ticket's is worked before it, and
  two tickets on one dataset are worked together.
- Then decidability: a ticket whose acceptance is fully stated (named columns, rules,
  tolerances) before one that needs an investigation to know what "right" is. The first
  kind is closed by building to the statement and checking it; the second needs the
  bisection (lineage-bisection) and gets the time the first kind did not take.
- A batch is the tickets one session can close with their proofs in the time the task
  statement leaves it. A smaller batch closed beats a larger one half-proved.

## One ticket, one unit

For each ticket, in this order, before the next one starts:

1. Read the ticket whole; name the dataset, the grain and the kind of change.
2. Write its oracle, one spec per dataset of the ticket,
   results/<ticket>/spec_<dataset>.json (invariants), from the ticket's own statements
   and the sources, before the dataset changes or exists. Then take its baseline before
   any of its files change (validation-report):
   `evidence.py baseline --ticket <ticket> --kind <kind> --select <model> ...`. Where a
   reset is pinned it puts the build namespace back to what production holds; it fixes
   the specs, and measures
   the code as it stands; for a cost or speed ticket add the billed quantity with its
   run conditions by hand.
3. Change one dataset, build it with selection, look at its rows. When a change only
   builds together (a producer renamed with its consumer, a contract with its users),
   the unit is the smallest set that builds, built together, each consumer checked.
4. `evidence.py prove --ticket <ticket>` (with `--with-downstream` when anything reads
   a dataset whose content changed). It also runs the ticket's acceptance check, when
   the setup pinned one, and records the verdict.
5. Write results/proofs/<ticket>.json (validation-report), run
   `evidence.py label --ticket <ticket> --write`, and update results/backlog.json.
6. Commit the ticket's files alone, with the ticket's key in the message, before the
   next ticket's baseline: a file edited before a baseline is not that ticket's change.

Independent datasets edited together and then built once cannot attribute a failure, and
a red check after a batch edit costs more than the batch saved. A ticket closed without
steps 2 and 4 passes its check with nothing measured behind it, and its label says so.

Where each baseline resets the namespace, a ticket is built and measured on what
production holds, without the other tickets' changes around it. That is how a ticket's
pull request sees it, and its acceptance check too when the task evaluates one ticket
at a time; a task statement that evaluates them another way decides. When a ticket's
code does not build without another ticket's change (it introduces something this
ticket's code refers to), name that ticket in the proof's `needs`; that
ticket is then worked first. When a ticket's checks can pass only on data another
ticket's change produces, its baseline selects that ticket's datasets as well, with
their specs, so its own measurement builds and checks them. What the whole backlog
does together is measured once at the end: the task's full acceptance run, and for a
shared parent the `--with-downstream` comparison in the proof of the ticket that
changed it.

## Between tickets

- A cause two tickets share (a time zone, a fan-out join, a status filter in a common
  parent) is fixed once where the lineage shares it, under the ticket worked first: its
  proof names the file in `changed_files`. The other ticket says in its cause where the
  fix is and names that ticket in `needs`; its own `changed_files` holds only what it
  changed itself. When nothing is left for it to change, it closes as `no_change`, with
  the evidence that its symptom is gone at that commit. The same mechanism in two
  unrelated datasets is two fixes.
- After a ticket closes, and after any ticket's edits are kept whatever its status, rerun
  the specs, and the acceptance checks where the task has them, of the closed tickets
  whose lineage the change touched; all of
  them when one run is cheap. A ticket that regresses reopens.
- Shared layers stay shared. When a ticket's cause sits in a shared layer (a staging model,
  a source, project configuration), the fix goes there only when the task allows it, as its
  own decision with every consumer in its blast radius (the guardrails), never as a side
  effect of one ticket; where the task forbids it, the cause is reported upstream and the
  ticket's own dataset carries only what the ticket asks of it.

## An acceptance check someone else owns

A check run by a CI job, a platform service or a reviewer, whose internals the task does
not show, is the ticket's acceptance oracle and is treated as one:

- It decides "done"; it does not explain "why". A pass without a mechanism in the proof
  and a measurement behind it is an unexplained pass, and the computed label is unsure.
  Its hidden definitions are listed as not inspected; they are not an assumption.
- It builds what the task statement says it builds, on the state the task statement
  names. When that is one ticket's datasets on what production held, measure the ticket
  the same way (the baseline's reset does), not on a namespace the other tickets changed.
- On a fail with a fix the data supports, re-read the ticket for what the check can see
  and the fix ignored: the exact relation name and schema, the materialization, the column
  names, types and precision, the stated tolerances, any date or window the ticket fixes.
  Then read whatever the check returns about its build. Never guess at a hidden expectation.
- A change made because the verdict flipped, with no reason in the data or the ticket, is
  void: the check is being fitted, not the dataset fixed.
- The check's own files, its service and its inputs are never read around, edited or
  probed; what it returns is all it gives.

## A ticket that resists

A bounded effort, then park: the ledger records `parked` with what was reproduced, what
was ruled out and the next question, and the session moves on. What the session leaves in
the tree is what ships, so a parked ticket's unproven edits are reverted before moving
on; an edit kept on purpose (a partial fix that is right as far as it goes) is named in
the note, and the closed tickets it can reach are rechecked as above. A kept edit has
no closed ticket's proof behind it, so every ticket measured with it in the tree reads
`unsure` until its own ticket closes or it is reverted. A premise the data
contradicts ends as `no_change` with its evidence (validation-report), which is a closed
ticket, not a parked one.

## The ledger

results/backlog.json, rewritten after every ticket:

```json
{"tickets": [
  {"ticket": "settlement-double-count", "status": "done", "acceptance": "pass",
   "proof": "results/proofs/settlement-double-count.json", "commit": "abc1234"},
  {"ticket": "active-merchant-definition", "status": "parked", "acceptance": "fail",
   "note": "symptom reproduced; the status filter ruled out; next: the late-arriving refunds"},
  {"ticket": "legacy-export", "status": "open", "acceptance": "not_run"}
]}
```

`status` is `done`, `no_change`, `parked` or `open`; `acceptance` is the task's check for
that ticket as `prove` last recorded it (`pass`, `fail`, `not_run`), never inferred; a
`no_change` ticket has no proof run, and its row carries the check as it was last run by
hand, or `not_run`. The label is not in the ledger: it is in the ticket's proof, computed. Every ticket of the
inventory has exactly one row; a `done` or `no_change` row names its own ticket's proof,
whose `change_kind` is `no_change` exactly when the row's status is. An attempt that
starts from an earlier one starts from its ledger. The backlog is finished when no row
is `open` or `parked`, not when a share of it passes.
