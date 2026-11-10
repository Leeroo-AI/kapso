# Deployment

Deliver through the channel the task names. Two targets cover a data-engineering
delivery, and they follow one another: the pull-request skill opens the change for
review, one pull request for a single ticket and one per closed ticket for a backlog
(ticket-backlog: the ledger says which tickets are closed); the warehouse-release skill
releases what a reviewer merged into production, when the repository has a release
command. A request to open pull requests, ship, deliver or release the result is a
deployment through one of these targets, never branches and pushes made by hand. A
release's deployment session and its start step run for the length of the tickets'
audits, so the repository's Kapso config sets the deployment deadlines to cover them
(warehouse-release, "What the deadlines must cover").

- The session names the units and their proofs; the target does the rest at publication.
  It renders every title and body from the proof and its evidence (pull-request-proof),
  computes the label and lowers it to what verification measured again, so a title or a
  verdict is never typed. A title carries no tag; the body opens by saying the change is
  verified, or names each point still open. A no-change outcome ships its evidence in
  the summary and no pull request.
- A pull request changes only the files its proof names, and it builds on its own: a
  unit whose code needs another unit's change is stacked on it (the proof's `needs`),
  and every branch is run through the task's acceptance check alone before it is
  pushed. Where the task pins no such check, or one that reads the build namespace, a
  branch cannot be checked alone and its pull request carries that as an open point.
  Validation artifacts the task asks for ship beside the code; everything else under
  results/ and prepared/ stays behind.
- Merging is the reviewer's act, and the summary hands them the stack's merge rule
  (pull-request, "Merging a stack"). A release takes only what is merged, is rehearsed
  where production is not touched, and is published only when every released ticket's
  checks pass on the rehearsal, nothing unaudited rides along and production as it
  stands was recorded. It is checked again on production, and when the publication
  fails in any way or a check fails there, the target rolls it back itself and checks
  production against that record: a release is never patched in place.
- A change to a schedule, a partition layout, an incremental model or a snapshot ships
  with its runbook (the proof's `runbook`, validation-report): what to run once, in
  what order, how to see that it worked, how to recover.
- No warehouse credentials, account names or connection strings; no personal data from
  any row; the project's own configuration is the only reference to the warehouse.
