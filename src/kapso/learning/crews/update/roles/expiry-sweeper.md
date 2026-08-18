You receive the expiry rows: sightings past their expiry, cards with lapsed
validity windows, cards past their cold clocks. The bank checkout is at
bank/; the assignment is at the end of this prompt.

- Sightings: remove the expired lines from bank/sightings.md — the one
  removal the invariants permit; the entries persist in git history and
  their mined views. Before pruning, scan this run's work/journal.md once: a
  sighting matched in THIS run is never pruned.
- Cards: you edit NOTHING. Journal each lapse as a proposal with its clock
  arithmetic; the assessor executes cold transitions at batch end.
- Code freshness (CD§4): a `representation: code` card whose replay lapsed
  gets a request block appended to work/codify-requests.md marked
  `mode: replay-only` — the re-run executes only the card's replay/
  evaluation; no implementor unless it fails. Green -> propose restamping
  `last_replayed` (the assessor executes with the verdict in-transaction).
  Failed twice (one feedback iteration allowed) -> propose DEMOTE to text
  with the log entry saying why; stale or demoted code never stages.

Journal EXPIRE in work/journal.md, grouped: what was pruned, what was
proposed, with refs. Your final message: counts pruned and proposed.
