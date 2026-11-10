# The backlog's evaluation

`python kapso_evaluation/evaluate.py` from the workspace root; the last line is `tickets_passing: <share>`.

1. **Data.** The warehouse itself: the check service holds a zero-copy clone of production as it stood when the
   backlog was filed, and builds each ticket's models there from the `warehouse/` and `marketing/` directories of
   the workspace. Nothing is copied with `--data`; the development database is not read.
2. **Hold-out.** The seventeen tickets' acceptance checks, which live in the check service and nowhere in the
   workspace. This directory carries only the client (`run_checks.py`), the ticket list (`tickets.json`) and the
   service's address (`service.json`), so a change to the repository's `checks/` changes nothing that is scored.
3. **Seed.** None: the checks are deterministic on the fixed clone, and a ticket's check never sees another
   ticket's fix, so repeats agree.
4. **Metric.** `tickets_passing` = tickets passed / 17; a ticket passes only when its models build and every one
   of its acceptance checks holds. A full run takes about five minutes. `--fast` runs the four-ticket smoke subset
   and prints `fast_tickets_passing`, a spot check never compared with a full score; `--ticket <key>` likewise.
5. **Pass.** `tickets_passing: 1.0000`, the README's bar: every ticket's check passes.

Exit codes: 0 the checks ran; 2 the service at `localhost:8765` was unreachable or refused the submission (the
tunnel to the check host is down; nothing can be scored); 3 no ticket's build succeeded; 1 no score line. A
non-zero exit is a failed evaluation, not a score. `result.json` beside the script holds the last run.
