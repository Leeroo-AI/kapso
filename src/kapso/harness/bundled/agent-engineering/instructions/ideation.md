# Ideation

Propose one dated hypothesis per attempt: the failure class it removes (from the triage),
the component that removes it, the model it is written for, and the measurement that
would retire it.

Work the expert's build order, and status each axis every iteration as active, saturated
or deferred:

1. the done-check and the specification the scaffold gives the model: what the scaffold
   runs to know it is done, and what its instructions, rules and examples say; the
   largest measured levers
2. the tool layer: which tools, their descriptions and examples, how results come back,
   how a malformed call is repaired
3. the loop and its bounds: phases, caps, loop detection, when the scaffold stops
4. the guards: what is denied in code rather than asked for in the prompt
5. the context policy: what stays in the window, what is distilled, what is dropped,
   where the working state lives, what is retrieved just in time
6. the model-facing scaffold: planning, review steps, best-of-n, memory, phase-dependent
   thinking; proposed last, with its expiry condition written down

- The scaffold-components skill is the catalogue with the measured effect of each
  component under its conditions and what is known not to work; a parameter change is
  worth an attempt only when it targets an observed failure and the task statement lists
  it as the scaffold's to change.
- When a proposed component shares a failure class or a build-order axis with one already
  in the scaffold, pair it with a single-flag re-ablation of the incumbent
  (scaffold-components), so an axis is not carried by two components when one would do; the
  re-ablation is a run, so it waits for a build that has met the target (feedback).
- Triage before proposing, as the episode-triage skill says, within a quarter of the
  session: the parent attempt's results and the transcripts the task statement permits,
  a handful of failed episodes read whole, the rest from the histogram. The next idea
  targets the largest addressable class, one in the harness or model bucket, among the
  episodes that fail in every run on record, and says which episodes it should flip.
- The first attempt has no run of the scaffold yet: triage the reference run when the
  task ships one, and propose from the endpoint's known limits and the task family's
  failure classes (for a conversation task, the support-conversations skill's).
- Every idea states its expected cost per episode, so feedback can name the trade-off.
- Prefer ideas whose failure the triage can show on the next run.
