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
  re-ablation is a run, so it waits for a build that has cleared the bar (feedback).
- Triage before proposing, with the episode-triage skill, within a quarter of the session:
  the parent attempt's results/episodes.jsonl and the transcripts the task statement
  permits (the attempt's own under results/, the reference run's under the task's
  files), a handful of failed episodes read whole, the rest from the histogram; record
  what was not read. The first attempt has no run of the scaffold yet: triage the
  reference run the task ships for the failure classes of the task family, and propose
  from the endpoint's known limits and the support-conversations classes, naming the
  first evaluation as the measurement that will show the scaffold's own failures. The
  next idea targets the largest addressable class, one in the harness or model bucket, since
  a class dominated by the grader or the environment is a ceiling, not a target, its share
  read from the transcripts or the episode-auditor rather than the symptom counts, and says
  which episodes it should flip. Read those episodes from the runs on record, not the last
  run alone: the gap worth an attempt is the episodes that fail in every run, measured
  against the union that ever passes, and an episode that flips between two identical runs
  is that run's noise, not a target, since flipping it back is no durable gain.
- For a conversation task, the support-conversations skill lists the failure classes
  that family adds and the components that answer them.
- Every idea states its expected cost per episode, so feedback can name the trade-off.
- Prefer ideas whose failure the triage can show on the next run.
