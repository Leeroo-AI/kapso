---
name: action-review
description: Compile the task's materials into an ordered, gated plan and review every consequential action against it before it is made, with effects classed by reversibility and compensation, prerequisites in order, sibling choices by evidence, values copied not paraphrased, arguments traced to their source, computed values with their arithmetic, every item accounted for, handoffs as written, checks before done, recovery by documented compensation only, and the calibration a model reviewer needs before it gates anything. Use when building the ground and act phases, or when the triage shows skipped steps, order violations, wrong arguments, missed items or duplicate actions.
---
# Action review

Classify each effect by reversibility, idempotency and the compensation its tool contract
supports: a write to a system of record, a submission, a message sent, a payment, a
deployment, a destructive command, a pushed commit. The runner scores the resulting state,
so a wrong action is not undone by a right one after it, and the review before the action
is the cheapest undo. The review is a separate step that sees only the working state, the
plan, the source evidence and the proposed action, and answers go or stop; deterministic
checks stay in code and the model reviews only the residual ambiguity.

## The compiled plan

One per operation the task asks for, written to the working state before any action:

```
operation:     <what is asked, in the task's words>
sources:       <documents, files, schemas read in full>
conditions:    - <condition> | evidence: <tool output or answer> | source: <where written>
prerequisites: - <step as documented>(<args>) | must precede: <the action> | result: <when run>
decision:      rule: <quoted> | branches: <condition -> action> | chosen: <action> because <evidence>
action:        <exact documented name or command> | signature: <as documented>
               args: <name> = <value> | from: user | tool output | documented list | computed
items:         - <each record, file, account the task named> -> the action that covers it
handoffs:      - <what another party does> | they supply: <what the materials assign to them>
after:         - <logging, confirmation, verification the materials require>
```

A field without a source is a question or a lookup, not a guess.

## Rules the measurements made necessary

- **Order is part of the state.** A request submitted before its prerequisites, or a close
  before its logging step, leaves a different record than the documented order even when
  every step eventually happens. Run the prerequisites in the documented order and wait
  for each result before the action.
- **Sibling actions are a decision, not a guess.** Approve and deny, freeze and unfreeze,
  create and update, merge and close: choose from the rule's evidence, write the evidence
  beside the choice, never do both.
- **Vocabulary is copied, not paraphrased.** Enumerated values (a class, a reason code, a
  status, a flag name) come verbatim from the documented list; a near-synonym or a
  shortened form is a different value to the system.
- **Arguments come from outputs.** Ids, paths, dates and amounts are copied from the tool
  output or the message that produced them, never retyped from memory, and the review
  checks the trace.
- **Computed values show their arithmetic.** For a rate, a fee, a limit, a count: the base
  figure, each adjustment with the condition that makes it apply, the exclusivity rule
  between adjustments, the result, and the material's own example recomputed as a check.
- **Every item is accounted for.** Map each requested item to an observed outcome; a
  documented batch action may cover several items, and its result says which.
- **A handoff is handed as written.** When the materials assign a step to another party
  (the customer runs a tool, a reviewer approves, a job runs on a schedule), the scaffold
  hands it over with exactly what the materials give the agent, says what the other party
  supplies, and waits for the result before the next step. Pre-filling the other party's
  part changes what they do, and their part is in the scored state.
- **A claim about the world is read back before anything depends on it.** "The tests
  pass", "my dispute was approved", "the table exists": one read tool call, not the
  requester's word or the model's own last diff.
- **Checks before done.** The completion checks available to the deployed scaffold run
  before an episode is declared complete: the permitted tests on the final diff, a
  read-back of the state, the deliverable's existence. The campaign's evaluator and its
  fast mode are builder-side measurements, never runtime tools. A failing check is work
  left, not a note.
- **Recovery by contract.** An action that might be wrong is not made. After an ambiguous
  result, read the state before retrying; use a documented correction or compensation
  only after observing the current state and checking the permission; otherwise stop or
  escalate. Never a speculative second write to conceal an uncertain first one.

## Prompt fragment for the compile step

"You have read the materials listed in the working state. Write the compiled plan for
<operation> in the format above. Every field names its source. List every item the task
named. Mark a field you cannot fill from the materials or the tool outputs as `UNKNOWN:
<the question to ask or the tool to call>`. Do not call a tool in this step."

## Prompt fragment for the review

"Here is the working state, the compiled plan, the source evidence and the action about to
be made. Answer with `go` or `stop: <reason>`. Stop when a prerequisite is missing or came
after this step, when an argument has no source or differs from its source, when a
computed value's arithmetic is not shown, when this action was already made, when an item
the task named has no covering action, or when a check the plan requires has not run."

## Calibrate the reviewer before it gates anything

A fresh call of the same model can repeat the actor's mistake or reject a correct action,
and a review before every action costs latency. Before enabling the reviewer as a gate,
replay permitted pre-action snapshots with correct proposals and with mutations that
remove a prerequisite, change a sourced argument, duplicate an effect or contradict the
current state; measure false approvals, false rejections and the added latency; give the
reviewer the source evidence, not only the actor's compiled plan; and keep the review only
where it improves the final outcome at the same total budget.
