# Agent-engineering guardrails (every module)

Read the task statement first: the task family, the model and its endpoint, the task runner
and its plug-in point, the development set, the trials, the metric, the budget. Its values
win over this page.

## Standing rules

- The scaffold is the thinnest wrapper that makes the model do the job, and every
  component in it is a dated hypothesis about what the model cannot do: recorded with the
  failure it was added for and the measurement, removed one at a time when it stops
  earning its place.
- The model is fixed. A scaffold changes the done-check, the tools, the loop, the context
  policy and the guards; never the weights, and never another model unless the task
  statement allows one. Endpoint settings the statement names as the scaffold's to change
  are interventions to measure; everything else about the endpoint is a fixed condition.
- Nothing is believed without the task runner: every claim is a measurement under the
  fixed conditions, with tasks and trials stated, judged by paired task-level flips
  against what two identical runs flip (the task's figure when given), never one number.
- Held-out tasks and answer keys never enter a session. The scaffold's runtime behaviour
  derives from the runner's public interface and the current task's own materials, never
  from recognition of evaluation identifiers or answers copied from development outcomes.
- Rules with zero exceptions live in code the scaffold's model cannot edit: hooks, deny
  rules, the sandbox, caps. A rule that keeps failing in a prompt moves to a hook.
- Tests, checks and scorers are read-only and outside the scaffold's reach; the model's
  grade of its own work counts for nothing.
- Every loop is bounded: steps, money, consecutive errors. An action with effects outside
  the session is reviewed before it is made, and recovered only by a documented
  compensation after reading the state, never by a speculative second action.
- Cost is a result: tokens, tool calls and seconds per episode beside the pass count.

## Where to look

- done-check-and-eval and episode-triage: how a scaffold is measured and read.
- scaffold-components: the catalogue by measured effect, the build order, the ledger.
- tool-layer, verification-and-stop, action-review, context-policy, grounding,
  deterministic-guards: the components, each with its evidence.
- local-model-serving: what a local endpoint does to the loop.
- support-conversations: what changes when the task is a customer conversation.
- episode-auditor and scaffold-integrity-auditor: the delegates verification asks.
