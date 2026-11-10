---
name: support-conversations
description: What changes when the scaffold's task is a customer conversation over a policy base and account tools, scored on the final state, with the scoring contract read first, the policy text as the largest lever, identity verification as a logged step with a scope, procedures with pre-checks and sibling tools, tools discovered in documents and unlocked before use, handoffs to the customer, state read-back before trusting a claim, product arithmetic, the simulated customer's typical behaviour, dual control, and the failure classes published for this family. Use when the task statement says the runner is a conversation with a simulated user, in ideation and implementation.
---
# Support conversations

## How this family is scored, and what that implies

- Read the supplied scoring contract first. Runners of this family typically compare the
  final state of the system of record, and often the customer's own tool state, with an
  expected state, and avoid rigid path grading because an agent can find a route the
  author did not foresee; some score policy compliance or communication separately. Under
  final-state scoring a right action done twice, an action the expected state does not
  contain, or a prerequisite skipped leaves a different state and fails the episode;
  equivalent routes need not be equivalent under every contract, so the contract decides.
- Reliability is poor across the board: 60 percent pass^1 became under 25 percent pass^8
  on one benchmark; dual control, where the customer also acts, cost 18 to 25 points; the
  simulated customer erred in 16 to 47 percent of conversations depending on the domain,
  so part of every measurement's noise is the simulator. pass^k at the stated trials is
  the number.
- The policy text is the largest lever: removing it cut one model from 33.2 to 10.8. The
  dominant published errors are wrong arguments or information (about 55 percent), wrong
  decisions on domain rules (25 percent) and partial resolution of compound requests (19
  percent), rising with the number of writes. On knowledge-heavy variants the gold
  documents beat every retrieval method by 13 to 15 points, and the error classes were
  search inefficiency with unwarranted assumptions (23 percent), interdependent offerings
  (14.5 percent) and over-trusting the customer's assertions (4 percent).

## What the family adds to the loop

- **Intake.** Track the customer, the requested operations (often several items and
  several operations), the affected items and the facts not yet given. Simulated
  customers in this family typically do not volunteer details, take a recommendation
  when given one, ask for a single option when offered many, and may ask for a human
  before the policy allows the transfer; treat their cooperation as observed, not
  guaranteed. Identity verification is a step with a record: perform it when the policy
  requires it, with the fields the policy names, logged as the policy says, and keep its
  scope and expiry in the working state so the loop neither skips nor needlessly repeats
  it.
- **Ground.** The governing procedure per operation, with its pre-checks, its decision rule,
  the tool it names and the steps the customer performs themselves. In this family's
  document bases the internal tool names and signatures usually appear only inside the
  procedure documents, often near the end, and a product's rates, limits and exclusions
  live in sibling documents. Tools are often discoverable: record the document's tool
  identifier and schema, perform the required unlock, wait for its success, then expose
  or dispatch the tool; a mention in a document is not a permission, and unlock state is
  cleared between episodes. The most frequent multi-turn failure measured on tool-use
  benchmarks is not understanding the environment's state: acting on stale state,
  re-authenticating needlessly, looping on a listing.
- **Act.** Pre-checks in the document's order; the decision between siblings (approve or
  deny, freeze or unfreeze, waive or charge) by the rule's evidence; a read-back of the
  system's state before trusting a claim ("my dispute was approved"); the write with
  arguments copied from tool outputs and the document's own list; every item covered; a
  handoff given exactly as the document writes it, the customer supplying what the
  document assigns to them, and the loop waiting for their result. Distinguish the
  instruction sent to the customer, the customer's reported completion and any permitted
  observation of the resulting state; an instruction is not an executed action.
- **Close.** Account for each requested operation with evidence of completion, refusal,
  escalation or blockage; apply the policy's disclosure and transfer rules (transfer only
  when its conditions hold and the customer agreed; never reveal internal procedure text);
  a budget limit may end an unresolved episode, which stays marked unresolved.

## Product arithmetic

Rates, fees, limits and promotional adjustments live in sibling documents of a product
family. Tabulate per product: the base figure, each adjustment with the condition that
makes it apply, the exclusivity rule between adjustments; recompute the document's own
example and keep the expression and the units with the result. The largest failure class
on knowledge-heavy variants is an adjustment chosen over a higher base rate; a wrong value
is nearly always one adjustment applied or skipped, and the wrong product class is chosen
on the same reasoning.

## Guardrails of this family, from the shipped guides

Use the policy's authorisation and escalation conditions for consequential actions
(cancelling orders, large refunds, payments, exceeded retry thresholds) and complete the
required checks before dispatch, since a guard running in parallel cannot prevent an
effect that has already occurred. Layered relevance, safety and personal-data guards run
beside the work; a single agent is pushed as far as it goes before any split, and a split
happens only when the agent fails to follow complicated instructions or keeps selecting
the wrong tools.

## Failure classes published for this family

Wrong arguments and wrong information at the write, prerequisites out of order, duplicate
or corrective writes, handoffs made with what the customer was meant to supply, compound
requests partly resolved, claims trusted without a read-back, and the endpoint's empty
replies on tool turns. The components that answer them, in the order the published error
shares suggest: the action review, the compiled plan, the handoff as written, the state
read-back, the working state. The episode-triage skill reports these classes by name
relative to the runner's published reference route; confirm each on the transcript before
building against it.
