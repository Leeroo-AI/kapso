# Feedback

Score against the task statement's metric and the parent attempt, or the reference run
the task statement names when there is no parent, not against an impression of the
transcripts.

- The score is the task's metric on the full set at the stated trials, read from the
  evaluator's summary (done-check-and-eval). Read pass^k and the cost per episode beside
  it: a change that raises the metric while lowering pass^k, or while multiplying the
  tokens, steps or tool calls, is a trade-off, and the feedback names it rather than
  hiding either number.
- Read a gain against a single run's noise, the episodes that flip between two identical
  runs; say when it is inside that, and ask for a second run before the attempt becomes a
  parent when the budget allows. With a small development set, describe the paired flips
  and their limits rather than declaring a threshold.
- Distinguish fixed conditions from the settings the task statement lists as the
  scaffold's: a permitted change to one of those is an intervention to judge, not an
  invalid evaluation.
- Judge the hypothesis, not only the number: did the component flip the episodes it was
  written for, and did new failure classes appear? The ledger entry says what was
  claimed. A number that rose while those episodes did not flip leaves the mechanism
  unconfirmed: the gain is not yet a win, and the attempt does not become a parent until
  the scaffold-components mechanism-dependence check passes or the hypothesis is rewritten
  to the episodes that actually moved.
- Say which axis of the build order the next attempt should move on, with the largest
  remaining failure class and its attribution bucket as the reason.
- Spend a run where it can change the decision. When a full run is a large share of the
  budget, a build that only ties or trails the target earns no confirmation run and no
  re-ablation: the budget buys a distinct idea instead. How many runs confirm a version
  is the task statement's; the feedback names a run that re-measured a result no rerun
  could promote.
- Deliver the best validated build, not the last one built. When an earlier version or a
  parent measured higher under the same fixed conditions, that build is the delivered
  scaffold, and the delivery reports every run of it, never its best single run.
- "Met" is the task's target on the full set at the stated trials, with the cost per
  episode stated and the replication the task statement asks for, not the first run
  that clears the number. Repeated selection on one development set does not predict
  the held-out score; say what evidence supports acceptance.
