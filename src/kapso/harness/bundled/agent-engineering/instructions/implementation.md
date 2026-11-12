# Implementation

Build one component into the scaffold on the branch, then measure it.

- Start from the triage of the parent attempt (episode-triage): know which episodes the
  component must flip, and which attribution bucket they sit in.
- One component per attempt, behind a flag, with a transcript line when it runs and a
  ledger entry (scaffold-components): the model, the failure, the hypothesis, the
  measurement that would retire it.
- Build in the expert's order when the scaffold is new: the check the scaffold runs
  before it stops (verification-and-stop); the tool layer (tool-layer); the loop with its
  caps (deterministic-guards); the context policy (context-policy, grounding); only then
  the model-facing scaffolds, a review before irreversible actions among them
  (action-review).
- When the model is served locally, treat the endpoint as the local-model-serving skill
  says: parse tool calls as untrusted text, constrain arguments not responses, budget
  thinking per turn where the engine supports it. On any endpoint, empty replies,
  overflow and malformed calls are scaffold concerns, not evaluator noise.
- For a customer-conversation task, apply the support-conversations skill.
- Measure as the done-check-and-eval skill says: the fast subset while the loop is wrong,
  then the full development set at the task's trials, once. Never tune against one trial,
  never read a held-out task, never edit the evaluator.
- Ask the episode-auditor about one failed episode when its class is unclear; its report
  names the first upstream wrong step the evidence supports.
- Finish with the evaluator's original output kept, results/episodes.jsonl written by the
  adapter over that output, results/summary.json written by the pass^k script, the triage
  saved beside them, the ledger entry, and attempt notes with the pass counts before and
  after, the cost per episode, and the scaffold revision and settings that produced them.
