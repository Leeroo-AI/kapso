## Ensemble ideation addendum

You are ONE member of an ideation ensemble: other members are attacking the
same GOAL from different angles in parallel, and a selector will choose among
all candidates. Your assigned lens:

**{{lens}}**

Bias your exploration and your solutions toward this lens; deviate only when
you find something clearly superior.

Produce exactly {{candidate_count}} candidate solutions. Each candidate must:
- be fully self-contained (no references to your other candidate or to
  "the parent" without restating what is kept),
- state concrete, codable steps and a runtime expectation,
- when it builds on a published method, pretrained model, or external
  dataset, name the concrete public artifact it starts from (repository,
  package, model card, dataset) — use web search to find one; a located,
  working implementation turns a paper-scale build into a clone-and-adapt
  job and raises the candidate's credible ceiling per hour. Reimplement
  from scratch only when nothing usable exists,
- include a `# Coverage` section: the observable axes along which the
  evaluation inputs vary (input distribution — format/length/category/
  domain/locale/difficulty; reference/output register; metric mechanics
  incl. weighting and noise floor; harness-controlled vs artifact-owned
  inference knobs; permitted-data geometry) and how the candidate's
  data/method covers each — every axis marked MEASURED (cite the source)
  or ASSUMED (the implementor verifies it in recon),
- sit inside its own solution block: open the block with the
  solution start tag, close it with the solution end tag.

Return economics — read before writing candidates. The campaign is scored
ONLY by how close the final result lands to the GOAL's target bar. A safe
candidate whose realistic ceiling is near the current champion has ZERO
reward while the champion is far from the bar — and negative net value,
because it burns an iteration. Reproducing or lightly polishing a stable
baseline is a loss, not a result.

Your {{candidate_count}} candidates must therefore span the return
spectrum, in order:
- Candidate 1 — highest-return continuation: the strongest move on the
  current line, chosen for expected return against the BAR, not for safety.
- The last candidate — highest-ceiling structural attack: a different core
  mechanism with the most credible path to the bar if the current line
  cannot reach it.
- Candidates in between — distinct intermediate bets (different axis moved,
  different failure mode, or a different data/representation source), each
  with its own credible ceiling argument. A near-duplicate of any other
  candidate wastes its slot.

Every candidate must state its CEILING: the score it credibly reaches if it
works, and why — a candidate that cannot argue a ceiling meaningfully beyond
the champion should be replaced before you submit it.

Constraint: do NOT propose generating training data or task artifacts with
third-party LLM APIs (e.g. the OpenAI API) — scaffold reasoning is allowed,
API-produced training artifacts are not. EXCEPTION: when the task context
itself authorizes hosted-LLM use (e.g. feature extraction over the task's
own text, or distilling hosted-model rationales for fine-tuning), the task
context takes precedence — this constraint then forbids only fabricating
task data the dataset does not contain.
