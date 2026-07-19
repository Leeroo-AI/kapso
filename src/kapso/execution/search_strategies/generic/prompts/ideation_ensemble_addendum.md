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
- include a `# Coverage` section: the observable axes along which the
  evaluation inputs vary (input distribution — format/length/category/
  domain/locale/difficulty; reference/output register; metric mechanics
  incl. weighting and noise floor; harness-controlled vs artifact-owned
  inference knobs; permitted-data geometry) and how the candidate's
  data/method covers each — every axis marked MEASURED (cite the source)
  or ASSUMED (the implementor verifies it in recon),
- sit inside its own solution block: open the block with the
  solution start tag, close it with the solution end tag.

Make the candidates meaningfully different from each other: pick two distinct
moves from {extend the current approach, remove/replace a weak part, large
retune, small targeted tune, change the core idea}.

Constraint: do NOT propose generating training data or task artifacts with
third-party LLM APIs (e.g. the OpenAI API) — scaffold reasoning is allowed,
API-produced training artifacts are not.
