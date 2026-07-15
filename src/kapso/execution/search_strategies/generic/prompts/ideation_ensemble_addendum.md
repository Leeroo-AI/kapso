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
- sit between its own <solution> and </solution> tags.

Make the candidates meaningfully different from each other: pick two distinct
moves from {extend the current approach, remove/replace a weak part, large
retune, small targeted tune, change the core idea}.

Constraint: do NOT propose generating training data or task artifacts with
third-party LLM APIs (e.g. the OpenAI API) — scaffold reasoning is allowed,
API-produced training artifacts are not.
