## MULTI-SELECT OVERRIDE (node expansion)

This round runs {{expansion_count}} implementation lanes in parallel, so the
single-solution output contract above is REPLACED: output EXACTLY
{{expansion_count}} final, self-contained solutions, ranked.

- Slot 1: the candidate (or synthesis) with the highest expected score.
- Slots 2+: deliberately COMPLEMENTARY picks — a different core mechanism or
  failure mode, the highest information value if slot 1's bet is wrong. Do
  not emit near-duplicates of slot 1: a second lane spent on a paraphrase is
  a wasted lane.
- Every slot must be fully self-contained (steps, hyperparameters, its own
  `# Coverage` section with MEASURED/ASSUMED marks) — each is implemented by
  an independent session that sees only its own solution.

Output format (STRICT): after your <selection_reasoning>, emit each solution
inside numbered tags, in rank order:

<solution_1>
...
</solution_1>
<solution_2>
...
</solution_2>
