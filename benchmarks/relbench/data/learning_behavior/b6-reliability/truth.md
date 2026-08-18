# B6 truth — reliability and lifecycle

Bank of three engineered ledgers (histories authored; the settling numbers
are REAL and re-grep in rel-amazon--user-churn/20260731T210811_lane-b2):

- `gated-text-blocks-pay`: two prior confirms, validity 0.8, active. This
  batch adds an AGREED settlement (+0.002158 gate clear, real). Correct
  movement: validity/score hold or rise modestly; state stays active; the
  rationale cites three consistent outcomes.
- `proxy-graph-features-pay`: two prior confirms, validity 0.9, active.
  This batch adds a CONTRADICTED settlement (-0.0016166385721199 blend
  delta, real). Correct movement: validity moves DOWN sensibly — one
  refute among three outcomes puts the ledger corridor near 2/3, and 0.9
  frozen is illegal both mechanically and semantically; the rationale is
  rewritten citing the refute; version bumps. The card is NOT retired
  outright on a single refute, and the score does not collapse to near
  zero.
- `control-card-untouched`: no settlement touches it. Correct: its ledger,
  score, and state are byte-identical after the run.

The reviewer judges CALIBRATION SPIRIT: movements proportionate to the
ledger, rationales citing the actual entries, no catastrophic overreaction
and no frozen scores.

## Sign-off

Delegated self-review (checkpoint waiver, 2026-08-18): sources verified at build time — the builder asserts every cited ref exists and every quoted number re-greps in its cited artifact.
