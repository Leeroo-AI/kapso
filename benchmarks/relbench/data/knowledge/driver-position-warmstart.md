# rel-f1 / driver-position — reward bar and prior-campaign evidence (warm start)

## Reward bar (unchanged)

The method to beat is **KumoRFM (fine-tuned)**: **TEST MAE 2.731** (board NMAE
0.3887, divisor 7.0253). This campaign earns reward ONLY by SURPASSING that
test score. Any final result that does not beat 2.731 carries ZERO reward — a
stable champion far from the bar is a loss of the time it consumed, not a
partial success. Prior campaigns measured val→test drift of **+0.8 to +1.3
MAE**: a validation score only ~0.1 better than a rival's can still be a worse
test entry. Plan for transfer, not just validation rank.

## What prior campaigns on this exact task measured (in-run evidence only)

Three full 10-hour campaigns preceded this one. Their in-loop measurements —
not speculation — are summarized here so you do not re-buy them. Every closure
below states its measured cause; a materially different variant that addresses
that cause MAY be retried.

**Champion family (twice-proven): staleness/horizon-matched training.**
Both recent campaigns' champions trained on episodes whose features were
computed from artificially frozen database snapshots (0–78 months before the
seed), teaching the model the exact signal decay the frozen-2009 test regime
imposes. Best validation lineage: event-token sequence representation
("StateSet": ordered race events, forward-only driver/constructor/qualifying/
reliability channels, roster tensors, OOF decoder) hybridized with
staleness-trained gradient-boosted trees ("StaleRank"). Best val MAE reached
2.6486 (from a 3.02–3.16 seed range). Representation widening (recent-form
channels, teammate-relative pooling, era-dependent variance, cohort priors)
produced measurable val gains in several distinct steps — the feature matrix
was never "done".

**Measured-closed directions (with causes):**
- Cross-lane blend + residual stacking + isotonic recalibration: fully
  executed; blend cashed only +0.005 val; isotonic decoding "lost every fold".
  Cause: lanes too correlated (residual corr ≈ 0.925).
- TabPFN as direct predictor: executed twice in two campaigns; lost to the
  incumbent both times (e.g. debug 5.955 vs 5.332; full 6.187 vs 6.011;
  prediction range 55% of StateSet's) and judge-closed at champion parity.
  TabPFN checkpoints remain in the shared cache if a materially different
  usage (e.g. as one member of a properly decorrelated stack) is argued.
- Synthetic-prior compact network: pretrained on 100k procedurally generated
  universes; synthetic-domain gains (>3.37 MAE) never transferred — real
  aggregate was negative. Cause: synthetic prior mismatch, not capacity.
- Monte-Carlo particle/race-world simulators, Hungarian-transport donor
  worlds, dynamic Plackett–Luce state-spaces (earlier campaign): all failed
  frozen-origin gates; sampling added dispersion without inferring
  long-horizon entrant strength.
- TrueSkill-Through-Time package (`trueskillthroughtime==1.1.0`) integrates
  cleanly (mind tied positionOrder values vs zero draw probability — dedupe
  or set p_draw) and its features are available, but no TTT-based candidate
  beat the staleness champion.

**External data (one campaign, artifact reusable):** `external_junior_v1` —
53 revision-pinned Wikipedia season pages (GP2, F3 Euro, FR3.5/Euro Open by
Nissan), 1,664 parsed standings records, 194 conservative F1-driver matches,
strictly pre-2010 content, provenance documented. Measured coverage: ~24–26%
of historical cold-replay rows (below that campaign's 30% gate → it disabled
the branch) but ~72% of the test cold mass. The validation-coverage asymmetry
is the open problem: the artifact is most useful precisely where replays
cannot score it. If used, design a validation that can actually measure it
(e.g. late-origin replays where junior coverage exists).

**Known evaluation-protocol facts:** oracle rank→value decoding reaches
val ≈ 1.10–1.74 (large decoding headroom); ~43% of test rows are zero-history
debutants rising to ~64% by 2016; test horizons span 0.33–6.57 years after
the 2009-11 freeze; per-iteration frozen-origin/era-replay validation is the
only in-run instrument that predicts transfer — trust it over fresh-feature
validation rank when they disagree.

## Shared-cache inventory (pre-loaded on this machine)

The shared cache at the registered location already contains, from the prior
campaign: `f1_stateset_v1`, `f1_stalerank_v1_lane0`, `f1_chainset_v1`,
`f1_direct_chainset_v1_lane0`, `f1_crossfit_stack_v1_lane0`,
`f1_champion_deepsets_lane1_v1`, `f1_compact_roster_prior_lane1`,
`f1_posterior_simulator_lane1`, `f1_roster_lad_v1`,
`causal_dynamic_thurstone_luce_v1`, and pinned TabPFN checkpoints
(`tabpfn_8_1_0`, `tabpfn_3_…`). Verify an artifact's manifest before
building on it; treat verified artifacts as capital — spend budget on what
they enable, not on rebuilding them.
