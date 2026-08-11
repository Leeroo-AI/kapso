# Wave-2 post-mortem — the five NOBANK re-runs, prediction-level forensics

Wave 2 (2026-08-09 → 08-11) re-ran eight classification tasks under the
migration-tier evaluation governance and the expanded context-handler
practices. Three re-runs beat their banked claims and were banked
(driver-top3 93.4→93.6, study-outcome 75.7→75.9, driver-dnf 82.7→83.3);
five finished below the banked test score and were rejected by the
keep-best gate (NOBANK), leaving the board unchanged. This document answers,
per NOBANK task: was the shortfall wrong modelling, wrong feature
engineering, wrong training construction, or noise?

Method: one-way local forensics, never fed back into any campaign. Both
prediction sets are scored on the official splits (row spaces verified
aligned — rel-stack tasks, whose task tables are resampled per cache build,
were instead compared through each campaign's own logs); disagreement is
measured by Spearman rank correlation; slices split test entities by
history; significance is a paired row bootstrap of the test-AUC delta
(3–4k draws). Mechanism findings come from reading each campaign's traces
and winning code. Peek scripts: session scratchpad `ur_compare.py`,
`ur_boot.py`, `nb_compare.py`.

| Task | Banked test (run, archive) | Wave-2 test | Paired Δ (banked−new) | Verdict |
|---|---|---|---|---|
| rel-event/user-repeat | 81.16 (run_0009, `20260730T223847_lane-a3`) | 79.81 (`20260810T160954_lane-c8`) | +1.36 ± 1.41, P=82% | real design gap, unresolvable at n=246 |
| rel-event/user-ignore | 88.89 (run_0007, `20260809T215615_lane-c2`) | 84.29 (`20260810T214652_lane-c2`) | +4.61 ± 0.65, P=100% | real regression — gate saved the board |
| rel-stack/user-badge | 89.29 (run_0013, `20260810T020148_lane-c4`) | 89.20 (`20260810T180405_lane-c4`) | within ~1 SE | selection noise; banked val 90.39 is a val-only artifact |
| rel-avito/user-visits | 67.78 (run_0020, `20260809T191905_lane-c5`) | 67.61 (`20260811T043108_lane-c5`) | +0.16 ± 0.13, P=90% | statistical twins; run froze on a flat gate |
| rel-stack/user-engagement | 91.51 (run_0005, `20260810T011603_lane-c5`) | 91.35 (`20260811T003120_lane-c4`) | 0.4 SE of a difference | selection variance; cross-lane ensemble never fired |

All archives under `gs://leeroo-kapso-relbench-artifacts/runs/<ds>--<task>/`;
banked evidence in `benchmarks/relbench/claims/<ds>--<task>/`.

## rel-event/user-repeat — training construction, decided on 77 cold rows

Banked val 77.31 / test 81.16; wave-2 val 74.74 / test 79.81. Predictions
correlate 0.923. On the 169 test entities seen in train the models tie
(83.84 vs 83.88); the whole gap lives on the 77 unseen entities
(75.35 vs 72.50). The banked champion earns that slice by construction:
a 7-phase temporal replay expands training to 29,338 rows, feeds 245
features, and blends four "borrow-strength" prior channels whose stacked
weight is 68% — cold entities inherit calibrated priors from related
populations. The wave-2 model trained on the raw 4,110 rows with 43
features, and its stacker drove the cold-expert weight to 0.

Verdict: the banked design is genuinely better for cold entities, but with
n=246 (single-model bootstrap SE 2.76) the +1.36 delta is P=82% — no 6h
re-run can even measure itself on this task. Next lever: nothing. The task's
test set cannot statistically reward further work; keep the banked artifact.

## rel-event/user-ignore — feature engineering for the cold slice; the gate's clearest win

Banked val 83.79 / test 88.89; wave-2 final val 83.29 / test 84.29 —
a real 4.6-point regression (CI +3.36..+5.91) that the keep-best gate
correctly refused to bank. Predictions correlate only 0.783. Seen-entity
slice: 83.95 vs 80.13; unseen (n=625): **98.38 vs 86.94**.

Mechanism: the banked champion is a two-expert mixture with a hard cold
router — `g = 1{hist_inv_total == 0}` (`graph_moe.py:938-942`) — whose cold
expert sees only history-free relational signal: 16-dim friendship-graph
SVD, Louvain-community empirical-Bayes invite rates, one- AND two-hop
neighbor invite-count propagation with quantiles, and blast-size context,
all rank-normalized within origin. Because "ignore" is driven by event
blasts hitting whole friend neighborhoods at once, those features nearly
determine zero-history users. The wave-2 model routes nothing and carries
only degree + friend_row_count for cold rows (its own cold-slice AUC 0.585).

Verdict: missing feature engineering (neighborhood invite-propagation
features), not modelling. Next lever if re-run: seed the banked champion's
cold-expert feature block as the starting design (see systemic finding 2).

## rel-stack/user-badge — selection noise on top of a val-only artifact

Banked run_0013: val 90.39 / test 89.29. Wave-2: shipped 89.20, best
archived candidate 89.43. Row spaces differ across cache builds, so the
comparison ran through each campaign's own logs: the banked run's 90.39 val
is concentrated on zero-history slices and is a val-only artifact — its
saved model is non-reproducible (checkpoint md5 mismatch, degenerate neural
head), and every test number in play (89.20 / 89.29 / 89.43) sits within
about one bootstrap SE. Verdict: pure selection noise; the board value is
fine, and nothing was actually lost. Next lever: none specific to badges —
this task is at its noise floor short of a genuinely new signal source.

## rel-avito/user-visits — twins separated by a frozen search

Banked val 71.33 / test 67.78; wave-2 val 71.06 / test 67.61. Spearman
0.960; seen/unseen slices near-identical (71.06/70.83 and 58.01/57.96);
paired delta +0.16 ± 0.13 — a coin-toss the banked side happened to win.
The interesting failure is *why* the re-run only tied: its accept gate was
a flat "keep if +0.005 val" rule that rejected 11 consecutive positive
candidate blocks, while the run's own clustered bootstrap — the correct
resampling unit — said several were real; the agent trusted the flat rule
and froze. This is the loss mode that produced context practice 5
(gates calibrated to measured clustered SE, target P(improvement)≈0.8).

Verdict: noise on the outcome, process defect in the search. Next lever:
re-run now that practice 5 ships in the handler context.

## rel-stack/user-engagement — max-val selection inside a 2-SE band

Banked run_0005 val 90.76 / test 91.51; wave-2 run_0013 val 90.82 / test
91.35. Both campaigns' bootstrap SE ≈ 0.0027, so the cross-campaign
difference SE is ≈ 0.0038: the test gap is 0.43 SE, the val gap 0.16 SE,
and each campaign's entire candidate ladder spans ~1.7 SE. The banked
"GraphSAGE+GBDT" headline is itself val-negative in its own log (GBDT-only
0.908004 vs shipped 80/20 blend 0.907609) — the 91.51 was a GBDT result
plus a favorable coin-flip. Verdict: construction/selection variance, not
features or modelling.

The one real lever either campaign surfaced and neither exercised: the
cross-lane ensemble. Wave-2's four lane finals (0.90565 / 0.90584 /
0.90628 / 0.90820) were only max-selected, never rank-averaged, despite
being far less correlated (0.922–0.943) than the within-lane pairs that
*were* blended (0.981–0.993); and its semantic-text lane pays 3–4× more on
the dormant-user mass (deltas +0.0052/+0.0060/+0.0041) than overall. A
forward-fold-gated rank average of trajectory-GBDT × GraphSAGE × semantic
dormant specialist is the credible route past 92.

## Cross-cutting findings

1. **Gate mis-calibration was the largest self-inflicted loss** (user-visits
   frozen, user-repeat's cold expert zeroed by an uncalibrated stacker
   threshold). Fixed: context practice 5 (`context.py`, commit ba0652ce).
2. **Re-runs rediscover from scratch.** user-ignore's 12h run never found
   the neighborhood-propagation features its own banked champion already
   proved. Proposal (approval pending): seed the banked champion into the
   run archive as a stamped baseline so re-runs start at the frontier.
3. **Max-val selection inside the noise band wastes decorrelated lanes**
   (engagement, badge). The generic remedy is already implied by practice
   5's SE framing; an explicit "ensemble decorrelated finalists before
   max-selecting" practice is a candidate next context addition.
4. **Tiny-test tasks cannot pay for re-runs.** user-repeat's n=246 gives
   SE ≈ 2.8; no plausible 6h improvement is measurable. Task selection
   should screen on test-set resolving power, not just headline gap.
5. **The keep-best gate performed exactly as designed** across all five:
   one true regression blocked (ignore), two coin-tosses kept at the better
   banked value (visits, engagement), two noise cases kept (badge, repeat).
   Every wave-2 trace is still archived in GCS for mining.
6. **Bookkeeping: the scorecard regenerates RESULTS.md from box-local
   `tmp/relbench/` only.** NOBANK quarantines emptied three task dirs, and
   the next harvest's regen silently flipped their banked rows back to
   "pending" (repaired in a6ce8d6d; watcher now restores banked generator
   inputs after quarantine). Design fix worth taking: make committed
   `claims/` the generator's source of truth for done rows, with `tmp/`
   used only for running-status — then a regen can never lose a banked row.
