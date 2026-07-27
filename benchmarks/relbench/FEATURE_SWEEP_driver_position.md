# Feature-engineering sweep audit — rel-f1 / driver-position

Purpose: per-iteration record of which features each experimentation run actually
used, to judge whether the tabular feature-engineering space was swept properly.
Compiled 2026-07-27 from local artifact pulls of the two generic-search campaigns:

- **R9** (10h codex-primary campaign, 2026-07-26 → 27): `gpu_results_run3/runs/run_0001..run_0008`
- **run4** (K=2 campaign on `relbench-dp-gpu-0727` / `0727b`, 2026-07-27): `gpu_results_run4/runs/run_0001..run_0011`

Sources per run: `code/PLAN.md` (idea), `code/changes.log` (what was built and
measured — outcomes below are quoted from these logs, not inferred), the feature
code itself (`f1_features.py`, lane modules), and `manifest.txt` (official val
MAE). changes.log files are cumulative per lineage; per-run facts were extracted
by diffing consecutive logs.

---

## R9 per-run feature record

| run | val MAE | lane idea | feature families used / newly introduced | outcome |
|---|---|---|---|---|
| run_0001 | 2.7133 | Contextual hybrid baseline | NEW: Pool-A/B identity+label recovery; **causal dynamic Plackett–Luce skill filter** (`rank_filter.py`: driver/constructor skill + reliability states); fresh vs stale L1 LightGBMs; dedicated cold-driver model; context/gap MAE stacking; cohort clipping. Tabular features (`models.py`, `identity.py`): age (`driver_age_years`, dob), `latest_constructor_id`, `prior_starts`, seed month/year, cohort size/cold count/fraction, `information_gap_days`, **`nationality_frequency`**, `prior_seed_appearances`, `prior_seed_gap_days`, `has_no_results` | first champion |
| run_0002 | 2.7097 | Staged cohort-ranking ensemble | NEW: **16-source opponent-relative cohort features** (`cohort_features.py`): position means last3/last5, day-windows 60/180, **`position_ewma_090`**, qualifying position+percentile last5, grid last5, `driver_teammate_finish_delta`, `teammate_finish_pct_recent`, standings position + **`standings_leader_share`**, constructor finish-pct/points/reliability last20, context inactivity / information gap / uncertainty / cold fraction / known-constructor count. Models: LambdaRank + percentile LightGBMs, exact-pairwise/soft-rank **DeepSets**, prequential projection+stack | champion (projection accepted at fresh-warm weight 0.142; LightGBM/DeepSets challengers failed historical gates) |
| run_0003 | 2.6873 | Frozen alpha-0.05 projection pair | No new features — banked the projection Model-A/B pair (`projection_bank.py`), gap-knot interpolation (projection only at gap 0, 365+ day knots pinned to champion) | champion (banked) |
| run_0004 | 2.6347 | LambdaRank + rank transport | NEW: three-seed raw LambdaRank final bundles; **27-knot monotone L1 conditional-median rank transport** (2 cohort-size buckets, 200 pseudo-observations); prequential grid selection (alpha 0.15, transport 0.20). Hurdle probe (appearance-count, classified-status, classified-pace quantiles, non-classified-order quantiles, correlated simulation) **failed expansion gates by −0.0453 MAE** and was dropped | champion |
| run_0005 | 2.6347 | Cache verification | No new features (byte-pinned fallback reconstruction fix; re-registration) | champion (re-registered) |
| run_0006 | 2.6329 | rank_fresh pair | No new features — fresh-production separation, byte-identity assertions (`fresh_producer.py`) | champion |
| run_0007 | 2.6260 | Honest fold-k−1 policy | No new features — genuine fold-k−1 selection (alpha 0.20 / transport 0.25, 96.08% positive bootstrap). Rejected on their untouched-fold gates: percentile rank averaging, independent rank bands, value blender (+0.0021 only), matched cohort/era **zero-history prior**, causal **previous-season race-density bucket** | champion |
| run_0008 | 2.6260 | Pseudo-freeze horizon calibration | NEW: **causal pseudo-freeze views at 120/180/240 days** (`pseudo_freeze.py`) + state-specific non-increasing horizon weights across 9 knots; teammate form re-derived from constructor trailing-40 window. Stale evidence +0.0158 pooled but **violated the per-knot degradation gate** → test rows anchored at champion | final champion (test 3.5631 / NMAE 0.5072) |

## run4 per-run feature record

`f1_features.py` is **byte-identical (md5 10288475) for run_0003 through run_0011** —
the registered tabular feature matrix froze at the run_0003 champion; every later
lane added mechanism-level signals on top and fell back to the incumbent when its
gates failed. All parity rows below registered val 2.7969464699733613.

| run | val MAE | lane idea | feature families used / newly introduced | outcome |
|---|---|---|---|---|
| run_0001 | 3.0208 | Baseline episodes + GBDT (577-line `f1_features.py`) | NEW: exact 60-day label reproduction; **annual frozen pseudo-episodes** (staleness in the training distribution); roster/seed context (`roster_size`, `prior_seed_count/fraction`, `days_since_first_seed`, `seed_history_missing`); driver snapshots (`no_history`, `history_starts`, `career_length_days`, `last_constructor`, `last{N}_position_{mean,median,std,ewm}` + missing flags, `birth_date`); standings + constructor standings (position/points/wins + missing flags); aggregate history means over positionOrder, points, wins, podiums, DNFs, grid, qualifying, **`finish_minus_grid`**, **`finish_percentile`**, **`qual_percentile`**, **`teammate_residual`**; Model A/B split, 3-seed LightGBM L1, inverse repeated-label + era weighting, roster clipping | baseline champion |
| run_0002 | 3.4174 | Pairwise/rank lane | NEW: **regularized Elo** (`elo`, `elo_after`), EW race-normalized finishes, leave-one-driver-out constructor features, `career_*` block (dnf/podium/wins rates, points sums+logs, starts+log, finish/grid/qualifying percentiles, teammate residual), `constructor_tier`, **`constructor_tenure_years`**, `constructor_count`, `nationality`, novelty/roster history, ordered weighted pair expansion → binary pair LightGBM + median-quantile calibrator with rank-only monotone projection | worse (uncapped-core gate failed at frozen-origin 3.504/3.590; direct val 3.4174) |
| run_0003 | **2.7970** | Champion surgery (577→624 lines) | NEW in the frozen matrix: **roster-relative block** (`roster_rank_{col}`, `roster_delta_{col}`, `roster_known_count`, `roster_no_history_fraction`, `roster_returning_fraction`), **`projected_{col}`** extrapolation features, `returning_driver`, career/365/730 finish- and qual-percentile means, `career_teammate_residual_mean`, `constructor_pace_365/730` (+last20), `last5/10/20_position_mean`, `days_since_last_result`, `age_years` | champion for the rest of the run |
| run_0004 | parity | Ordinal CDF + assignment | Lane signals: legal **future-race-count supervision**, historical roster/race-count templates, monotone ordinal LightGBM CDFs, expected-L1 **Hungarian assignment**, regime/horizon simplex blends. Measured: template projection **+0.439 MAE worse**; ordinal pilot −0.0021 pair accuracy → stop rule | parity fallback |
| run_0005 | parity | Auxiliary-target value stack | Lane signals: chronological **auxiliary labels** (future starts, race count, entered fraction, DNF rate, finish variance, points/start); absolute / roster-normalized / roster-residual value members; regime specialists; start-weighted **conservation** anchors; regime-horizon constrained blends. Stage one +0.1226 dev, but hard-origin −0.0412 + regime/rehearsal/bootstrap gates failed | parity fallback |
| run_0006 | parity | Donor-world retrieval | Lane signals: 1,674 historical **donor worlds** (46,379 frozen roles, 123,926 race-role outcomes); deployment-observable world descriptors (age/career quartiles, constructor concentration/tier/top-fraction, completed-lap fraction); cohort-constrained Hungarian matching + complete-sequence transport + pace rerank. Scout failed both continuation gates | parity fallback |
| run_0007 | parity | Causal race-world MC simulator | Lane signals: race-count/field-size/participation/**seat-tier**/reliability/classified-pace-quantile/DNF heads + coherent particle simulation over union rosters. Scout: simulator mean **5.5722 vs incumbent 4.9628**, no-history −0.4777 → lane closed | parity fallback |
| run_0008 | parity | OU/Kalman dynamic latent state | Lane signals: probit observations, age/era baselines, irregular-time **OU Kalman filter/smoother** (driver + constructor backfitting, season shocks), **debut/survival/return priors**, anonymous constructor tiers + legal transitions, Poisson-binomial race ranks, ≤12-feature depth-3 L1 residual. Dev +0.1011, no-history +0.1199, but hard mean −0.0204 / hard worst −0.0530 / rehearsal −0.0439 → promotion failed | parity fallback |
| run_0009 | parity | Discriminative race-slot twin | Lane signals: multiclass race-count, binary participation, field-moment, direct L1 + **listwise LambdaRank** heads on frozen race worlds; max-entropy pattern enumeration; roster-aware pairwise rank reconstruction; weighted-median decoding; moment-correct conservation. Dev +0.1235 / no-history +0.2915 (bootstrap 0.999) opened the hard pass, but hard mean −0.0621 / worst −0.0111 / rehearsal −0.0543 → failed | parity fallback |
| run_0010 | parity | **TabPFN-v2 in-context** | Lane signals: Prior-Labs TabPFN-v2 regression checkpoint (SHA-256 documented), chronology-censored context dedup + weighted sampling, regime/horizon context quotas (`context_regime`, `context_band`, `context_weight`), raw + roster-normalized median heads, LOOO specialist gates, nonnegative LAD blending. **Four-origin pilot failed mean and no-history gates → zero TabPFN weight** | parity fallback |
| run_0011 | parity | Causal event SSL | Lane signals: 3-layer event SSL with 35% grouped outcome masking, 60/180/365-day hazards, ListMLE + pairwise order loss, 8 **cohort prototypes**, constructor decay/dropout, roster attention. Pilot −0.3449 mean / −0.4161 no-history; corrected retry −0.4404/−0.4986 → abandoned | parity fallback |

---

## Feature families swept (verified in code, with introducing run)

1. **Rolling form, count windows** — last3/5 (R9 run_0002), last5/10/20 (run4 run_0003), last{N} mean/median/std (run4 run_0001).
2. **Rolling form, day windows** — 60/180d (R9 run_0002), 180/365/730d + career (run4 run_0001/0003, run_0009 twin blocks).
3. **Exponential decay** — `position_ewma_090` (R9 run_0002), `last{N}_..._ewm` (run4 run_0001), `ew_finish_percentile` (run4 run_0002).
4. **Era-invariant percentile transforms** — finish/qual/grid percentiles of field (both campaigns, run4 run_0001; R9 run_0002).
5. **Race-craft delta** — `finish_minus_grid` (run4 run_0001).
6. **Teammate isolation** — `driver_teammate_finish_delta` (R9 run_0002), `teammate_residual` + career mean (run4 run_0001/0003); teammate-form derivation hardened in R9 run_0008.
7. **Standings snapshots** — driver standings position/points/wins, `standings_leader_share` (R9 run_0002; run4 run_0001).
8. **Constructor strength** — pace/points/reliability windows (R9 run_0002 last20; run4 constructor_pace_365/730/last20), `constructor_tier`, tenure (run4 run_0002).
9. **Roster / seed context** — roster size, prior seed count/fraction, days-since-first-seed (run4 run_0001); roster-relative ranks/deltas/known-fraction (run4 run_0003); cohort cold counts (R9 run_0001).
10. **Staleness / horizon machinery** — annual frozen pseudo-episodes + `horizon_days` (run4 run_0001); `projected_{col}` (run4 run_0003); R9 pseudo-freeze 120/180/240d + 9-knot horizon weights (run_0008); projection gap-knots (R9 run_0002).
11. **Aging** — birth date/age (both), age quartile world descriptors (run4 run_0006).
12. **DNF / reliability** — dnf rates/means (both), reliability heads (run4 run_0007/0008), hurdle status heads (R9 run_0004, gated out).
13. **Qualifying/grid aggregates** — position + percentile forms (both).
14. **Pairwise / rank encodings** — LambdaRank (R9 run_0002/0004; run4 run_0009), pairwise BCE + soft-rank DeepSets (R9 run_0002), ordered pair expansion (run4 run_0002), ordinal CDFs (run4 run_0004).
15. **Latent ratings** — dynamic Plackett–Luce skill filter (R9 run_0001), regularized Elo (run4 run_0002), OU/Kalman latent states (run4 run_0008).
16. **Analog/donor retrieval descriptors** — donor worlds + world descriptors (run4 run_0006).
17. **World-model / simulation heads** — participation, seat-tier, field-size, pace quantiles (run4 run_0007/0009).
18. **Pretrained in-context columns** — TabPFN context curation (run4 run_0010).
19. **SSL embeddings** — masked-event encoder + cohort prototypes (run4 run_0011).
20. **Demographics** — `nationality_frequency` (R9 run_0001), nationality (run4 run_0002/0011).
21. **Auxiliary future-facing supervision** — future starts / race count / DNF rate / points-per-start as aux targets (run4 run_0005).
22. **Conservation / field-size anchors** — roster clipping (both), sum-conservation (run4 run_0005/0009).

## Never tried (grep-verified across all 19 run code dirs)

1. **Explicit trend/slope features** — NEVER. No `slope`/`polyfit`/momentum/window-delta
   (e.g. last3_mean − last10_mean) anywhere. Trend is only *implicitly* reachable by
   the GBDT combining window levels + EWM.
2. **Circuits relation** — NEVER. Zero references to `circuit` in any run's code:
   the `circuits` table / `races.circuitId` join was never read in either campaign.
   No driver×circuit, constructor×circuit, or circuit-era feature exists.
3. **Interaction/cross features** — NEVER as engineered features (no explicit
   products/ratio crosses beyond the percentile normalizations; trees left to
   find interactions).
4. **Target / leave-one-out encodings** — NEVER (identifier columns were dropped
   instead, e.g. the TabPFN lane drops `driverId`).
5. **Season-phase features** (round within season, races-into-season) — NEVER.
   The only `round` tokens are the Python builtin.
6. **Head-to-head matrices** — NEVER.
7. **Career-phase relative-to-own-peak** — NEVER (age curves exist; own-peak
   normalization does not).
8. **Seat-change dynamics** — MOSTLY NOT: `last_constructor` joins, tenure and
   constructor-count exist (run4 run_0002); no change-recency/upgrade-downgrade
   flags anywhere (R9's `changed_constructors` is loop bookkeeping, not a feature).
9. **Field-strength-adjusted opponent quality** — TRIED in roster/cohort-relative
   form (R9 16-source opponent-relative features; run4 roster_rank/delta block);
   not as opponent-strength-weighted historical aggregates.
10. **Un-mined raw columns** — no code touches pit stop, lap-time, or qualifying
    session-time (q1/q2/q3) detail; `laps` and `statusId` are used (R9
    rank_filter, run4 race worlds).

## Verdict

**The driver-form tabular space was swept properly; the relational breadth was not.**
Across 19 runs the campaigns covered windows × decay × percentile-normalization ×
teammate isolation × standings × constructor strength × roster context ×
staleness — plus five distinct latent-rating/mechanism families and two
pretrained families, every one measured against frozen-origin gates. That is a
genuinely thorough sweep of *driver-history* features.

Two structural limits are visible in the record:

- **The feature file froze the moment a champion emerged.** In run4,
  `f1_features.py` is byte-identical from run_0003 to run_0011; all later budget
  went to decoders/latent states/world models/pretraining, which reused the
  frozen matrix and fell back on gate failure. No iteration after run_0003
  attempted to widen the tabular matrix itself.
- **The sweep never left the results/standings/qualifying tables.** The circuits
  relation was never joined; season-phase, head-to-head, and trend/delta
  transforms were never generated.

Highest-value untried directions, grounded in the campaigns' own measurements
(no-history cohort ≈ 4.05 MAE vs ≈ 2.8 established, run4 diagnostics; champion
2.80 val → ≈ 4.96 on test-like frozen origins):

1. **Trend/slope + window-delta features** — cheap columns on the existing
   matrix; the model currently must reconstruct trajectory from level pairs;
   directly testable under the existing frozen-origin gates.
2. **Circuit-composition features** — the only unread DB table; per-driver/
   constructor circuit affinity and per-circuit chaos (DNF rate, field spread)
   are structural signals that survive multi-year staleness better than form.
3. **Season-phase features** — a standings snapshot in month 3 means something
   different from month 10; one integer column (races-into-season) disambiguates
   the campaign's single most-used snapshot family.
4. **Simple cold-cohort tabular priors** — debut-age × era × constructor-tier
   cell means as plain columns; every mechanism-level attack on the 43%
   no-history mass (prototypes, survival priors, TabPFN quotas) failed its
   gates, while the champion's only cold signals remain roster/seed context.
5. **Seat-change dynamics** — change-recency and tier-delta flags; the frozen
   2009 state makes *last known seat* stale for exactly the drivers the test
   weights most.

Uncertainty notes: feature semantics are derived from identifier names plus
changes.log statements; R9 lane internals (e.g. exact 16 sources of the cohort
features) were not re-derived line-by-line. Whether the sanitized cache retains
qualifying q1/q2/q3 columns was not checked — the claim above is only that no
code references them.
