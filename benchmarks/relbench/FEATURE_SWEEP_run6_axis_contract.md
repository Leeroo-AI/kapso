# Feature audit — 0727d axis-contract campaign (run6), rel-f1/driver-position

Per-experiment feature documentation for all 20 registered runs of the
axis-contract campaign (box relbench-dp-gpu-0727d, 2026-07-27→28, framework
commit 277781d2). Source: `gpu_run6ax_final_pull.tgz` → `runs/run_0001..0020`
(code + manifest val scores + changes.log). Deltas are from byte-level diffs
of the feature-constructing modules (md5 identity map), not from prose.
UNCOMMITTED working document (user-directed).

Official final: run_0018 champion, val 2.6486 → test 3.6259 (NMAE 0.5161).

## Campaign shape

Two K=2 lanes per iteration; long codex sessions registered multiple full
evals each, so run ids ≠ iterations. Lane families: **StateSet** (event-token
neural roster model over a causal skill filter, lane of runs 1–6, 8) and
**StaleRank** (tabular GBDT over a relational FeatureStore, run 7), fused
from run 9 onward into a **residual stack**, extended by chainset /
direct-chainset / field-relative modules (runs 13–20).

val trajectory: 2.8119 → 2.7025 → 2.6781 → 2.6670 → 2.6619 → **2.6486**.

## Per-run record

**run_0001 — val 2.8119 (StateSet v1 seed, iter 1 lane 1).**
Modules: `stateset_data.py` (861 L), `stateset_model.py`, `main.py`.
Representation: per-driver token vector of **42 channels** = 33 base + 9
within-roster percentile ranks, consumed by a permutation-equivariant
two-block roster Transformer (median + ordering + roster-shape losses
0.55/0.30/0.15). Base channels (from the `raw.append` block, order
preserved): `driver_pace`, `driver_pace_sd`, `driver_reliability`,
`driver_reliability_sd`, `driver_qual`, `driver_qual_sd`,
`driver_finish_rate`, `driver_grid_pct`, `constructor_pace`,
`constructor_pace_sd`, `constructor_reliability`,
`constructor_reliability_sd`, `expected_seat` (3-tier transition prior ⋅
tier strengths), `seat_sd`, `seat_probability_weak/mid/strong`, `age_z`,
`age_z²`, `log_career_starts`, `observed_flag`, `unobserved_flag`,
`gap_months` (freeze→seed gap), `last_staleness` (months since last
result), `roster_size`, `log_seed_appearances`, `nationality_sin`,
`nationality_cos`, `last_tier`, `uncertainty` (driver+constructor+seat sd),
`constructor_count`, `age_bucket`, `gap_bucket`; ranks over
[pace, reliability, qual, finish, constructor_pace, expected_seat, career,
age, uncertainty]. All channels derive from a forward-only
`CausalSkillFilter` (Kalman-style states {pace, var, rel, rel_var, qual,
qual_var, finish, grid, career, last_const, last_tier, last_ns} per driver,
{pace, var, rel, rel_var, count} per constructor, exponential half-life
decay + process noise, seat-transition/debut event priors), trained on
episodes with artificial freezes (8/21/42/67/84-month) and first-race
masking. Outcome: seed baseline.

**run_0002 / run_0003 — val 2.7025 (StateSet v2, iter-1 lanes; 0003 is the
twin registration).** `stateset_data.py` 5a64eda6 (+187 L). DELTA vs r1:
**+11 channels** — `recent_position`, `recent_grid`, `log_recent_points`,
`recent_completion` (decayed rolling recent-form state added to the
filter), `driver_standing_position`, `driver_standing_points`,
`driver_standing_wins`, `constructor_standing_position`,
`constructor_standing_points`, `constructor_standing_wins`,
`standing_staleness` (months since last standings row); driver- and
constructor-standings histories ingested into the filter; rank set extended
(adds recent_position, …). Outcome: −0.109 val, new champion.

**run_0004 — val 2.6781 (StateSet v3).** 7d00e513. DELTA: added
`rookie_performance_prior(...)` — cohort-estimated rookie prior INJECTED
into filter state for unobserved drivers (pace=1.2·(prior−0.5),
finish/grid=prior, variance bump) and a `+0.40·(rookie_prior−0.5)` term in
the rating score; added keyed roster caching. No new channels. Outcome:
−0.024, new champion.

**run_0005 — val 2.6797 (StateSet v5).** f2f94a69. DELTA: **removed** the
r4 in-state rookie injection; **added channel 46** `seed_tenure_months`
(months since driver's first seed appearance, from `_seed_tenure_months`
transform) and its rank. Outcome: marginally worse than r4 solo.

**run_0006 — val 2.6670 (StateSet v6) — mid-campaign champion.** 63202377:
byte-identical features to r5 except cache version bump; the gain came
from model/main side (two-seed fitting, timestamp-level OOF blending,
promotion gates). This 1139-line `stateset_data.py` (47-channel vector:
33 + 11 + seed_tenure + extended ranks ≈ 47 with rank block) is the
representation frozen into ALL later runs (r8, r11–r20).

**run_0007 — val 2.7765 (StaleRank tabular lane, iter 1 lane 2).**
`features.py` 121b113c (771 L; becomes `stalerank_features.py` verbatim in
r11+), `labels.py`, `models.py` (LightGBM), `cache_utils.py`. A ~110-column
relational tabular family, all snapshot-censored (`relational_v3`),
including: identity/tenure (`career_races`, `career_seasons`,
`days_since_debut`, `age_at_debut`, `age_band`, `age_snapshot`,
`constructor_tenure_days/races`, `constructor_switch_count`); result
windows (`result_position_mean_60/90/120/365d/all`,
`result_finish_percentile_mean_30/60/90/120/180/365d`,
`position_ewma_90/365d`, `finish_percentile_ewma_90/365/730d`,
`position_median/q25/q75/std`, `grid_mean`, `laps_mean`, `points_mean`,
`podium_mean`, `win_mean`, `dnf_mean`, `result_count_365d`,
`result_days_since_last`); **explicit trend features** via `_slope()` —
`result_velocity_30_365`, `result_velocity_90_365`,
`result_velocity_365_career`; teammate isolation (`teammate_diff`,
`teammate_diff_mean`, `teammate_mean`); standings (`standing_latest_position`,
`standing_position_change`, `standing_days_since_last`) and constructor
standings (`constructor_standing_latest_position`, `…_days_since_last`,
`constructor_official_points_latest`, `constructor_percentile_mean_90/365d`,
`constructor_position_mean_365d`); qualifying
(`qualifying_latest_position`, `qualifying_days_since_last`); **cohort
target-priors** (`age_band_position_prior`, `age_band_prior_count`,
`nationality_position_prior`, `nationality_prior_count`); interactions
(`constructor_strength_gap_interaction`); season phase (`seed_month_sin`,
`seed_month_cos`, `seed_year`, `snapshot_year`,
`season_current/previous_finish_percentile_mean`); seed-appearance
dynamics (`seed_appearance_count/fraction/log/rate`,
`seed_days_since_first/previous_appearance`,
`seed_first_appearance_after_snapshot_days`, `seed_opportunity_count`);
roster context (`roster_size`, `roster_size_bucket`, `roster_cold_fraction`,
`roster_observed_count`, `field_size`); gap encodings (`gap_days`,
`gap_months`, `gap_bucket`, `gap_4_12`, `gap_13_30`, `gap_31_54`,
`gap_55_80`); **driver-id ordinal debut proxies** (`driver_id_log`,
`driver_id_snapshot_offset/ratio`, `raw_driver_id`,
`maximum_observed_driver_id`); per-source missingness flags
(`result/qualifying/standing/constructor_source_missing`,
`source_missing_pattern`, `cold_start`). Outcome: weakest val of iter 1 but
(per campaign frozen-scout evidence) the best-transferring lane; kept as
blend member.

**run_0008 — val 2.6670 (parity; posterior-simulator attempt).** Adds
`stateset_replay.py` + `stateset_simulator.py` (1386 L): Monte-Carlo
posterior simulator over the filter states with stale-bucket weights
(`stale_13_30/31_54/55_80/over_80`), age_stage/career/tier conditioning.
changes.log: mis-measured first gate (generic 3.8594 vs the named 3.4992
long-gap rehearsal); corrected evaluation 3.5237 → failed continuation;
champion re-registered byte-identical. Feature file unchanged (63202377).

**run_0009 — val 2.6845 (first fused stack, iter ~3).** New modules:
`stack_features.py` a2cde03f, `stack_bases.py`, `stack_models.py`,
`stack_cache/manifest.py`; stalerank_* files vendored in
(`stalerank_features.py` fe318a77 variant, version
`relational_v4_no_future_seed`). The stack frame fuses: StateSet filter
channels (driver/constructor pace/sd/reliability/qual, seat tier
probabilities, recent_*, standings, nationality trig, staleness/gap
scalings) + StaleRank domain features (prefixed) + **lane meta-features**:
`base_mean`, `base_difference`, `lane_disagreement`, `opponent_residual`,
`opponent_residual_ewma_3/6/12`, `rookie_prior`, `rookie_prior_sd`,
`single_car`, `single_car_fraction`, `stalerank_missing_count`,
`manifest_regime/gap_bucket/gap_months`. LEAK FIX recorded: r9's variant
zeroed `appearances` and `seed_tenure` channels (1f8cb143 sets
`appearances = 0`, tenure→0.0) because querying seed counts at target date
"would expose post-snapshot test appearances". Outcome: 2.6845 — stack
below champion; also fixed a roster-size float32 rounding bug.

**run_0010 — val 2.6746 (StaleRank-side refit of the stack).**
`stack_bases.py` 06662c16 (expert weighting {fresh 1.0, default 0.7}).
Outcome: better but still short of 2.6670; all residual variants rejected
by promotion gates.

**run_0011 — val 2.6619 — NEW CHAMPION (matched fusion).** stalerank
features reverted to r7 semantics (121b113c, `relational_v3`; appearance
counts queried at snapshot — the legal form), stateset back to 63202377
(seed-tenure channel restored), `stack_features` bumped to
`stateset_fused_v3_matched` / `domain_fused_v3_matched`,
`stack_bases` b062e610 (SHA-pinned prior-final predictions
`stateset_direct` + `stalerank_final` as base columns). The champion is now
a **residual stack over the two lanes** with matched appearance semantics.
Outcome: −0.005 vs r6.

**run_0012 — val 2.6619 (parity; TabPFN lane).** Adds `pfn_lane.py`
(1263 L): TabPFN-3 checkpoint (sha e923ba9…) fed a reduced tabular view —
`history_length/log`, `staleness*`, `gap_bucket`, `roster_size`,
`month_sin/cos`, `nationality_sin/cos`, `seed_year`, `snapshot_year`,
`single_car_fraction`, `raw_driver_id`, lane meta features
(`base_mean/difference`, `lane_disagreement`, `opponent_residual_ewma_*`) —
producing `pfn_q10/q50/q90` quantile members. Gate result (changes.log):
q50 MAE 6.1868 vs StaleRank 6.0112 on the precommitted held era → stop
rule rejected expansion; **PFN never entered any registered prediction
path** (r13+ mains carry only manifest/config mentions; no pfn import).

**run_0013 / run_0014 — val 2.6619 (parity; ChainSet donor lane).** Adds
`chainset_data.py` 76b66fed (1249 L), `chainset_model.py`,
`chainset_pipeline.py`: no-identity "chain" episodes with donor cohorts —
features `pseudo_pace`, `pseudo_qual`, `complement_position`,
`complement_position_sd`, `survival_prior`, `survival_prior_sd`,
`tenure_months`, `cohort`/`regime`/`gap_bucket` keys, long-gap gain
diagnostics (`long_31_80_*`). Gate: calibrated Model-B MAE 3.2283 / blend
3.2256 vs strict 3.20 threshold → champion test lane retained.

**run_0015 — val 2.6619 (parity; direct ChainSet).** Adds
`direct_chainset_data.py` (684267ab) + model + pipeline: the widest
tabular frame of the campaign — `state_000..state_046` (full StateSet
token), `roster_{signal}_{stat}` roster summaries,
`relative_{signal}_minus_mean` / `_robust_z` / `_percentile`
(field-relative re-encodings), `relation_*` (entire StaleRank domain
frame), `missing_*` indicators, plus OOF auxiliary heads
(`oof_dnf_probability`, `oof_dnf_variance`, `oof_completion_fraction`,
`oof_reliability_interaction`), `complement_dispersion/uncertainty`,
`veteran_slot_occupancy`, `teammate_transition`, `survival`,
`seed_appearances`, `seed_tenure_months`, `time_since_seed_months`,
`stateset_stalerank_disagreement`. Outcome: parity (stage rate 0.88
artifacts/min noted; keyed stage banking added).

**run_0016 — val 2.6619 (parity; synthetic-prior compact network).** Adds
`compact_roster_model.py` (`compact_context_roster_q50_v8`: context/query
token sets over roster tensors, GELU compact net, 100k generated
universes) + reduced `chainset_data.py` (insurance passthrough) +
simulator variant. Gates: synthetic gate passed, **zero-shot real gate
failed**; DNF proxy discrepancy (52.77% vs prescribed 5–40%) recorded.

**run_0017 — val 2.6619 (parity; DeepSets graft).** Adds
`deepsets_model.py` + `deepsets_pipeline.py`: grafts the INHERITED-shape
DeepSets cohort ranker (frozen_pairs episodes, permutation gates,
roster-size strata `roster_le20/21_23/ge24`, alpha blend search) onto the
stack. Gate: OOF shared correction gained 1.2847 MAE on reconstructed
anchor but **regressed 0.9954 on the frozen official anchor** →
persist-first gates retained champion.

**run_0018 — val 2.6486 — FINAL CHAMPION (field-relative ChainSet).** Adds
`field_relative_chainset.py` (1958 L). Mechanism: takes the
direct-chainset feature frame (via `materialize_features`), converts
targets and predictions into **field-relative space** (`f0_projection` /
`normalized_roster`, `incumbent_level_projection`), fits LightGBM
(`objective`, `num_leaves`, `lambda_l1/l2`, `max_bin`, bagging/feature
fractions — full param grid in-file) on field-relative residuals, applies
**cold empirical-Bayes shrinkage** (`cold_empirical_bayes`,
`cold_shrinkage`, `cold_cap`, age buckets `age_lt25/25_29/ge30/unknown`,
appearance buckets `app_0/1_5/ge6`, gap buckets `gap_31_54/55_80/
beyond_80`), and selects a **fixed convex blend** with the incumbent by
**leave-one-origin-out** over four multi-decade forward origins
(1991→92-95, 1995→96-99, 1999→00-04, + later), with bootstrap-90 gates
(`bootstrap_actual_mae`, `blend_optimum_range`) and byte-identical
fallback. LIVE prediction path per `main.py` imports:
`field_relative_chainset` + `stateset_data` + `stalerank_features/labels`
+ `stack_bases (fit_predict_stateset, fit_predict_stalerank)` +
`stack_features` + `stack_models (fit_residual_stack)` + caches. Dead
weight in code dir: chainset_/direct_chainset pipelines (imported only via
direct_chainset_data for features + chainset_data.preserve_champion),
deepsets absent, pfn absent. Outcome: −0.0133, final champion.

**run_0019 — val 2.6486 (parity; roster-LAD).** Adds `roster_lad.py`
(1557 L): convex (cvxpy/Clarabel) roster-level location-scale correction —
era shift/scale parameters with `era_penalty`, `adjacent_penalty`,
disagreement features (`disagreement_mean/iqr`), **interaction feature
`cold_fraction_x_log_staleness`**, `dnf_probability/variance`,
`completion_fraction`, F0 quantile summaries (`f0_mean/q25/iqr/mae`).
Gate failed → `fallback_byte_identical`.

**run_0020 — val 2.6486 (parity; causal Thurstone–Luce).** Adds
`causal_thurstone_luce.py` (2065 L): native causal TrueSkill-style rating
lane — `driver_strength`, `driver_uncertainty`, `constructor_strength`,
`constructor_uncertainty`, `rating_age`, discounted Beta-binomial
reliability, rookie priors, R0 pairwise ranks, two-seed 16-unit supervised
residual block, **Poisson-binomial q50 decode**, alpha grid. Gates
(`cold_gain`, `long_stale_gain`, `candidate_spearman` vs
`incumbent_spearman`, bootstrap-90) not passed → F0 byte-identical
fallback. Campaign ends; champion = run_0018.

## Feature genealogy (compressed)

r1 StateSet 42-ch causal-filter tokens → r2 +recent-form & championship
standings (+11 ch) → r4 +rookie-prior state injection → r5 swaps injection
for seed-tenure channel (47-ch final form, frozen thereafter) → r7
parallel ~110-col relational tabular family (windows, EWMAs, velocities,
teammate, cohort priors, id-ordinals, missingness) → r9 fusion frame
(lanes + meta-features; leak-fix zeroes appearance channels) → r11 matched
fusion (leak-fix done right: snapshot-queried appearances) = residual
stack champion → r12 PFN view (rejected) → r13 chain/donor episodes
(rejected) → r15 direct frame = union(StateSet tokens, roster stats,
field-relative re-encodings, relation_*, missing_*, OOF heads) → r16
synthetic-universe compact net (rejected) → r17 DeepSets graft (rejected)
→ **r18 field-relative projection + cold empirical-Bayes over the r15
frame = champion** → r19 LAD decode / r20 Thurstone–Luce decode (both
gate-rejected).

## Consolidated feature union (family → members, introducing run)

- **Event-token/filter channels (r1, r2, r5):** the 47-channel set listed
  under r1/r2/r5 (pace/reliability/qual states ± sd for driver and
  constructor, seat-transition probabilities, age terms, career, gap and
  staleness, roster size, appearances, nationality trig, tier,
  uncertainty, recent_position/grid/points/completion, driver and
  constructor standings position/points/wins, standing staleness,
  seed_tenure_months, within-roster percentile ranks).
- **Relational tabular (r7):** ~110 columns as enumerated in run_0007 —
  multi-window means/EWMAs/percentiles, velocities (explicit slopes),
  teammate residuals, standings deltas, qualifying, cohort target-priors,
  interaction, month-of-year phase, seed-appearance dynamics, roster
  context, gap encodings, driver-id ordinals, per-source missingness.
- **Lane meta-features (r9):** base_mean/difference, lane_disagreement,
  opponent_residual + EWMAs 3/6/12, rookie_prior(_sd), single_car(_fraction),
  stalerank_missing_count, manifest regime/gap.
- **Roster/field-relative encodings (r15):** roster_{signal}_{stat},
  relative_{signal}_minus_mean / _robust_z / _percentile,
  complement_position/_sd/dispersion/uncertainty, veteran_slot_occupancy,
  teammate_transition, survival priors, OOF heads (dnf_probability,
  dnf_variance, completion_fraction, reliability_interaction),
  stateset_stalerank_disagreement.
- **Decode-side inputs (r18–r20):** f0_projection & incumbent-level
  projections, cold empirical-Bayes buckets (age × appearances × gap),
  era shift/scale (r19), rating states + Poisson-binomial decode (r20),
  cold_fraction_x_log_staleness (r19).

## Never-touched in this campaign (grep-verified, all 20 code dirs)

- **circuits table: 0 references** — still the only unread DB relation
  (same gap as run4; FEATURE_SWEEP_driver_position.md).
- **Races-into-season / round number: absent** (month sin/cos present in
  19 files, but no within-season race index).
- **Head-to-head matrices: absent** (all `pairwise` hits are ranking-loss
  code in stateset_model.py / TL pairwise ranks, not driver-vs-driver
  history features).
- **external_junior_v1: 0 references.** Note: that artifact was built on
  the PARALLEL box (0727c) and never existed in 0727d's shared cache;
  0727d used no external data at all. It reached a 0727d-derived cache
  only in the later 0727e transplant.
- **TabPFN:** implemented once (r12 `pfn_lane.py`), gate-rejected, never
  in a live registered path afterwards.
- Formerly-never-tried items now COVERED (contradicting the run4-era
  gaps): explicit trend/slope (r7 velocities), target-encoding-style
  cohort priors (r7 age-band/nationality position priors), interaction
  features (r7, r19), season-phase partially (month encodings only).

## Verdict

Against run4's frozen 624-line `f1_features.py` (byte-identical runs
3–11), the axis contract demonstrably widened feature-space coverage: 6
distinct StateSet representation versions, a second full tabular family,
a fusion frame, field-relative re-encodings, and three decode-side input
families; three of run4's five "never tried" gaps (trend/slope, cohort
target priors, interactions) were closed, and the feature file kept
changing to the final iteration. The residual gaps are now sharply
defined. Highest-value untried directions, grounded in this campaign's own
measurements (long-gap and cold strata carried the largest residual MAE in
every gate report; `long_31_80_gain` and `cold_gain` were the binding
gates):
1. **Circuit-composition features** (calendar-mix pace/DNF priors per
   roster) — the only unread relation; long-horizon-stable by
   construction, aimed exactly at the `beyond_80` gap bucket where r18's
   gates bound.
2. **Races-into-season / within-season index** — disambiguates the
   heavily-used standings snapshots (a month-3 standings position ≠
   month-10); one integer column into the r7 frame.
3. **Head-to-head/co-occurrence aggregates** (driver-vs-roster-complement
   career win rates) — complement_position exists, but no pairwise
   history; cheap from the results table.
4. **External pre-cutoff pedigree for the cold mass** — 0727d never had
   external_junior_v1; its cold gates (`cold_gain` ≥ threshold) failed
   repeatedly with purely internal features, while the artifact covering
   ~72% of test cold rows sat unused on the sibling box.
5. **Season-boundary reshuffle features** (championship-position deltas
   across the winter break) — the filter models decay continuously; no
   feature marks the discrete season boundary where seat/tier changes
   concentrate (the r1 seat-transition prior conditions on it internally
   but exposes no channel).
