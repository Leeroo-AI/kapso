
# Core Idea: Jointly forecast the next calendar and each seeded driver’s race-by-race participation

Treat the target as two coupled problems: determine which circuits will be on the upcoming Formula 1 calendar and estimate the rounds during which each driver will have a seat. Use the complete set of seeded drivers as legal transductive information: every listed driver is known to participate at least once during the forecast horizon.

# Body:

1. **Load and validate data**
   - Load the dataset, task splits, and database only through the prescribed RelBench API with `download=False`.
   - Preserve validation and test row order exactly.
   - Build chronologically ordered race seasons from `races`.
   - Join `results` to `races` to obtain `(date, year, round, circuitId, driverId, constructorId)` participation events.

2. **Build legal rolling-origin training cases**
   - Create historical January 1 prediction cases for all years with a complete following season.
   - For model A, retain only cases whose entire 365-day target interval ends on or before the validation timestamp.
   - For model B, retain all cases whose target interval ends on or before the test cutoff, including the official validation examples.
   - Reconstruct each historical seed cohort as the distinct drivers who compete in the following 365 days. This mirrors the information contained in the validation/test source rows without exposing future circuits.

3. **Forecast the upcoming calendar as circuit-round probabilities**
   - For every circuit and future round position, calculate:
     - Presence in each of the previous 1–8 seasons.
     - Consecutive appearance streak.
     - Years since last appearance.
     - Exponentially decayed appearance frequency.
     - Last observed round and median round over recent seasons.
     - Round-position stability.
     - Whether the circuit disappeared and later returned historically.
     - Country and geographic redundancy with other likely circuits.
   - Fit two small calibrated models using historical rolling origins:
     - A binary circuit-presence model.
     - A conditional round-position model for circuits predicted present.
   - Use LightGBM or CatBoost with shallow trees and strong regularization because there are few seasons.
   - Blend model probabilities with an explicit persistence prior:
     ```text
     P_calendar(c) =
         0.55 * learned_presence_probability
       + 0.30 * recent-season persistence
       + 0.15 * long-run recurrence
     ```
   - Generate an expected ordered calendar by assigning likely circuits to rounds with maximum-weight bipartite matching, using predicted presence and round-position probabilities.

4. **Infer driver status from the full seed cohort**
   - For each seeded driver, compute features at the seed time:
     - Days since last start, qualifying appearance, and standing.
     - Starts in the previous 90, 180, 365, and 730 days.
     - Previous-season first and last rounds.
     - Fraction of previous-season rounds entered.
     - Number of constructors represented recently.
     - Career starts, age, debut recency, and prior gaps between seasons.
     - Last constructor and whether multiple seeded drivers share that constructor.
   - Add cohort-level features:
     - Number of incumbent drivers with very recent starts.
     - Number of rookies or returnees.
     - Drivers per last-known constructor.
     - Rank among seeded drivers by recent starts and recency.
   - The cohort features distinguish likely full-season incumbents from reserve drivers and mid-season substitutes better than independent driver scoring.

5. **Learn race-round attendance probabilities**
   - On historical cohorts, label every `(driver, future_round)` pair according to whether the driver competed in that round.
   - Fit a compact gradient-boosted binary model or regularized logistic model with:
     - Driver and cohort features.
     - Future round index and normalized round position.
     - Interactions between round and incumbent/rookie/returning status.
     - Historical first-round and last-round transition statistics.
   - Use grouped rolling-origin out-of-fold predictions for calibration.
   - Apply isotonic or Platt calibration fitted only on legal historical out-of-fold predictions.

6. **Convert calendar and attendance forecasts to circuit relevance**
   - For every driver and circuit, compute:
     ```text
     score(d, c) =
         sum_r P_calendar(c at round r) * P_attend(d, r)
       + 0.08 * driver_historical_affinity(d, c)
       + 0.03 * circuit_global_tie_break(c)
     ```
   - Cap driver-circuit affinity so it cannot override strong calendar evidence.
   - For full-season incumbents, rank mainly by calendar certainty.
   - For rookies and long-absence returnees, favor early rounds.
   - For drivers classified as likely substitutes, favor rounds with the highest learned substitution hazard.

7. **Tune only through historical rolling backtests**
   - Optimize MAP@10 across historical target seasons.
   - Weight the latest four legal seasons approximately `1, 1.5, 2, 3`.
   - Search:
     - Calendar persistence/model blend.
     - Number of expected calendar rounds.
     - Attendance-model regularization.
     - Cohort-feature strength.
     - Driver-affinity cap.
   - Choose robust parameters by leave-one-season-out mean MAP and worst-season MAP, not a single season.
   - Do not use official validation labels to fit or select model A.

8. **Train the two required models**
   - Model A uses only historical cases ending by the validation seed time and produces `val_predictions.npy`.
   - Model B uses all legally available history plus validation labels and produces `test_predictions.npy`.
   - Refit the same selected architecture rather than retuning against hidden test behavior.

9. **Debug and output**
   - Debug mode uses fixed blend weights, a logistic attendance model, and at most the most recent six historical seasons.
   - Full mode performs the rolling-origin search and shallow boosted fitting.
   - Assert shape `(27, 10)`, integer dtype, ten unique circuit IDs per row, and values in `[0, 77)`.
   - Save both arrays with `np.save` in `KAPSO_RUN_DATA_DIR`.

# Runtime expectation:
Debug: 20–60 seconds. Full: 3–10 minutes.
