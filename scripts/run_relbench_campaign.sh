#!/usr/bin/env bash
# RelBench campaign driver — sequenced waves covering all 66 native tasks.
#
# Usage:
#   bash scripts/run_relbench_campaign.sh <wave> [iterations] [mode]
#   waves (in intended order):
#     w0-baseline-duel   8 tasks  v1, small/med DBs; every entity win beats BOTH
#                                 RelAgent and KumoRFM-ft (see BASELINES.md)
#     w1-v1-remainder    5 tasks  rest of v1 on small/med DBs
#     w2-relagent-v2     6 tasks  RelAgent's exact v2 subset (direct duel)
#     w3-autocomplete   20 tasks  no leaderboard; v2-paper baselines are the bar
#     w4-v2-open         6 tasks  v2 rec + multiclass with only v2-paper baselines
#     w5-big-v1         16 tasks  8h-tier DBs (amazon/hm/stack) + the Kumo outlier
#     w6-completeness    4 tasks  saturated cells — match, don't chase
#     smoke              1 task   harness sanity
#   rel-mimic/patient-iculengthofstay is excluded: requires credentialed
#   PhysioNet + BigQuery access (baseline is near-random 55.01 AUROC — run it
#   as soon as credentials exist).
#
# Bars per task live in benchmarks/relbench/data/sota.json and
# data/baselines.json (injected into the problem context automatically).
# Results land in tmp/relbench/<dataset>--<task>/final_report.json.

set -uo pipefail

WAVE="${1:-smoke}"
ITER="${2:-20}"
MODE="${3:-RELBENCH_CONFIGS}"

# Wave 0 — head-to-head vs RelAgent + KumoRFM-ft on cheap DBs.
# Bars (RelAgent | KumoRFM-ft): user-attendance MAE 0.241|0.238;
# driver-position MAE 4.019|2.731; ad-ctr MAE 0.033|0.034 (RelAgent's board #1);
# study-adverse MAE 37.194|44.225 (RelAgent's 2nd board #1);
# study-outcome AUROC 71.86|71.16; user-repeat AUROC 78.20|80.64.
# Rec add-ons (KumoRFM-ft only; RelAgent has no rec results):
# condition-sponsor-run MAP 11.65; user-ad-visit MAP 4.17.
W0_BASELINE_DUEL=(
  "rel-event user-attendance"
  "rel-f1 driver-position"
  "rel-avito ad-ctr"
  "rel-trial study-adverse"
  "rel-trial study-outcome"
  "rel-event user-repeat"
  "rel-trial condition-sponsor-run"
  "rel-avito user-ad-visit"
)

# Wave 1 — remaining v1 tasks on small/medium DBs.
# Bars: user-ignore AUROC >89.43 (KumoRFM-ft; overall 91.2 PluRel);
# driver-dnf AUROC >82.63 (ft; overall 84.6 KumoRFM-2);
# user-clicks AUROC >68.36 (RelAgent; overall 69.4 RGP);
# site-success MAE <0.301 (ft/RelGNN; overall RT-ft NMAE 0.5519 ~= MAE 0.263);
# site-sponsor-run MAP >28.02 (ft; board is stale at 19.0).
W1_V1_REMAINDER=(
  "rel-event user-ignore"
  "rel-f1 driver-dnf"
  "rel-avito user-clicks"
  "rel-trial site-success"
  "rel-trial site-sponsor-run"
)

# Wave 2 — RelAgent's exact v2 subset (direct duel on its chosen ground).
# Bars (RelAgent): beer-churn 84.70, user-churn 98.63, brewer-dormant 83.33,
# paper-citation 82.62 AUROC; user-count MAE 6.021, author-publication MAE 0.462
# (also report R2 to compare with the v2 paper: 0.625 / 0.249).
W2_RELAGENT_V2=(
  "rel-ratebeer beer-churn"
  "rel-ratebeer brewer-dormant"
  "rel-arxiv paper-citation"
  "rel-ratebeer user-count"
  "rel-arxiv author-publication"
  "rel-ratebeer user-churn"
)

# Wave 3 — full autocomplete sweep (no leaderboard; v2-paper GraphSAGE /
# RelGT-AC are the de facto SOTA). Ordered cheap -> expensive; soft first.
W3_AUTOCOMPLETE=(
  "rel-f1 results-position"
  "rel-f1 qualifying-position"
  "rel-event event_interest-interested"
  "rel-event event_interest-not_interested"
  "rel-event users-birthyear"
  "rel-trial studies-enrollment"
  "rel-trial studies-has_dmc"
  "rel-trial eligibilities-adult"
  "rel-trial eligibilities-child"
  "rel-avito searchstream-click"
  "rel-avito searchinfo-isuserloggedon"
  "rel-salt sales-group"
  "rel-salt sales-payterms"
  "rel-salt sales-shipcond"
  "rel-salt sales-incoterms"
  "rel-salt item-incoterms"
  "rel-stack badges-class"
  "rel-ratebeer beer_ratings-total_score"
  "rel-amazon review-rating"
  "rel-hm transactions-price"
)

# Wave 4 — v2 tasks with only v2-paper baselines (rec + multiclass).
# Bars: driver-circuit-compete MAP 76.18 (our smoke heuristic already scored
# 76.7 on test); ratebeer recs MAP 1.89/1.46/1.85; author-category acc 50.74;
# paper-paper-cocitation MAP 35.39.
W4_V2_OPEN=(
  "rel-f1 driver-circuit-compete"
  "rel-ratebeer user-beer-favorite"
  "rel-ratebeer user-beer-liked"
  "rel-ratebeer user-place-liked"
  "rel-arxiv author-category"
  "rel-arxiv paper-paper-cocitation"
)

# Wave 5 — 8h-tier v1 databases (amazon/hm/stack) + the KumoRFM outlier task.
# Tight margins; run after the pipeline is battle-tested. Notable bars:
# hm user-churn AUROC >71.23 (beating both = overall #1); post-votes MAE <0.064;
# amazon recs 2.93/2.25/1.63 MAP (KumoRFM-ft, absent from the stale board);
# user-visits AUROC 78.3 (KumoRFM-ft outlier; rest of field <=69.4).
W5_BIG_V1=(
  "rel-hm user-churn"
  "rel-hm item-sales"
  "rel-hm user-item-purchase"
  "rel-stack user-engagement"
  "rel-stack user-badge"
  "rel-stack post-votes"
  "rel-stack user-post-comment"
  "rel-stack post-post-related"
  "rel-amazon user-churn"
  "rel-amazon item-churn"
  "rel-amazon user-ltv"
  "rel-amazon item-ltv"
  "rel-amazon user-item-purchase"
  "rel-amazon user-item-rate"
  "rel-amazon user-item-review"
  "rel-avito user-visits"
)

# Wave 6 — saturated cells: aim to MATCH (>=99, or 85.23+ on top3), not to
# invest. Completes the 65-task coverage (66th = rel-mimic, credentialed).
W6_COMPLETENESS=(
  "rel-salt item-plant"
  "rel-salt item-shippoint"
  "rel-salt sales-office"
  "rel-f1 driver-top3"
)

SMOKE=(
  "rel-f1 driver-position"
)

case "$WAVE" in
  w0-baseline-duel|baseline-duel) TASKS=("${W0_BASELINE_DUEL[@]}") ;;
  w1-v1-remainder)   TASKS=("${W1_V1_REMAINDER[@]}") ;;
  w2-relagent-v2)    TASKS=("${W2_RELAGENT_V2[@]}") ;;
  w3-autocomplete)   TASKS=("${W3_AUTOCOMPLETE[@]}") ;;
  w4-v2-open)        TASKS=("${W4_V2_OPEN[@]}") ;;
  w5-big-v1)         TASKS=("${W5_BIG_V1[@]}") ;;
  w6-completeness)   TASKS=("${W6_COMPLETENESS[@]}") ;;
  smoke)             TASKS=("${SMOKE[@]}") ;;
  *) echo "unknown wave: $WAVE"; exit 2 ;;
esac

echo "Wave: $WAVE (${#TASKS[@]} tasks) | iterations=$ITER mode=$MODE"
FAILED=()
for entry in "${TASKS[@]}"; do
  read -r DS TASK <<<"$entry"
  echo -e "\n============================================================"
  echo "TASK: $DS / $TASK"
  echo "============================================================"
  PYTHONPATH=src:. python -m benchmarks.relbench.runner \
    -s "$DS" -t "$TASK" -i "$ITER" -m "$MODE" \
    2>&1 | tee "tmp/relbench_campaign_${DS}--${TASK}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "FAILED: $DS/$TASK (exit $status)"
    FAILED+=("$DS/$TASK")
  fi
done

echo -e "\nWave complete. Reports:"
for entry in "${TASKS[@]}"; do
  read -r DS TASK <<<"$entry"
  R="tmp/relbench/${DS}--${TASK}/final_report.json"
  [ -f "$R" ] && echo "  $R"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'Failed tasks:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
