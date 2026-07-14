#!/usr/bin/env bash
# RelBench campaign driver — runs Kapso over prioritized task waves.
#
# Usage:
#   bash scripts/run_relbench_campaign.sh <wave> [iterations] [mode]
#   waves: autocomplete-soft | uncontested | recommendation | soft-cells | smoke
#
# Each task runs sequentially (one GPU). Results land in
# tmp/relbench/<dataset>--<task>/final_report.json.

set -uo pipefail

WAVE="${1:-smoke}"
ITER="${2:-20}"
MODE="${3:-RELBENCH_CONFIGS}"

# Wave 0: head-to-head duel vs RelAgent + KumoRFM-ft (see BASELINES.md).
# All are v1 tasks where BOTH baselines report numbers (any win counts twice),
# on small/medium databases (cheap iterations). Bars to beat, from
# data/baselines.json (RelAgent | KumoRFM-ft):
#   user-attendance  MAE  0.241 | 0.238   (zero-inflated label; RT-ft anomaly shows huge headroom)
#   driver-position  MAE  4.019 | 2.731   (tiny DB; overall #1 needs < 2.63)
#   ad-ctr           MAE  0.033 | 0.034   (RelAgent's flagship board #1 — beat the agent archetype)
#   study-adverse    MAE 37.194 | 44.225  (RelAgent's 2nd flagship)
#   study-outcome  AUROC  71.86 | 71.16   (honest overall bar ~72.5 GelGT)
#   user-repeat    AUROC  78.20 | 80.64   (overall bar 83.6 GelGT)
# Rec add-ons cover KumoRFM-ft where RelAgent has no results:
#   condition-sponsor-run MAP 11.65 | user-ad-visit MAP 4.17 (best known anywhere)
BASELINE_DUEL=(
  "rel-event user-attendance"
  "rel-f1 driver-position"
  "rel-avito ad-ctr"
  "rel-trial study-adverse"
  "rel-trial study-outcome"
  "rel-event user-repeat"
  "rel-trial condition-sponsor-run"
  "rel-avito user-ad-visit"
)

# Wave 1: autocomplete tasks with near-random published baselines (no leaderboard).
AUTOCOMPLETE_SOFT=(
  "rel-salt sales-group"
  "rel-salt sales-payterms"
  "rel-salt sales-shipcond"
  "rel-salt sales-incoterms"
  "rel-salt item-incoterms"
  "rel-amazon review-rating"
  "rel-event event_interest-interested"
  "rel-event event_interest-not_interested"
  "rel-event users-birthyear"
  "rel-avito searchstream-click"
  "rel-avito searchinfo-isuserloggedon"
  "rel-f1 results-position"
  "rel-f1 qualifying-position"
  "rel-trial studies-enrollment"
  "rel-trial studies-has_dmc"
  "rel-stack badges-class"
  "rel-ratebeer beer_ratings-total_score"
  "rel-hm transactions-price"
)

# Wave 2: v2 tasks where only the paper baselines exist.
UNCONTESTED=(
  "rel-ratebeer user-beer-favorite"
  "rel-ratebeer user-beer-liked"
  "rel-ratebeer user-place-liked"
  "rel-ratebeer beer-churn"
  "rel-ratebeer user-churn"
  "rel-ratebeer brewer-dormant"
  "rel-ratebeer user-count"
  "rel-arxiv paper-citation"
  "rel-arxiv author-category"
  "rel-arxiv author-publication"
  "rel-arxiv paper-paper-cocitation"
)

# Wave 3: the stale recommendation board.
RECOMMENDATION=(
  "rel-trial site-sponsor-run"
  "rel-trial condition-sponsor-run"
  "rel-hm user-item-purchase"
  "rel-avito user-ad-visit"
  "rel-stack user-post-comment"
  "rel-stack post-post-related"
  "rel-f1 driver-circuit-compete"
  "rel-amazon user-item-purchase"
  "rel-amazon user-item-rate"
  "rel-amazon user-item-review"
)

# Wave 4: soft classification/regression board cells.
SOFT_CELLS=(
  "rel-avito ad-ctr"
  "rel-trial study-adverse"
  "rel-trial site-success"
  "rel-avito user-clicks"
  "rel-f1 driver-position"
  "rel-event user-attendance"
  "rel-avito user-visits"
  "rel-trial study-outcome"
)

SMOKE=(
  "rel-f1 driver-position"
)

case "$WAVE" in
  baseline-duel)     TASKS=("${BASELINE_DUEL[@]}") ;;
  autocomplete-soft) TASKS=("${AUTOCOMPLETE_SOFT[@]}") ;;
  uncontested)       TASKS=("${UNCONTESTED[@]}") ;;
  recommendation)    TASKS=("${RECOMMENDATION[@]}") ;;
  soft-cells)        TASKS=("${SOFT_CELLS[@]}") ;;
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
