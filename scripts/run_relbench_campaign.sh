#!/usr/bin/env bash
# RelBench campaign driver — sequenced waves covering all 66 native tasks.
#
# Usage:
#   bash scripts/run_relbench_campaign.sh <wave> [iterations] [mode]
#   waves: roi (recommended flat ROI-sorted queue, all 65 tasks) or thematic:
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

# CPU-local queue: the ROI order restricted to datasets that are safe on a
# CPU-only box with ~32 GB RAM (db.zip sizes measured 2026-07-14: rel-f1 1MB,
# rel-salt 34MB, rel-event 100MB, rel-arxiv 145MB, rel-avito 347MB,
# rel-trial 548MB — all comfortably in-memory; excluded: rel-stack 840MB
# text-heavy, rel-ratebeer 2.2GB, rel-amazon 6.1GB, rel-hm 31M-row modeling).
# The handler detects the missing GPU and steers agents to duckdb/GBDT
# pipelines automatically; note RelAgent's own method is pure SQL+GBDT, so
# every cell where it is the bar is in principle CPU-winnable.
CPU_LOCAL=(
  "rel-event user-attendance"
  "rel-f1 driver-circuit-compete"
  "rel-f1 results-position"
  "rel-f1 qualifying-position"
  "rel-f1 driver-position"
  "rel-event event_interest-interested"
  "rel-event event_interest-not_interested"
  "rel-event users-birthyear"
  "rel-event user-repeat"
  "rel-event user-ignore"
  "rel-f1 driver-dnf"
  "rel-salt sales-group"
  "rel-salt sales-payterms"
  "rel-salt sales-shipcond"
  "rel-salt sales-incoterms"
  "rel-salt item-incoterms"
  "rel-trial studies-enrollment"
  "rel-trial studies-has_dmc"
  "rel-avito searchinfo-isuserloggedon"
  "rel-avito searchstream-click"
  "rel-trial study-adverse"
  "rel-avito ad-ctr"
  "rel-trial study-outcome"
  "rel-trial condition-sponsor-run"
  "rel-avito user-ad-visit"
  "rel-avito user-clicks"
  "rel-trial site-success"
  "rel-trial site-sponsor-run"
  "rel-arxiv author-publication"
  "rel-arxiv paper-citation"
  "rel-arxiv author-category"
  "rel-arxiv paper-paper-cocitation"
  "rel-trial eligibilities-adult"
  "rel-trial eligibilities-child"
  "rel-avito user-visits"
  "rel-f1 driver-top3"
  "rel-salt item-plant"
  "rel-salt item-shippoint"
  "rel-salt sales-office"
)

# ROI-sorted flat queue (recommended default): expected claim value x win
# probability / compute cost. Tier S = tiny DBs, near-certain or
# strategy-critical wins; Tier A = medium DBs, flagship takedowns + soft
# uncontested cells; Tier B = 8h-tier DBs needed for the category-mean gates;
# Tier C = outlier/saturated cells, match-only. Same 65-task coverage as the
# thematic waves.
ROI=(
  # --- Tier S: rel-f1 + rel-event (minutes-per-iteration DBs) ---
  "rel-event user-attendance"            # beats both baselines AND the regression-category keystone (RT anomaly)
  "rel-f1 driver-circuit-compete"        # smoke heuristic already beat SOTA (76.7 > 76.18) — bank it
  "rel-f1 results-position"              # uncontested AC, bar R2 0.528
  "rel-f1 qualifying-position"           # uncontested AC, bar R2 0.239
  "rel-f1 driver-position"               # beats RelAgent at 4.019; Kumo 2.731 is the stretch
  "rel-event event_interest-interested"  # AC bar ~ coin-flip
  "rel-event event_interest-not_interested"
  "rel-event users-birthyear"            # AC bar R2 < 0
  "rel-event user-repeat"                # beats both at >80.64
  "rel-event user-ignore"                # beats both at >89.43
  "rel-f1 driver-dnf"                    # stretch 82.63, nearly free to try
  # --- Tier A: salt/trial/avito/ratebeer/arxiv (medium DBs) ---
  "rel-salt sales-group"                 # AC bar 15.8 acc — join-lookup giveaway
  "rel-salt sales-payterms"              # AC bar 37.5
  "rel-salt sales-shipcond"              # AC bar 56.9
  "rel-salt sales-incoterms"             # AC bar 62.2
  "rel-salt item-incoterms"              # AC bar 69.4
  "rel-trial studies-enrollment"         # AC bar R2 0.436 (v2 baselines at 0)
  "rel-trial studies-has_dmc"            # AC bar 78.5
  "rel-avito searchinfo-isuserloggedon"  # AC bar 73.0 (LightGBM at 50)
  "rel-avito searchstream-click"         # AC bar 55.9 ~ random
  "rel-trial study-adverse"              # RelAgent's board-#1 flagship takedown
  "rel-avito ad-ctr"                     # RelAgent's other board-#1 flagship
  "rel-trial study-outcome"              # beats both + honest overall #1 at >72.5
  "rel-trial condition-sponsor-run"      # rec, best-known 11.65
  "rel-avito user-ad-visit"              # rec, best-known 4.17
  "rel-avito user-clicks"                # beats RelAgent >68.36; overall 69.4
  "rel-trial site-success"               # beats both at <0.301 MAE
  "rel-trial site-sponsor-run"           # rec 28.02 — riskier, biggest rec headline
  "rel-ratebeer user-count"              # RelAgent v2 duel (MAE 6.021 / R2 0.625)
  "rel-arxiv author-publication"         # RelAgent v2 duel (MAE 0.462 / R2 0.249)
  "rel-ratebeer beer-churn"              # RelAgent v2 duel 84.70
  "rel-ratebeer brewer-dormant"          # 83.33
  "rel-arxiv paper-citation"             # 82.62
  "rel-ratebeer user-churn"              # 98.63 — near-saturated duel cell
  "rel-ratebeer beer_ratings-total_score" # AC bar R2 0.394
  "rel-ratebeer user-beer-liked"         # rec bar 1.46 — very soft
  "rel-ratebeer user-place-liked"        # rec bar 1.85
  "rel-ratebeer user-beer-favorite"      # rec bar 1.89
  "rel-arxiv author-category"            # multiclass bar 50.74
  "rel-arxiv paper-paper-cocitation"     # rec bar 35.4
  "rel-trial eligibilities-adult"        # AC, strong bar 93.7 — hardest AC cells last
  "rel-trial eligibilities-child"        # AC, bar 87.3
  # --- Tier B: stack/hm/amazon (8h-tier; required for category-mean gates) ---
  "rel-stack badges-class"               # AC bar 82.8
  "rel-hm transactions-price"            # AC bar R2 0.736
  "rel-amazon review-rating"             # AC bar R2 < 0 — soft but big DB
  "rel-hm user-churn"                    # beating both = overall #1 (>71.23)
  "rel-hm item-sales"                    # tight (<0.034 MAE)
  "rel-hm user-item-purchase"            # rec 3.14 — Kaggle-proven recipe
  "rel-stack user-engagement"            # >90.70
  "rel-stack user-badge"                 # >89.86
  "rel-stack post-votes"                 # tight (<=0.064 MAE)
  "rel-stack user-post-comment"          # rec 13.34 (Kumo) / 14.0 (RelGNN)
  "rel-stack post-post-related"          # rec 12.5
  "rel-amazon user-churn"                # tight 70.78
  "rel-amazon item-churn"                # tight 82.84
  "rel-amazon user-ltv"                  # 13.949
  "rel-amazon item-ltv"                  # 41.765
  "rel-amazon user-item-purchase"        # rec 2.93
  "rel-amazon user-item-rate"            # rec 2.25
  "rel-amazon user-item-review"          # rec 1.63
  # --- Tier C: outliers + saturated (match, don't chase) ---
  "rel-avito user-visits"                # KumoRFM-ft outlier 78.3
  "rel-f1 driver-top3"                   # 99.62 saturated
  "rel-salt item-plant"                  # 99.46
  "rel-salt item-shippoint"              # 98.39
  "rel-salt sales-office"                # 99.88
)

case "$WAVE" in
  roi)               TASKS=("${ROI[@]}") ;;
  cpu-local)         TASKS=("${CPU_LOCAL[@]}") ;;
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
