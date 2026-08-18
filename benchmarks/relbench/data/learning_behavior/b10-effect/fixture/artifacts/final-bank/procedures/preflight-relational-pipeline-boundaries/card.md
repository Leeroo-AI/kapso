---
type: procedure
representation: text
title: Preflight relational pipeline boundaries
description: >-
  Validate keys, schemas, dtypes, time units, fold locality, and debug limits before a cached relational pipeline can produce evidence.
tags: [data:relational, validation, pitfall]
timestamp: 2026-08-18T02:18:28Z
scope: domain
scope_conditions: "cached relational features cross SQL, array, compiled-kernel, graph-conversion, or forward-fold boundaries"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-0.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a relational renewal lane independently encountered recoverable query-contract failures.
    effect: >-
      A reserved query alias and an assumed timestamp field broke feature construction before evaluation; explicit alias and schema checks repaired the build. This boundary failure exercises the preflight procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-1.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; an independent sequence lane crossed array, accelerator, and query-projection boundaries.
    effect: >-
      Precision promotion caused an accelerator dtype mismatch and ambiguous or omitted projections caused query binder errors; explicit dtype and projection contracts repaired each failure. This independent lane exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-2.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a third lane independently reused a cached matrix after changing its target block.
    effect: >-
      A stale source-to-destination column slice caused a shape mismatch during rejected widening, and correcting the explicit slice let the completed matrices be reused. This schema-version failure exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-4.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the champion lane independently crossed nullable-key, static-map, fold-ranking, and output-cleanup boundaries.
    effect: >-
      Null keys, unsafe integer missingness, cross-fold ranking, and ad hoc generated-file cleanup all failed or risked false evidence; filtering, dtype promotion, fold-local transforms, and safe exclusion repaired them. This cross-campaign recurrence exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-2.md#difficulties-from-changeslog-agent-result-difficulties-not-streamed
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate survival lane independently exercised debug-size, slice-metric, and process-control boundaries.
    effect: >-
      Oversized debug chains needed deterministic caps, a one-class slice needed an undefined metric, and an already-rejected fit was stopped by exact process identity before restoring the retained version. These debug controls exercise the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-5.md#difficulties-recovered-from-changeslog-one-shot-artifact-unrecoverable
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a compiled hazard lane independently encountered key-range and kernel-constant failures.
    effect: >-
      Out-of-register foreign keys broke keyed aggregation and a reflected feature-name list broke compilation; filtering key ranges and using a compile-time feature count repaired the pipeline. This independent lane exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-6.md#difficulties-recoverable-record-from-changeslog-one-shot-artifact-unrecoverable
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a graph lane independently crossed foreign-key, static-entity, and nullable-event boundaries.
    effect: >-
      Dangling keys and null IDs broke graph and pseudo-seed construction, while an overbroad causality assertion mistook static creation times for events; sanitizing keys and restricting assertions to event nodes repaired the build. This independent lane exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-4.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate campaign independently exercised mixed-date and datetime-unit boundaries.
    effect: >-
      Mixed snapshot dates broke comparisons and mismatched datetime units collapsed decay weights until coercion and explicit unit normalization restored the intended arithmetic. This cross-campaign recurrence exercises the procedure.
reliability:
  validity: 0.80
  boundary: 0.82
  coverage: 0.48
  score: 0.74
  rationale: >-
    Validity 0.80 reflects repeated failure-and-repair exercises in three campaigns, but no served confirmation or injected-fault replay. Boundary 0.82 is strong across schema, key, dtype, time-unit, fold-locality, metric, compilation, cache, and debug contracts. Coverage 0.48 spans all three churn campaigns and many pipeline boundaries but applies the 0.5 unvisited discount. Overall 0.74 balances those dimensions. Cross-campaign mechanisms receive full participation weight; same-campaign lanes add discounted breadth, and duplicate streamed records count once. Most score-moving next: codify the checklist and replay it against injected boundary faults.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from recurrent relational-pipeline boundary failures and repairs [E1-E8].
supersedes: null
contradicts: []
probe: >-
  Inject one fault at each boundary—missing projection, invalid key, nullable integer, unit mismatch, pooled-fold transform, one-class slice—and require preflight to fail before model fitting.
---

Method [E1][E2][E3][E4][E5][E6][E7][E8]: bind every cached matrix to an explicit schema, projection order, feature count, dtype, datetime unit, and source hash, then assert them on load. Before indexing or graph conversion, separate nullable events from entity edges, filter or null out-of-register keys, and apply causality assertions only to true event nodes. Keep selection transforms within each fold, report metrics as undefined on one-class slices, and cap debug chains deterministically so debug output cannot masquerade as evidence. Compiled kernels use compile-time feature counts; process termination targets an exact decided fit; generated outputs are excluded safely instead of deleted ad hoc. Run this preflight before expensive feature generation and again before a cached artifact is reused.
