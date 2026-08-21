# K-ramp: pay the shared build once under k=8

**Status: PROPOSAL.** The behavioural change lives in the generic search
strategy (`src/kapso`), which is framework core — so this is a design spec
pending approval, not an implemented change. The config/runner surface below is
the benchmark-side wiring that lands once the core supports a per-round K.

## Problem

`k=8` fans out 8 implementation lanes in **every** round, round 1 included. On a
cold start — a real competition with no pre-seeded cache — all 8 lanes begin at
the same instant with no shared substrate.

Many tasks have an expensive **build-once-reuse** substrate. Task 1's is the
frozen-encoder embedding table: ~10 min to extract features over 1,283 clips,
after which every idea (adapter, calibration, transductive TTA) reads it in
seconds. If each of the 8 lanes builds it independently, we pay ~10 min ×8 in
parallel **and** every lane loses its first ~10 min to the build instead of
exploring — the "dilution" measured on prior parallel runs. Our practice runs
hid this by pre-seeding the cache; a URL-in competition can't.

The shared-cache mechanism (`--shared-cache-dir` → `KAPSO_SHARED_CACHE_DIR`)
carries the substrate across rounds/lanes **once it exists**, but nothing builds
it before the fan-out, and 8 lanes checking an empty cache at t=0 can't
"check-before-build" their way out of a simultaneous stampede.

## Proposed change

A per-round K schedule instead of a single fixed `node_expansion_value`:

- **Round 1 = `K_warm` (default 1):** one lane builds the substrate + a baseline
  and seeds the shared cache.
- **Rounds ≥ 2 = `K_full` (default 8):** fan out on the now-warm cache; each lane
  forks ideas on top of the built substrate.

`parent_policy=best` already expands rounds ≥2 from round 1's winner, so the warm
baseline naturally becomes the parent the 8 lanes build on.

### Config surface (`benchmarks/ioai2026/config.yaml` → `run_defaults`)

```yaml
run_defaults:
  hours: 1.75
  node_expansion: 8         # K_full (rounds >= 2)
  node_expansion_warm: 1    # K_warm (round 1); set == node_expansion to disable the ramp
```

### Where it touches core

The generic strategy's round loop reads `node_expansion_value` once and uses it
every `_expand_round`. The ramp makes per-round K a function of the round index:

```
K(round) = K_warm if round == 1 else K_full
```

Lane env pins (`expansion_lane_env`) are generated for `K_full` (the max) and
sliced to `K(round)` each round — no new pinning logic. Everything else in
`_expand_round` is unchanged.

### Why `K_warm = 1` (not 2+)

One builder = zero redundant builds and a single clean parent for round 2. Even
`K_warm = 2` risks two concurrent builds of the same substrate. For a task with
no heavy substrate (round 1 is cheap), set `node_expansion_warm == node_expansion`
to skip the ramp entirely.

### Zero-core-change interim (usable today)

Run a short `--node-expansion 1` pass that builds + banks a baseline into a
`--shared-cache-dir`, then launch the k=8 run pointed at the same dir. Two
launches, no core change — clunkier, but it unblocks cold-start competitions
before the ramp lands.

## Validation

- Round 1 spawns exactly 1 lane; the shared cache holds the substrate at the end
  of round 1; rounds ≥2 spawn `K_full` lanes and none rebuild it (assert on cache
  file mtime).
- `k=1` and "ramp disabled" (`node_expansion_warm == node_expansion`) paths stay
  byte-identical to today's behaviour.
