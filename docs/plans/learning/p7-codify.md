# P7 — The procedure code path

**Goal:** text procedures that earn it become verified executables: the
codify seeder, the codify run (evolve minus ideation), placement, freshness
(CD in full; closes the path gated off since P4). **Design sources:** CD
§1–§8, UC (docket integration), MD§3.2 (procedure card fields), MD§5.1
(code staging into the shared workspace). **Depends on:** P4 (docket, bank),
P5 (serving stages code), store archives (fixtures). **Doubts:** D8
(gcp_ephemeral machine type + billing — P7 starts `target: local`).

## Work items

1. **Seeder live** (CD§1): un-gate the `codify` docket rows — executed-verdict
   filter, reference-closure recurrence (the shared transparency function
   from P4.3), guards (active, uncontested, skip-until-new-evidence after a
   failed attempt); `learning.codify.min_recurrence` (v1: 2).
2. **Procedure-codifier live** (CD§6): the compatibility gate crew-side
   (before any machine spins; declined → PASS with the argument; two
   incompatible realizations flagged as a split candidate), run request,
   outcome fold (flip, version bump, log, journal CODIFY).
3. **Codify-run driver** (CD§2): evolve minus ideation, reusing the existing
   substrate — implementor session (card-as-spec framing addendum;
   adapt-don't-author from the cited archived implementations), registered
   evaluation as the reproduction gates (decision outcomes exact; numeric
   within ±`tolerance_z`·SE of the fixture run's recorded outcome; artifact
   outcomes property-checked; the anti-weak-test check: assertions must
   implement the card's `expected_outcome`), the feedback judge (four claims
   questions: reproduction, faithfulness, preconditions honesty, ledger
   consistency; verdict + feedback), bounded iterations
   (`max_iterations` v1: 3). Pass = mechanical green AND judge endorsement;
   the evaluation becomes `replay/`; **the flip commits only inside a
   learning transaction holding a green run** (validator from P4.2 now
   exercised). Actually-invoked check: workspace staged with fixture inputs
   only, outputs must be freshly produced.
4. **Placement** (CD§3): `target: local` first (preconditions-satisfying box,
   sandboxed workspace, iteration timeout); `gcp_ephemeral` behind D8 —
   machine type from `preconditions`, standard env bootstrap, staging from
   the artifacts bucket, unconditional teardown, preemptible-safe.
   Environment is always current (rot detection by design — never a
   historical env).
5. **Freshness** (CD§4): `last_replayed` + `replay_max_age` via the expiry
   docket; eval-only re-run; one feedback iteration on failure, else demote
   to text with the log saying why; stale/demoted never stages.
6. **Merge inheritance** (CD§5): successor inherits code only if replay
   passes re-run in the same MERGE transaction; else born text and
   re-nominated by the seeder.

## Tests

- Seeder: executed-verdict filter (a mention-only fixture entry never
  counts); closure union example; skip-until-new-evidence.
- Reproduction gates: decision mismatch fails; metric outside band fails;
  weak assertion (green but weaker than `expected_outcome`) fails
  validation; fixture-output leak trips actually-invoked.
- Transaction rule: a flip without a green run in the same transaction is
  rejected by the validator.
- Freshness: aged `last_replayed` seeds the expiry row; demote path writes
  the log entry.

## Done gate

One real card codified end to end (green run, flip, staged into the shared
workspace at next serve) and one freshness re-run green — the CD§8
forward-gate story, on real material.
