---
name: invariants
description: The property checks that prove a dataset right, with the minimum set a senior engineer insists on per dataset, reconciliation layered from schema to counts to control totals to rows, conservation checks as the cheap catch for fan-out and filter bugs, the acceptance condition kept apart from the stakeholder's hypothesis, what a check before and after a change does and does not show, an engine-neutral spec format, and a script that runs the spec through dbt show on a dbt project. Use in setup for the baseline, in implementation before and after a change, and in verification to recompute the proof.
---
# Invariants

```
python3 $KAPSO_HARNESS_ROOT/skills/invariants/scripts/invariants.py <spec.json> \
  [--project-dir .] [--profiles-dir DIR] [--target NAME] [--out results/invariants_after.json]
```

Runs every check of the spec through `dbt show --inline` (all of them in one statement,
each on its own when that statement fails, so the check at fault is named), prints a
table (name, status, evidence) and writes the results as JSON with the measurement
context (project, target, code revision) and the spec's SHA-256. Exit status 1 when a check fails or errors, and an error when the spec
yields no check. One spec file per dataset; the attempt's spec is the baseline spec plus
the checks the change adds, and every check has its own name. On a stack without dbt the
same check semantics still say what to verify with the project's own query interface, and
the proof records each check's SQL and result; the evidence scripts do not read such a
record, so the computed label stays `unsure` there (validation-report).

## The minimum per dataset

Every dataset carries unique and not-null on its grain key; every source carries freshness
thresholds and schema tests; low-cardinality fields get accepted values and foreign keys
get relationships; every table gets a volume monitor; published datasets add checks on
the measures people read (sums or maxima by critical dimensions), because those are the
data products. Prioritise semantic checks on published measures and on the changed
resource's consumers, and state the coverage boundary.

## Two kinds of check, and reconciliation layered on top

- Assertions validate an explicit expectation and fail the job; anomaly monitors track a
  metric over time and warn. Fixed volume thresholds ignore growth and seasonality;
  excessive failures cause alert fatigue, which is its own incident.
- Reconciliation goes schema, then counts, then control totals, then rows or hashes,
  because row counts lie: a control total detects a wrong load but not which row. A
  passing check means within the declared tolerance; choose absolute and relative
  tolerances from the measure's units, precision and accepted error policy.
- Conservation checks (sums that must net to zero, counts that must match across stages,
  attributed revenue against revenue) are the one cheap check that catches fan-out and
  filter bugs which uniqueness and null tests cannot see.
- A structural check cannot see semantics: a currency column holding both valid values
  passes every assertion while the numbers flip; that is what the keyed diff is for
  (data-diff).

## The spec

```json
{
  "model": "merchant_payments_daily",
  "grain": ["merchant_id", "day"],
  "not_null": ["merchant_id", "day"],
  "non_negative": ["payment_count", "gross_amount"],
  "bounds": {"refund_share": [0, 1], "health_score": [0, 100]},
  "finite": "auto",
  "monotonic": [{"column": "running_gross_amount", "order_by": "day", "partition_by": ["merchant_id"]}],
  "no_gaps": [{"column": "day", "partition_by": ["merchant_id"]}],
  "reconcile": [{"name": "gross amount equals the settled ledger entries",
                 "left": "select sum(gross_amount) from {{ ref('merchant_payments_daily') }}",
                 "right": "select sum(amount) from {{ ref('stg_ledger_entries') }} where state = 'settled'",
                 "tolerance": 0.01}],
  "completeness": [{"name": "every merchant-day present", "model_key": "merchant_id || '|' || day",
                    "source": "(select m.merchant_id, d.day from {{ ref('dim_merchant') }} m cross join {{ ref('dim_date') }} d where d.day between '2026-01-01' and '2026-03-31') s",
                    "source_key": "merchant_id || '|' || day"}],
  "custom": [{"name": "the method shares of a merchant-day sum to one",
              "violations": "select count(*) from (select merchant_id, day from {{ ref('merchant_payment_methods_daily') }} group by 1, 2 having abs(sum(share_of_day) - 1) > 0.001) broken"}]
}
```

Every key but `model` is optional. `finite: "auto"` discovers the floating-point columns
from the information schema and errors when the metadata is unavailable; a list names
them instead. A `custom` or generated query must return exactly one row with one non-null
value, a finite non-negative integer for a violation count. `reconcile` compares two
scalars as exact decimals within `tolerance` (absolute; `relative_tolerance` for a ratio).
`no_gaps` finds interior daily gaps among the rows present; to prove coverage, declare the
expected population as the `source` of a `completeness` check (an entity list crossed with
a period spine, as above) so missing endpoints and whole partitions count. `bounds` and
`monotonic` ignore null values; pair them with `not_null`. A cumulative is monotonic only
when its increments are constrained in sign; otherwise reconcile successive differences
to the increments and the final value to the total. The SQL goes through dbt's Jinja, so
`ref` and `source` resolve and `{% if target.type == 'snowflake' %}` branches a dialect.

## Writing the spec from a ticket

- The grain and the keys come from the dataset's purpose, not from its current output.
- Every sentence of the ticket that compares two things is a `reconcile`; every "all X
  present" is a `completeness`; every "never negative", "between", "no infinity" is a
  bound or `finite`; every "after the rerun" is a replay comparison (pipeline-reliability);
  every join or filter that could lose or multiply rows gets a conservation check.
- Separate the stakeholder's hypothesis from the acceptance condition: each check names
  the contract, source definition or independent calculation that justifies it, and no
  target number is copied from the ticket into the oracle. A ticket's expectation can be
  wrong, and the check is what shows it.
- Run the checks against both the original and the changed code on the same inputs. A
  check that fails before and passes after shows the correction; a check that passes in
  both states shows what was preserved, and the proof lists both. The validation-report
  skill's evidence.py does exactly that for a ticket: it reads the spec of each dataset
  from results/<ticket>/spec_<model>.json, runs it on the code as it stands (baseline),
  on the build after the change and on what production holds (`--target` with the
  reference target), and works out which checks moved. Write the spec first; the
  baseline fixes it by its hash, so a spec edited after the change no longer counts.
- Write at least one check that would fail on the defect the ticket describes: on what
  production holds now, or on the code as it stands. A spec every state passes proves
  nothing moved.
- Before and after every changed expression, exercise its boundary cases: null against
  zero, an absent match against a duplicated match, equal ordering keys, an empty group,
  the time or status boundary the ticket names.
