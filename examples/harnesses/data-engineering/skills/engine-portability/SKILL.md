---
name: engine-portability
description: Keep a change correct on every engine the project ships, with the engines, versions and session settings declared, the semantics that diverge between engines (division, non-finite numerics, dates and time zones, ordering ties, identifiers and metadata), a fixture protocol that executes the same small input on each engine and compares values and types, and the three evidence classes compiled, executed and compared. Use when a change depends on numeric, timestamp, ordering, identifier or metadata behaviour, and before declaring a change finished on a project with more than one engine.
---
# Engine portability

Declare the supported engines, their versions and the relevant session settings from the
project. For every changed expression that is engine-sensitive, execute the same small
input relation on each required engine and compare both values and output types. Do not
infer cross-engine compatibility from matching function names.

## Where engines diverge

- **Division and numeric types**: integer against decimal division; the type a division
  returns; zero denominators (an error on one engine, null or infinity on another).
- **Non-finite numerics**: whether NaN and infinity can exist, how they compare and sort,
  and which predicate finds them; a check that catches NaN does not necessarily catch
  infinity.
- **Dates and time zones**: parse formats and what an unparsable value does; the session
  zone applied by a cast; day boundaries and daylight-saving transitions; the argument
  order of a date difference.
- **Ordering**: ties without a tie-breaker; null ordering; default window frames, which
  depend on the engine and the function.
- **Identifiers and metadata**: case folding of unquoted names; whether a metadata query
  must be qualified by database or dataset; what the information schema calls a type.
- **Strings and regular expressions**: global against first-match replacement; escape
  rules; the names of aggregation functions.

## The fixture protocol

1. Record the expected behaviour from the contract before running anything: what the
   expression must return for each case.
2. Build a fixture relation that includes the distinguishing cases: integer and decimal
   division; a zero denominator; a null and a non-finite numeric; an invalid and an
   ambiguous date; timestamps at a day boundary and across a daylight-saving transition;
   tied ordering keys; a quoted identifier.
3. Execute the changed expression over the fixture on every required engine, through the
   project's own tooling (on dbt, `dbt show --inline` under each target), and compare
   values and types with the expectation.
4. Record the result in the proof's `portability` by evidence class, per engine:
   **compiled** proves rendering only; **executed** proves the engine ran it; **compared**
   proves the values matched the expectation. An engine that was not executed is an
   explicit validation gap, and local execution never establishes the production engine's
   semantics.

## Rules

- Write the portable form first and branch only where the dialects differ; on dbt, the
  branch and macro patterns are in dbt-workflow.
- Never silence a parse with a bare cast; a null where the raw value is not null is a
  finding for the invariants (`finite` and the parsed-nulls check).
- Use the project's adapter or metadata interface to resolve the actual catalog, schema,
  identifier and types; a metadata query copied from one engine is not evidence on
  another.
