Implement the provided solution for the problem. This is a build-only session:
Kapso estimates platform cost on the finalized candidate before granting execution.
Do not run the candidate or evaluation, train models, submit platform jobs, or
perform paid platform operations. Reading code and writing the candidate are allowed.
Kapso runs the evaluator after the candidate passes its cost cap and campaign budget.

## Implementation

Implement the solution completely. Read the evaluation contract to understand the
goal. Preserve provided evaluation code and protected data. When no evaluation is
provided, create `kapso_evaluation/evaluate.py`; it must test the goal fairly and
print its score. Optional measured cost uses a JSON manifest line:
`KAPSO_EVAL_MANIFEST {"cost": "2.5 credits"}`. A registered evaluator must retain
its existing manifest fields and entrypoint. Never alter its evaluation behavior.
Do not execute the evaluator in this session.

{{lane_brief}}

## Repository Memory

{{repo_memory_brief}}

{{repo_memory_detail_access_instructions}}

Record the repo memory sections consulted in `changes.log`, or record `none`.
Use the available research and knowledge tools when needed.

{{knowledge_tools}}{{inbox_tool_line}}

## Shared Campaign Cache

Inspect existing artifacts before implementing duplicate work. Do not create
expensive artifacts during this build-only session.

{{shared_artifacts_brief}}

## Previous Errors

{{previous_errors}}

## Budget

{{budget_status}}

## Problem

<problem>
{{problem}}
</problem>

## Solution

<solution>
{{solution}}
</solution>

{{inbox_section}}

## Final Output

Return these XML tags as the last part of the response. Evaluation has not run;
leave its output empty and its score null.

<code_changes_summary>Describe all implementation changes.</code_changes_summary>
<evaluation_script_path>kapso_evaluation/evaluate.py</evaluation_script_path>
<evaluation_output></evaluation_output>
<score>null</score>
<technical_difficulties>Describe failed attempts, causes and fixes, or none.</technical_difficulties>
