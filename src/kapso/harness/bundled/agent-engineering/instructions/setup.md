# Setup

Prepare the campaign; investigate nothing. The attempts do the task's work, and the first
evaluation is the first measurement. Keep the session short.

1. Read the task statement and the runner's documentation the task ships. Write the domain
   statement: the task family and how the runner scores it (hidden tests, the final state
   of a system of record, named checks, an outcome), the plug-in point the scaffold must
   expose, the model and endpoint with the settings that are fixed and the ones the
   scaffold may change, the development set, the trials, the metric, the wall time of one
   full evaluation, the budget, the reference run and its number when one ships, the
   files the task declares private (its protection list), all as the task's own files
   state them.
2. Send one request to the endpoint with the task's request settings and a tool
   definition (on an OpenAI-compatible endpoint, the local-model-serving skill's smoke
   probe) and record what came back: whether a schema-checked tool call and its round
   trip came back, its latency and, where reported, the context window and whether
   thinking is on. Nothing longer: no load test, no long histories, no baseline, no
   transcript reading. What the endpoint does under load is an attempt's finding.
3. Return. The runner owns the tasks; nothing is prepared under prepared/ unless the task
   statement names a resource every session must find ready, and no attempt starts in
   setup.
