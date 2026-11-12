# Deployment

- The launcher reads the endpoint from configuration, prints the scaffold version and the
  model it points at on start, and ships with its caps and deny rules on, since those
  were part of what was measured (deterministic-guards).
- Before handing over, run one complete permitted episode through the deployed scaffold
  against the live endpoint with the measured settings and a representative context size,
  and keep the transcript with the deployment record; on a locally served model run the
  local-model-serving smoke probe first, and tell advertised limits from tested capacity.
- Nothing from the evaluator ships: no task data, no answer keys, no development
  transcripts, nothing the task declares private. The ledger ships, so the next owner knows what each
  component is for and when to retire it.
