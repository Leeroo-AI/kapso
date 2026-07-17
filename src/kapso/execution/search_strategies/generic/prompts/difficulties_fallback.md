# Technical Difficulties Reconstruction

The implementation session below ended without reporting its
`<technical_difficulties>` (it may have crashed or been killed at its
deadline). Reconstruct that report from the evidence.

## What was being built

{{solution}}

## Evidence

- Session process record (raw stream-json events; thinking, tool calls,
  errors): `{{stream_artifact_path}}`
- The workspace you are in: training logs, PLAN.md, changes.log, and
  artifacts left by the session.

Read the evidence (grep the stream for errors, tracebacks, OOMs, retries,
kills) and reconstruct what the implementor struggled with.

## Required output

Return EXACTLY one tag as the last thing in your response:

<technical_difficulties>
Every difficulty the session hit — failed attempts, crashes, OOMs, silent
wrong results, misleading errors. For each: what happened (concrete error
signature or number), the root cause if determinable, and the fix that
worked (or "unresolved"). If the session was killed, state what it was
doing when it died. "none" only if the evidence shows a genuinely
uneventful build.
</technical_difficulties>
