---
description: "What a paused Kapso campaign is waiting on you for, and the reply that resumes it"
argument-hint: "[campaign] [note]"
---
Follow the `kapso` skill (`/kapso:kapso`) for: $ARGUMENTS

With no campaign, the one `kapso inbox` lists as waiting. Show the request rows (key, hit, tried, fix, next); when the user says it is done, reply with a note through `kapso inbox reply <path> <id> "<note>" --detach`, never with a value. Pass `--config kapso-config.yaml` on every line when the project root holds that file.

When the user wants to tell a running campaign something it did not ask (a fact, a correction, a change of direction), leave it with `kapso inbox note <path> "<text>"`, the user's own words as one quoted argument. Nothing pauses or resumes: the session running now does not see the note, and every ideation, implementation and judging session that starts afterwards reads every note. A note cannot be removed, and never carries a value.
