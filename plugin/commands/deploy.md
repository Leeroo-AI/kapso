---
description: "Deploy a Kapso campaign's delivered solution to a target, prove one call, and hand back the handle"
argument-hint: "[campaign] [--branch <attempt>] [--target local|docker|modal|bentoml|langgraph|<harness target>]"
---
Follow the `kapso` skill (`/kapso:kapso`) for: $ARGUMENTS

The campaign is the argument; with none, the newest finished one `kapso watch --all` lists. Run `kapso deploy <path> --target <t>` once (`--target auto` when none is given), in the foreground when the tool's timeout allows and once in the background writing to a log when it does not, then read the summary: the deployment's number, the proof call, the next actions pinned to the number. Pass `--config kapso-config.yaml` on every line when the project root holds that file.
