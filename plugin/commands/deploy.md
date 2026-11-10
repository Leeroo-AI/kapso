---
description: "Deploy a Kapso campaign's delivered solution to a target (running software, or a harness's own delivery such as pull requests or a release), prove one call, and hand back the handle. Use it whenever the user asks to ship, deliver, open pull requests for, or release what a campaign built"
argument-hint: "[campaign] [--branch <attempt>] [--target local|docker|modal|bentoml|langgraph|<harness target>]"
---
Follow the `kapso` skill (`/kapso:kapso`) for: $ARGUMENTS

The campaign is the argument; with none, the newest finished one `kapso watch --all` lists. Run `kapso deploy <path> --target <t>` once (`--target auto` when none is given), in the foreground when the tool's timeout allows and once in the background writing to a log when it does not, then read the summary: the deployment's number, the proof call, the next actions pinned to the number. Pass `--config kapso-config.yaml` on every line when the project root holds that file.

A campaign built with a harness may have delivery targets of the harness's own (the skills of its deployment module that carry `scripts/deploy.sh`: pull requests, a release). A request to ship, deliver, open pull requests or release the result is a deployment through that target, never branches, pushes or pull requests made by hand: the target works from the delivered commit, checks what it opens and records it.
