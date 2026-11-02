# Decisions taken during the implementation, for the owner to check

Started 2026-10-24. The owner asked for the implementation to run
autonomously and for any architecture or design decision that was not
discussed to be written here so it can be checked later. Newest first.
Each entry: what was decided, why, what the alternative was, and where it
shows.

## 2026-10-26, WP7 (planned; taken while writing the sub-plan)

**The session-start brief rides as the MCP server's instructions, not a
hook.** The owner's decision of 2026-10-12 left the brief to the live-UI
layer; since WP3 the host-session server serves the intro of what it
loads as its instructions, which both hosts show the session on connect.
That is the brief: no `SessionStart` hook, nothing injected into every
prompt, nothing for Codex to lack. The plugin ships no hooks file.

**What a host session loads is a line in the project's config.**
`experience.session_sources` (a list of references, empty as packaged) names what
`kapso-mcp` loads when `--experience` is absent; `kapso-mcp` finds the
project's `kapso-config.yaml` by the one rule every command follows (the
git top level, else the working directory), so the one convention the
owner settled (decision 9: the project's config is passed by every
command) carries it. The alternatives: a plugin setting (the owner said
the plugin carries none), serving by existence (WP3 rejected it: a build
is served exactly what it loads, and a session should be too), or the
person adding a second MCP server by hand with `--experience mine` (what
the docs say today; it stays for a session without the plugin). A listed
reference whose source does not exist leaves the bank gate off with the
add line in the notes and the instructions, the other gates served: the
host starts the server without a person typing the flag, and a session
that only wanted the research tools must keep them (Codex's review); an
explicit `--experience` that does not exist stops the server with the
add line, as a build is refused.

**Six skills and ten commands, the commands pointing at the skills.**
A command is the explicit entry with arguments (`/kapso:build <wish>`),
a skill the procedure that also triggers on its description; a command's
body names the skill and the moment's CLI line and nothing else, so the
procedure is written once. Codex has no commands: the skills' names
(`$kapso-build`) and the sidecars' default prompts are its entries.
`research` and `bank` from the 2026-10-12 tree are gone with WP3's verbs
(`experience add --research`, `experience share`); `review` and
`harness` and `experience` join; `doctor` is not a command (one CLI line
the `kapso` skill knows and `/kapso:setup` runs), and `experience` stays
for the team's moment. A command's body carries no procedure.

**`kapso-evaluation` builds the evaluator for one approval.** The shipped
skill said "never write an evaluator on the user's behalf"; the owner's
revision of the skills document (2026-10-24) has the agent build or audit
it against the contract and ask once. The later word wins; the proof's
no-evaluator step is the check. The evaluator a session builds is passed
with `--evaluation` once approved, so it is protected from that moment,
the same as one the person brought.

**The campaign tools live in the host-session server, host sessions
only.** `kapso_build_start` and `kapso_status` wrap `kapso build --detach`
and `kapso watch --json` as subprocesses of the server, which runs outside
the host's shell sandbox (Codex ends a shell's background children; a
launch through the tool survives). They are a gate like the others, never
emitted for a campaign's own sessions, as the inbox gate is never emitted
for a host's. The owner's decision 7 put the tools after the commands;
they are the last change of this package, not a later one, since the
MASTER's scope names them.

**One copy of the operating skill, under the plugin.** The three per-host
folders under `skills/` were identical since the `--detach` rewrite; the
plugin's `skills/kapso/` is the one copy, with the Codex sidecar beside
it, and OpenCode (which reads `.claude/skills` and `.agents/skills`)
symlinks it. `skills/` is deleted (Rule 7); the README's agent note and
the docs point at the plugin's file.

**`kapso harness export` is the union staging.** The author flow's A4
(deferred from WP2) writes a host's layout from the harness's skills,
delegates, servers and hooks over every module, with the staging's own
per-host writers, and says what the host cannot reproduce: the routing
and the pages. It is a convenience for an ordinary session, not a second
way to run a build; a build stages per module as before.

**The launcher is Claude Code's path to the tools; Codex's is the
install.** The probe (`scripts/redesign/runs/wp7-hostfacts/FACTS.md`):
Claude Code expands `${CLAUDE_PLUGIN_ROOT}` in the plugin's MCP file and
runs the launcher; Codex takes the command verbatim (no variable, no
relative path), and a plugin-registered stdio server, though started and
answering, is not offered to a `codex exec` session, while `codex mcp
add` is. So the plugin ships one MCP file per host under
`hosts/<host>/mcp.json` and none at the root: Claude Code's names the
launcher (`kapso-mcp` on PATH, else `uvx --from leeroo-kapso==<pin>
kapso-mcp`, else `pipx run --spec leeroo-kapso==<pin> kapso-mcp`, else
one stderr line naming `/kapso:setup`; an executable found and failing
is never followed by another, its exit code is the launcher's); Codex's
and Cursor's name `kapso-mcp` on PATH, and the setup skill runs `codex
mcp add kapso -- kapso-mcp`, the form verified to reach a session. The
owner's decision 3 (the tools work from the first session wherever uv or
pipx exists) holds on Claude Code and narrows on Codex to "after the
install", by a host fact, not by choice. The pin is the package's
version, pinned by the version test; until the release it is not on PyPI
and a bare machine sees the setup line.

**The proof's Codex half runs on the development box.** The fresh
machine's Codex key is over its quota (WP4's finding); the Codex steps
run in a scratch project on the dev box with its ChatGPT login, the
Claude Code steps on the fresh machine. The owner's key moves the Codex
half to the machine; said in the plan.

**A plugin skill is `/kapso:kapso`, not `/kapso`.** Claude Code names a
plugin's skills under the plugin's namespace (verified: the probe's
`skills/kapso-probe/SKILL.md` answered as `/kapso-probe:kapso-probe`);
the owner's decision 10 kept the bare `/kapso`. The host decides; the
commands keep the owner's `/kapso:<verb>` form.

**The campaign gate is host-only by three means.** `GateDefinition.host_only`
keeps `get_mcp_config` from emitting it for a campaign session; the Codex
adapter passes `-c mcp_servers.kapso.enabled=false` to every campaign
session (verified to hide a user-config server's tools), since Codex
campaign sessions carry the person's own config (WP4's finding) and the
setup skill's `codex mcp add kapso` would otherwise ride into them; and
the gate's tools refuse by the working directory when the server runs
inside a campaign's session workspace. A nested campaign launched from
inside a campaign is the thing these prevent.

**A campaign can be stopped, and every long operation launches
detached.** `kapso build --output <path> --stop` ends the status file's
process group (SIGTERM, then SIGKILL after `budget.stop_grace_seconds`),
marks the status failed with `stopped by the person` and removes the
receipt; `--resume` continues from the checkpoint, which the orchestrator
writes per iteration (it handles no signal, and needs none). `kapso learn`
and `kapso experience add --repo|--research` take `--detach` through the
one detach helper. Codex's review found the plugin could launch but not
end a campaign, and left learn and ingest in the foreground for hours;
both are engine seams this package adds rather than procedures each
skill reinvents.

**`kapso harness export` writes files, not sessions, and no hooks.** The
skills copied, the delegates as the host's agent files, the servers as
`.mcp.json` or `codex mcp add` lines, the union over modules by
`Manifest.listed()`; hooks stay out (a verification Stop gate or a
prepared-directory deny would block an ordinary session), and the output
says so with the routing and the pages it cannot reproduce. The earlier
wording ("the staging's own writers") would have written a campaign
session's shape, which an ordinary session does not read.

**The evaluator the skill builds is protected, not versioned.** The
packaged config ships the maintainer off, so `--evaluation` protects the
directory file by file and registers nothing (`solution.evaluator` is
None); the `kapso-evaluation` contract names `result.json`'s fields, runs
the baseline in a copy so no `result.json` is left in the directory
passed with `--evaluation` (refused at launch), and keeps hold-out labels
out of both copied directories. "Versioned" is said only when the config
turns the maintainer on.

**The sidecar mirrors nothing.** `kapso-evaluation` is `user-invocable:
false` on Claude Code (hidden from the menu, the model's to call) and
`allow_implicit_invocation: true` on Codex, which has no menu hiding;
each host's file carries its own control, and the test checks each for
its own field. The earlier "mirror" would have disabled Codex's implicit
invocation, the opposite of the skills document.

**The host-session server connects even with every gate off.** Before,
`kapso-mcp` exited 1 when nothing could be served (no `codex` on PATH,
no repo memory, not in a campaign, nothing loaded), which is exactly a
fresh project: the host showed "failed to connect" as the engineer's
first impression (Claude Code's review). Now the campaign tools are
always served, and a session with every other gate off gets the notes in
the server's instructions. A host spawns its servers once at session
start, so the setup skill's closing lines say how the session reaches
the tools it just installed (`/mcp` reconnect on Claude Code, a new
session on Codex).

**The installed plugin is kept out of campaign sessions.** The probe:
a user-scope plugin's skills are listed by every `claude -p` and `codex
exec` session, so an implementation session could reach `kapso-build`
and launch a campaign inside a campaign. The Claude adapter passes
`enabledPlugins: {"kapso@kapso": false}` in every campaign session's
settings (verified to hide the skills); Codex has no per-session switch
(a `-c plugins."…".enabled=false` override did not hide them), so the
Codex adapter stages an execpolicy rule refusing `kapso build`, `kapso
learn` and `kapso deploy` inside its sessions (the mechanism WP2
verified) and the six descriptions say "never inside a Kapso campaign
session". The proof lists a campaign session's skills and tools on both
hosts.

**The Codex session's disabled server is defined whole.** The line
that keeps the person's own `kapso` server out of a campaign session
(`mcp_servers.kapso.enabled=false`) made Codex refuse its entire config
on a box with no such server ("invalid transport": a table with an
`enabled` key and no command), and every Codex session of both
machines' campaigns failed on it in the round; the adapter now names the
command beside the switch (`mcp_servers.kapso.command="kapso-mcp"`,
`enabled=false`), a complete definition of a server that never starts,
whether or not the person has one.

**The manifests list no skills or commands: discovery by convention.** The
Interface froze a `skills` list and a `commands` entry in the manifests;
the shipped manifests carry neither, and both proven hosts (Claude Code,
Codex) discover `skills/<name>/SKILL.md` and `commands/<name>.md` from the
plugin root. The tree test asserts the directories; a listing would be a
second copy of them. Cursor is unverified, and said so. (pull-request
review, Claude 15)

**The stop reaches every process under the writer.** `kapso build --stop`
signalled the status writer's process group; the coding-agent CLIs run in
sessions of their own and survived it. The stop now reads the process
table for everything under the writer and signals a group through its
leader and a process that shares its parent's group alone (a foreground
launch's script is never taken with it). The window between the liveness
check and the signal is not caught: CLAUDE.md's no-`try/except` rule, and
the traceback names the pid. (Codex 1, Claude 14 and 18)

**The Codex trigger arm grades a refusal on the first read.** A confusion
case fails when the forbidden skill was the first skill the session read;
a read after the expected skill passes and is counted in the report as
`read_after`. The earlier rule (any read beside the expected skill) made a
false positive unfalsifiable; the first read is the routing. (Codex 4,
Claude 13)

**A config-listed bank is fetched on first use when its remote answers.**
`experience.session_sources` names what every session is served; a bank
reference not yet in the store is cloned the first time, after a
`git ls-remote` by exit code, so a teammate's config works before any
`--experience` load; a remote that does not answer is a note, the gate
off. The parser returns a path or index reference that is not there for
the host session to judge (`must_exist=False`); `--experience` stays
strict. (Codex 5 and 6, Claude 9)

**The semantic budgets hold the rubric's caps.** Twelve lines outside
fences for the build and review messages and three setup messages, as the
skills document promises, even where the round's records exceed them: a
budget is the contract, the overrun a finding. (Codex 19, Claude 12)

**A question is a turn that ends asking.** The Interface's count; a
numbered list of questions in one turn is one question, so the interview's
eight-question cap bounds turns, not marks. (Claude 25)

**Codex's isolation is proven by a live probe step.** The rules file in a
gitignored subdirectory of a repository (an implementation session's
workspace) with the adapter's command lines rejects `kapso build`
(verified three ways 2026-10-27); the round's `codex-rules-probe` step
runs that case and records the refusal, deferred on Claude Code whose
mechanism is the plugin switch. (Claude 1, Codex 22)

## WP9: the semantic suite (planned, 2026-10-27)

**The simulated user is the other vendor's CLI with a card.** Codex plays
the engineer when Claude Code is the host and the reverse, prompted with
the persona card, the fixture's README, the whole history and the host's
message, answering one message at a time and `DONE` when satisfied. A
scripted policy stays available (`--user scripted`) for the mechanical
round WP7 ran. Alternative rejected: a bespoke API client for the persona,
which would add a provider dependency the CLIs already carry.

**The judge is Gemini on Vertex when reachable, else the idle vendor.**
The design asks for a third vendor; the credit project's `gemini-4-argon`
through the LiteLLM bridge is one, reached from the development box where
the judge runs over fetched transcripts. When it is not reachable the
vendor that played neither side judges, and the record names the model.
A judge that returns malformed JSON fails the run: no retry hides a rubric
it cannot follow.

**Two of the design's fixtures are not new directories.** `trading-strategy`
is `examples/trading` (the author's fixture since WP2); `empty-wish` is a
README the driver writes (a directory holding one README is no example).
`pipeline-dbt` is new, the third domain for the author test.

**The learn's two fail-loud cases are fixed in this package.** S6 needs a
learn that completes; the plugin round saw it fail twice, on a flow file's
unquoted fragment and on a crew that wrote no report. The repair is the
mining frame's own repair round extended to a document that does not
parse, and one retry for a crew session, then fail loud with the name;
`src/kapso/learning/` is edited here for those two sites only.

**The provider limit is injected by a shim.** A `PATH` shim before the
host's CLI answers the vendor's rate-limit error N times, copied from a
real one, then execs the CLI: the adapter's handling of that error is what
S5 tests, and a real limit cannot be scheduled.

**The record's second shape and the suite's one home (after the plan
review).** The round's record carried conversation counts only; the suite's
tests need attempts, injections, builds and readings, metric bars, a
two-pass judge and pins. `tests/semantic/suite.json` holds every knob and
bar (the sampling cadence stays the engine config's, read through
`load_config`), `report.py` computes the verdict the runner's exit status
carries, `budgets.json` is folded in and deleted, and WP7's three records
are re-derived into the shape (Rule 7). An arm that ran on the development
box is `advisory` in its record and cannot be a fresh-machine acceptance.

**The judge is a standalone two-pass over fetched records, with no
fallback vendor.** The reviewers showed the "idle vendor" does not exist
once Claude Code and Codex have played host and user. The judge runs on
the development box (Gemini on Vertex through LiteLLM, preflighted before
any machine run) over the rendered conversation per step, two blind passes
with a third adjudicating a straddle, the minimum over steps binding;
without the route the run is unjudged, the hard metrics stand and the
judged acceptance is blocked. Alternative rejected: a same-vendor judge
marked as a weakness, which would grade a vendor's own flow.

**The persona is offered every host turn and keeps its whole reply.** The
round's loop ended when a turn did not ask; a persona must be able to
challenge an answer. `DONE` alone ends a step; a turn cap is an explicit
incomplete; a persona-side CLI failure raises rather than reading as a
satisfied user. The driver's out-of-band acts (the key written before the
person says so; the harness's module answers as the trading author's
knowledge) are named in the record as interventions. A planted wrong fact
in the task view tells a persona that reads from one that nods.

**The Codex credit on the fresh machine is the owner's action, and every
Codex arm waits for it.** The key refuses a response with
`credit_balance_exhausted` (2026-10-27). Until credit exists, Codex arms
run on the development box under the owner's login and are advisory; the
fresh-machine arms that need Codex are incomplete, with the reason. No
model is swapped to make a run pass: the grader roles stay on Codex.

**S4 reads S5's tampering campaign; S6's learn runs on the development
box.** A reading on a campaign without a void or a rejection answers two
questions vacuously, so the tampering injection's campaign (a void with
hash evidence, a rejection, a partial cost) is the one read. S6's learn
needs the grader roles (Codex), so it runs on the development box over
the fetched campaign and its bank reaches the machine as a bare remote the
second build loads by reference: the second engineer's own path.

**S1's harness task uses a path, said as a deviation.** The design says "by
name"; the bundled directory holds no harness until WP8, so the task names
the trading harness at `./harness` and the record says so.

**The bounded exception to "a fix in its package".** A sentence in a merged
package's skill, command, trigger case or page that the suite shows wrong
may be fixed here when one sentence settles it; engine code never. The
review found the earlier wording ("a line") contradicting the ownership
list.

**A Codex campaign session trusts its own workspace.** The execpolicy
rule that refuses the campaign verbs lives in the workspace's project
layer (`.codex/rules/`), which Codex enforces for a trusted project
only; the adapter declares the trust on the command line
(`projects."<workspace>".trust_level="trusted"`), touching nothing under
`~/.codex`. The sessions already run with full-access sandboxing, so
the trust changes nothing else of the engine's; a seeded repository's
own `.codex/config.toml`, if it ships one, is read by its sessions from
now on. The campaign gate's refusal is by campaign directory, not by
session workspace: its three tools refuse whenever the server's working
directory lies inside a campaign (the `.kapso/experiment_history.json`
mark the host session already uses), so a person inspecting a finished
campaign from inside it runs the verbs in the shell.

**`kapso --version` and `kapso-mcp --version` exist.** The setup skill
tells one install from two by reading both, and neither command had the
flag: both print the package's version now, pinned by the version test
beside the manifests' and the launcher's.

**The trigger suite's pins.** Every case runs on `claude-opus-5-5`, the
model a Max session runs by default, twice, with six turns, three
minutes and the read-only tools (`Read`, `Glob`, `Grep`, `Skill`), so a
run is the routing decision and little else; a command that hands over
to a skill counts as that skill firing, in every case (the round showed
Claude Code answering "how is the campaign doing" by invoking
`/kapso:watch`, the plugin routing the request to the operating skill
through its command), and a near miss refuses the skill and the commands
that are its alone (`harness` hands over to two skills and is refused
for neither); the Codex runner
skips the explicit cases (Codex's explicit form is `$kapso-<skill>`,
proven in the host facts), reads a skill load from the stream's command
that read the `SKILL.md` or the rollout's `<skill>` message, pinned by a
captured fixture, and passes a near miss's refusal when the expected
skill was loaded as well: a Codex session reads skills as references
(the build skill sends it to the evaluation skill, the harness skill to
the operating one), so reading the skill under test beside the expected
one is routing by reading, not a misroute; the report names the first
skill read, and a case with no expected skill stays strict.

**The round's learn runs on another model when the token's window is
spent.** The fresh machine's Claude token had no weekly window left for
`claude-fable-5`, the model the learning crews ship on, while the
all-models window had room; the driver's `--learn-model claude-opus-5`
puts the crews on that model for the round through the project's
config, and the record says so. The shipped config is unchanged; a
person's campaign never has its models swapped by the engine.

**The plugin tree is kept out of the docs build.** `plugin/` joins
`.mintignore`: the trigger suite's prompts hold text the site's MDX
parser reads as tags (a `WAITING ON YOU` block with `<note>`), and
nothing under `plugin/` is a page.

**A session-built evaluator keeps its hold-out inside the evaluation
directory.** The one thing the product promise rests on: labels left
under `--data` are copied to `kapso_datasets/` and reachable by every
attempt. The contract pins the churn fixture's pattern (`private/` named
by `evaluation/private_dir.txt`, inside the directory passed with
`--evaluation`, protected with it), and the protocol names the split rule
and the leakage checks run; a criterion the data cannot measure is said
plainly.

**The version is a development number until the release.** `0.7.10` is
on PyPI without `kapso-mcp`: the launcher's pin to it would resolve a
real wheel and die on the missing entry point (Claude Code's review).
`0.8.0.dev1` is on no index, so the bootstrap branches fail to resolve
until WP11 ships, and the version test pins one number across the
package, the manifests and the pin; the marketplace entry carries none
(the manifest's wins: one copy fewer).

## 2026-10-26, WP5 (after the pull-request review of #160)

**The target's scripts are beside the record before the session runs.**
A named target's skill directory is copied into `deployments/<n>/target/`
right after the record is minted; under `auto` the session is told to
write the result file with `target` and `reasoning` the moment it
decides, and the engine copies the chosen target's directory as soon as
it reads that (at the acceptance, or when a failed session left a
parseable file naming an offered target). The plan said "copied at deploy
time", which the build read as "after the result passed", so a failed
deployment's `--stop` line could not run (both reviewers' first P1).
`Software.open` opens a record whose target was never chosen, and
`stop()` says there is nothing to run.

**The result file is validated at a process boundary.** Rule 2 says a
malformed JSON document raises; the Interface says a malformed result
comes back to the session as a finding. Both hold: the engine checks the
parse by running the interpreter on the file (`json.load` in a
subprocess, its exit code the verdict, its last stderr line the finding),
then reads what parsed. A session's sloppy file is a resume, never a
traceback, and the engine's own code catches nothing. The alternative, a
`DeployError` that fails the deployment at once, would spend the resume
the Interface promises on the most ordinary model slip.

**The record stays `deploying` through the acceptance, and the handle
that proved the deployment is the one returned.** `Software` has
state-free `_call` and `_answers` the acceptance uses, so `running` is
written once every check passed; no other process can call a deployment
still being proved, and a crash mid-acceptance leaves `deploying`, never
a false `running`. One handle is built before the session and returned
at the end with its worker alive (the proof call started it); a resume
closes it first, since the session may rewrite the module.

**HTTP transport failures are answers, read off a future.** `run()` on an
`http` deployment posts in a thread; the request's exception (refused,
reset, timed out) is read with `Future.exception()` and becomes the
`{"status": "error"}` answer the Interface promises, with the exception's
name and message. No `try`: the boundary is the executor's. A 2xx body
that is not JSON still raises, as the Interface says.

**The run kind is the target's to declare.** `metadata.run_kinds` in a
target skill's front matter (every bundled target and `paper-trading`
declare theirs) and the acceptance refuses a result naming another; a
`function` module must be a file of the workspace; artifacts must be
relative paths inside it. Without this a session could pass the checks
by pointing `run.url` at any service that answers 2xx.

**A tool's probe lives in its skill's front matter.** `metadata.requires`
takes an optional `probe` (`command`, `timeout`, `fix`); `readiness` and
the doctor run it bounded (killed at its timeout, a readiness line rather
than an exception). The docker skill declares `docker info`. The engine
no longer names `docker` anywhere: the two docker-specific branches
(`_readiness_line`, the doctor's daemon row) are gone, and any target can
declare what proves its tool usable.

**Env values are verbatim, in docker's format, and the worker has its
own six-line loader.** `.env` is `NAME=VALUE` per line, the value to the
end of the line: what `docker --env-file` reads. The worker reads it the
same way (no python-dotenv: it unquoted and expanded, so the function
and the container saw different values for `"a b"` or `${X}`), and the
local target's `deploy.sh` proves the entry point through the worker
itself instead of sourcing the file as shell. Names are matched whole;
values may not hold a line break, a carriage return or a NUL. The modal
skill says its dotenv secret unquotes.

**The engine notes what it does in `engine.log` beside the record.**
Stop and start lines, a stop script's failure, "the local target has
nothing to stop": written to the file as they happen, so a later handle's
`logs()` and a later `--logs` show them (before, they lived in the handle
that made them). `stop()` and `start()` return the lines they noted and
the verb prints them under its own, so a cloud stop that leaves a
platform deployment behind says so where the person is looking.

**A stop keeps a failed deployment's error.** `--stop` on a failed record
runs the scripts and writes `stopped` with the error kept, so `--status`
still says why it failed; `start()` clears it.

**Every child the engine runs is its own session, killed as a group.**
Lifecycle scripts, `run.py` and the worker start with
`start_new_session=True`; a timeout kills the process group, so a build
or a runner a script started dies with it and the deadline is a bound,
not a hope.

**Record writes are atomic and every change re-reads first.**
`DeploymentRecord.write` replaces the file through a temporary one;
`update(**changes)` reloads, applies, writes. Two handles on one
deployment never undo each other's fields; there is no inter-process
lock, which the plan's findings say.

**A new endpoint from a lifecycle script re-routes an `http`
deployment's calls.** `KAPSO_ENDPOINT=` from `start.sh` updates the
record's `endpoint` and, for the `http` kind, `result.run.url`: BentoCloud
gives a started deployment a new URL, and `run()` must follow it.

**The verb's lifecycle flags exclude each other, and a deploy flag beside
one is refused.** `--status --stop` chose silently before; argparse's
group refuses two lifecycle flags, the verb refuses `--target` with
`--stop` and `--number` on a deploy. `--status --number n` shows that one
row; a running row whose files are gone, a failed row's error and a
deploying row's requested target are shown.

**The summary's next step is Python, not JSON.** The proof's inputs were
printed with `json.dumps`, which emits `false`, `true` and `null` inside
a Python line; the line now carries the import, the path and the config
quoted for Python, and the inputs as Python literals; the CLI lines quote
the path for the shell and carry `--config`. The summary also prints the
engine's own answer to the proof call beside the session's when they
differ (a stateful target answers differently), and the interpreter a
function runs in.

**The engine's targets are listed by the deployment prompt, not the
guidance.** The plan's Consumed paragraph asked the guidance builder to
list them with the harness's skills; the build lists them in the prompt's
target block, with each one's readiness beside it, and the session record
names them under `engine_skills`. The guidance stays the harness's page.
Recorded here since the plan's text said otherwise.

## 2026-10-26, WP5 (taken while building)

**The front-matter reader lives in `kapso.core`.** The preflight imports
the targets' registry (for the doctor's rows), and the harness check
imports the preflight (for its requirement rows): a registry that read
front matter through the harness check would close a cycle. One reader,
`kapso.core.front_matter.read_front_matter`, serves the check, the staging
and the registry.

**`KAPSO_INTERPRETER` is one of the deployment's variables.** The engine's
own interpreter runs the function worker and every `run.py`, so the
solution's requirements must be installed into it and the scripts must
run it; the plan's list of variables lacked it. The recorded interpreter
is what every later lifecycle call uses, from any process.

**The deployment role is the roles module's.** `deployment_role(config,
coding_agent)` sits beside `check_deployment` in `kapso.core.roles`, so
the preflight's rows, the live tier and the session resolve the one spec
from one definition.

**An inference-only adapter as the deployment's host is refused by key,
not by the preflight's inference-only row.** The hosts table never holds
one (an adapter without a staging cannot enter it), so `deployment.cli:
openai_compatible` is `deployment.cli: openai_compatible is not in
hosts`, from the config or from the call; the verb's `--coding-agent`
keeps argparse's choice of the adapters that can code.

**Only the engine's targets are staged as engine skills.** A harness
target arrives with the harness's deployment module, as every skill of
the module does; staging it twice would list it twice in the record.

**The changed-harness check runs before the record is minted.** The
staging's hash check is shared (`check_harness_unchanged`) and the session
calls it first, so a harness edited since its load deploys nothing and
leaves no `deploying` record behind.

**The function worker takes the env file as an argument.** The engine
hands the worker the file's path on its command line and the worker loads
it before the import; reading `KAPSO_ENV_FILE` from the environment inside
the engine's own code would be configuration from the environment.

**The docker scripts probe health with the interpreter, not curl.** The
proof machine's host may lack `curl`; `urllib` in `$KAPSO_INTERPRETER` is
always there. The `.dockerignore` lines are appended once each, the
person's own lines kept.

**`logs()` says when its script failed.** A logs script that exits
non-zero would otherwise read as "no logs"; the line names the exit and
the stderr.

**`Software.open` takes a config path for the timeouts.** The handle
needs `call_timeout` and `script_timeout`; `open(campaign, number,
config=None)` reads them from the file named, the packaged defaults
without one, as the CLI's `--config` threads it.

**The cutover and the trading example's target landed in one commit.**
The example's test needs the session, the session's harness-target test
needs the example's scripts; one commit keeps both green (the plan listed
them as changes 6 and 7).

**The retired `--strategy` is scoped to the deploy verb.** RelBench's
runner keeps `--strategy generic|tree` for its search strategy; the
guard's pattern is `kapso deploy … --strategy`.

## 2026-10-26, WP5 (planned; taken while writing the sub-plan, revised after its review)

**One deployment session, not a selector and an adapter.** Today's
selector session exists because the adapter is prompted with one target's
instructions pasted in; with the targets as skills the session opens the
skill it needs, so the choice under `auto` and the adaptation are one
session's work: the choice first, with its reasoning and the readiness of
every offered target, then the adaptation. One record, one cost, one
place to read. Alternative rejected: keeping the selector as a cheaper
first session, which would stage the skills twice and record a choice
nobody can act on without the second session.

**The deployment result is a file the engine names and checks, not tags
in the output.** `kapso_deployment.json` at the workspace root, in the
shape WP2's `verdict.json` set for verification. The engine accepts it
only against what it can observe: the session completed, the shape, the
target requested or offered, the artifacts on disk, the health script,
and the proof call run again by the engine. A completed session whose
file fails a check is resumed once with the finding under the same
deadline; a session killed or failed is a failed deployment, never a
missing file. The `<run_interface>` and `<endpoint_url>` tags parsed out
of the transcript go with the adapter.

**A deployment has an identity of its own and a record from its first
second.** `<slug>-<hash6>-<n>` and a port bound free at allocation, both
recorded; `deployments/<n>/` minted by an atomic `mkdir`; the record
written with `state: deploying` before the session starts and completed
at its end (`running`, or `failed` with the error), so what `deploy.sh`
started is always reachable through `Software.open(campaign, n).stop()`
or `kapso deploy --stop`. Re-deploying takes a new number and leaves the
earlier deployment running under its own name and port. Why: the review
showed the first plan's path-derived identity letting deployment 1's
`stop()` remove deployment 2's container, and a bad result leaving a live
container nobody could reach.

**The record is the whole handle.** It carries the commit, both harness
identities and the harness root, the interpreter, the name and port, the
values the session recorded, the options, the env names (the values live
only in the workspace's `.env`), the target's skill directory copied
beside it (the scripts every later call runs), and paths relative to its
own directory, so a moved campaign still opens and a removed harness or
package version does not change what a deployment runs.

**Three run kinds; the `function` kind runs in a worker process.** `http`
and `script` as before; `function` is a worker per handle
(`kapso.deployment.worker`) that imports the module once and answers one
JSON line per call, started in the workspace with the `.env` values:
isolation by process, the model loaded once, a reopen from any
interpreter, and no module eviction or `sys.path` surgery in the engine.
A raising function ends its worker and the traceback is the error; the
next call starts one again (the rules forbid a catch in the worker, and
the exit code is the honest channel). The `none` kind of the first plan
is gone: a target with nothing to call ships no `run.py`, and the
paper-trading target's `run()` is one cycle on demand, which is the flow's
"the engineer tries one cycle".

**The lifecycle is the target's scripts, with a contract.** `bash` for
`.sh`, `sys.executable` for `run.py`, the workspace as working directory,
the deployment's values as `KAPSO_*` variables (transport of resolved
values, never configuration read by the engine), `script_timeout` and
`call_timeout` from the config, stdout lines `KAPSO_ENDPOINT=` and
`KAPSO_VALUE_<KEY>=` writing back into the record (a cloud target's new
URL after a restart), a non-zero exit as `DeployError` with the stderr
except in `stop()`. The scripts are the target's, never the session's to
edit. Deterministic parts of today's runners move into them: docker saves
its logs before removing the container and recreates it from the recorded
values (the container port, the GPU request); modal stops with `--yes`;
bentoml asks BentoCloud for its URL and posts with the SDK's own
credentials; langgraph keeps its thread in the workspace.

**`options` is the channel the example harness already documented.**
`deploy(options={"schedule": …})`, `--option k=v`, `KAPSO_OPTION_<NAME>`
on the scripts, recorded; a bundled target reads none. Why: the trading
example's `paper-trading` skill and `harness-design.md` §3 both say the
deployment module receives the call's options, and the first plan had no
way to pass one.

**A harness target is a deployment skill that holds `scripts/deploy.sh`.**
The structural mark of a workflow, so a reference skill listed on the
module (`market-data`) is not a target and no manifest field names one
(`harness-design.md`: no engine-facing fields). A harness target named
like a bundled target or `auto` is refused when the harness is loaded for
a deploy. `metadata.requires` is read from the engine's bundled targets
only; a harness target's tools are `environment.commands`, which `check`
already reads.

**The `deployment` block is a session spec over the hosts table, plus the
target, three deadlines and the ports.** `cli: null` follows
`search_strategy.params.host`, so a project that moves to Codex changes
one key; `model: null` takes the host's, which removes a packaged
duplicate of `hosts.claude_code.model`; `timeout` (the session),
`call_timeout` (a call), `script_timeout` (a lifecycle script),
`port_base` and `port_span` are the operational values the rules want in
the config. `coding_agent` leaves the block, `--model` the verb, `model=`
the call, as they left `build()`.

**The workspace is its own repository under `.kapso/deployments/<n>/`.**
A copy of the campaign's working tree at the delivered commit (dirty
tracked files refused by name), without the engine's and the hosts' own
directories, with the untracked data the solution reads, then `git init`
and one commit, so a session's git commands and the staging's collision
check stop at the workspace and never reach the campaign above it. The
docker target writes a `.dockerignore` so `.env` and the staged files stay
out of its image. Nothing new appears beside the campaign; the page says
what a campaign grows by.

**The verb stops, starts and reads what it started.** `kapso deploy
<campaign> --status | --stop | --start | --logs [--number N]`, through
`Software.open`; `--interactive` is gone (for the `function` kind it
blocked the one process that could call the software; for the others the
deployment outlives the process anyway). A CLI user is never sent to
Python to stop what the CLI started.

**The engine's target skills are staged like the engine's hook.** The
staging takes `engine_skills`, directories copied beside the harness's
skills into the session's skills directory on either host, with or
without a harness (the no-harness path stages them and writes the
record), in `writes()` and the record, listed by the guidance builder with
the host's opening syntax. Alternative rejected: the deployment package
copying into `.claude/skills` or `.agents/skills` itself, which would
duplicate the staging's knowledge of the hosts.

**`DeployError` is the engine's own refusal around a deployment**, printed
as one line by the CLI with the other engine errors; WP1's two refusals on
the solution keep their wording and take the type, and a dirty campaign
joins them.

**The engine skill's deploy section is this package's.** Deploy once in
the foreground with a tool timeout above the session's; the record is
written at the start; reopen by number; stop with the verb. The plugin
command stays WP7's. Why: the repository's own findings record a coding
agent backgrounding the verb and polling to its turn limit.

**The cloud targets ship unproven live.** Modal, BentoCloud and LangGraph
Platform need accounts the proof machine does not have; their scripts are
what today's runners did, exercised against fake executables, and the
release gate runs them where the owner's credentials are. The plan says
so rather than claiming a proof it cannot make.

## 2026-10-26, WP4 (taken while building)

**The hosts a build runs on are the adapters with a staging, declared in
the registry.** `agents.yaml` names the staging each build-capable adapter
gets (`host: claude_code` on the three Claude adapters, `host: codex` on
Codex); `check_hosts` refuses the other adapters with those four, and the
staging's own table is read from the same key. This reverses the earlier
entry that every registered adapter stages and the staging's refusal is
unreachable: the pull-request review showed `hosts.gemini` passing
validation and failing at the first session. Alternative rejected: a
staging for `gemini`, `openhands`, `aider` and `openai_compatible`, which
have no layout to stage into. Shows in the hosts table and the `-a` row.

**A judge session that fails to start leaves the attempt unjudged with
its error.** The generator reads the invocation's `success`: a failed
launch (a quota, a login revoked, a crashed CLI) is `the judge's session
failed: <error>`, nothing to parse or resume; the raise for a completed
turn with no session handle stays, that being a wiring bug. The feedback
flow's existing catch marks the attempt unjudged too, so no scored
candidate stays deliverable after a judge error. Why: the review showed
the proof run's exhausted quota ending a campaign with a traceback in the
one state the field exists for.

**An implementation not launched is an attempt with no branch.** The lane
returns the attempt with the error and `unjudged` saying the
implementation was not launched; nothing resolves its commit, verifies or
judges it. Why: the review reached GitPython's `BadName` from a refused
session.

**Ideation deadlines are shares, never raised, bounded by their own
timeouts.** The sixty-second floors on a member's share and on its retry
window are gone (a share under the floor is a short session, as
everywhere else at the budget's end); the selector's share is bounded by
the room read when it launches, the members done, so time they left
flows to it and time they spent is not promised twice; a member's or the
selector's own `timeout` bounds its deadline further. The split test
reads the clamp twice: the shares from the first reading, the selector's
bound from the second.

**The web switch disables Codex's native search by config.** With `web:
false` the adapter and the raw ideation pass `web_search="disabled"`
(verified against the CLI, which accepts `disabled` and rejects a
nonsense value); omitting `--search` alone leaves the CLI's default
cached search on.

**A cut is attributed from the invocation it cut.** The judge's rooms
(the first invocation's, the retry's) are remembered and the finishing
cut noted when the room of the one that was cut was the budget's; a retry
refused for want of room is a finishing cut too. A verification's cut is
attributed from the invocation's own outcome (killed at its deadline
without completing), not from a comparison of its timeouts, so a verifier
that fails at once with room to spare keeps its retry.

**The trace's `stopped` row carries the sentence the campaign stopped
on.** The status file's recent ring holds it as `stopping: …` (the
admission refusal, the finishing cut); the trace reads the last such line
into `stop_note` and renders it after the detail. Alternative rejected: a
new status key, since the ring already carries the sentence and the
prefix is the convention.

**The resolver refuses a mapping where a value stood, expands `~`, and
opens the adapters' own maps.** `enabled: {typo: false}` is refused as
`bool expected, got a mapping` (a mapping is truthy, so it would have
read as true); `learner.ingestor.agent_specific`,
`learner.merger.agent_specific` and the new `inference.default.
agent_specific: {}` are open namespaces; a spec's `cli`, `model` and
`effort` must be non-empty strings when set, a list or `false` refused
rather than read as unset. `ConfigError`'s docstring is narrowed to what
the messages carry: the key and what is allowed, the file from the
resolver; a build has one override file, so the key locates the line.

**The orchestrator validates its config at construction, as the facade
and the doctor do.** A benchmark runner constructs it directly. And the
RelBench runner threads the bank serving as the engine's override (as the
facade does), not as a key of its runtime file, which the resolver
refuses.

**The engine's own sessions: one record each, only their own servers.**
The engine's inference sessions on Claude set `strict_mcp_config`, as
every build session does; the session records are numbered under the
staging lock, so parallel lanes minting their commit messages take
distinct records.

**The preflight skips the ideation role's row under an ensemble.** The
members' sessions replace the role's own, so its host is not asked for.

**The budget's floors are sourced from the packaged config** at import
(`PACKAGED_BUDGET`); the dataclass defaults are no longer literals that
could drift from `budget.min_iteration_seconds`.

**The admission's finishing span is said, not changed.** The round's
duration counts its finishing, which the reserve also holds back; the
admission errs early by about one finishing span, never late. The page
says so; the owner may want the round measured to its last lane's
implementation instead.

**The replanner's guidance is staged for the planner's own host.** The
one staging call that still named `claude_code` follows
`ideation.lens_planner.cli`, so a Codex planner reads Codex's guidance
and no Claude layout is written into the ideation worktree.

**The engine recorder is wired by the strategy, at the workspace's
birth.** The earlier decision wired it in the orchestrator after the
strategy existed, which left the strategy constructor's own inference
session (the repo-memory bootstrap) unrecorded in every campaign of the
proof run; the S4 readers found the gap. The strategy owns the workspace
and its records directory, so it wires the recorder the moment the
workspace exists and before its first inference. The import the earlier
decision avoided (`kapso.core.cli_inference` into the harness package) is
not needed: the strategy already imports the staging.

**The derived reserve is bounded like a set one.** The plan bounded a
set `reserve_minutes` by a quarter of the budget and left the derived
reserve to the obligations themselves. The proof run's ten-minute trading
build derived 11.5 minutes (the harness's ten verification minutes, the
judge's floor, the delivery) and launched its setup session with a
negative deadline; the forty-minute builds kept under three quarters of
their minutes for the search. Both reserves now take the quarter bound;
a finishing session takes what its room leaves, as before. And the setup
session follows the launch predicate every other session follows: a room
under the floor is a refusal with the reason, since the setup is the
campaign's precondition.

**The retired-name check reads code and prose differently.** A retired
config key is a leftover wherever it stands in a page, a README, a skill
or a config file, and in code only in its quoted-key form (`params[
"implementation_gates"]`, a dict literal): the strategy keeps attributes
and arguments of those names for what it resolved (`self.ideation_gates`,
`implementation_fallback_model=`), which are identifiers, not
configuration. The two session timeouts stay the benchmark runners' own
cap names in code and are retired as config keys in prose only. The
tests that refuse a retired form by name are allowed by path.

**The live and manual tests take override files.** The inbox live
fixture writes an override file for every variant (the gates off, the
implementation lane on the CLI under test) and marks the fixtures that
launch through the Python API with a file, so the CLI form passes
`--config` where it passed `-m MINIMAL`; the manual Bedrock test's config
is a hosts entry. Neither suite runs under the gate; both now describe
the design that exists.

**A changed harness is refused by file name.** The loaded harness keeps
each file's own digest beside the whole-tree hash, and the staging's
refusal says `changed a; added b; removed c` with the way out. Why: a
hash alone sends the person diffing a directory to find what a session's
shell (or their own editor) touched; the per-file digests live on the
loaded harness in memory, not in the launch record, which keeps the one
hash it has always kept.

**The hooks file is written for every staged session.** With the engine's
`protect-harness` hook there is no harness session without hooks, so the
file is written, listed among the staging's writes and recorded on every
module; a session repository that tracks `.claude/harness-hooks.json` or
`.codex/hooks.json` is refused by name for every module, as it was for
one with hooks of its own.

**The engine sessions' record is wired in, not imported.** The
inference client (`kapso.core`) records its sessions through a callable
the orchestrator sets once the workspace exists (the staging's
`engine_session_record` over the campaign's records directory); the
client imports nothing from the harness package. Why: `kapso.core
.cli_inference` importing `kapso.harness.staging` closed an import cycle
(staging reaches the adapters, the adapters' modules reach the client).
An inference with no campaign records nothing.

**The reconstruction's record says `implementation`, labelled
`difficulties/<branch>`.** It is staged with the implementation module
(its host, its servers and hooks, in the campaign directory like the
setup session) under its own session id; the trace's label grammar
places it with its attempt. Alternative rejected: a `difficulties`
module of the harness manifest, which would make every harness author
decide what a reconstruction may see.

**The judge says "its deadline"; the strategy says whose.** The
generator does not know whether the deadline that cut the judge was the
budget's room or the role's own timeout, so its reason reads `the
judge's session ended at its deadline`; the strategy, which computed
the deadline, rewrites it to `the budget's deadline` and reports the
finishing cut only when the budget set it. Why: a judge cut by its own
configured timeout with an hour of budget left is not a budget's end,
and stopping the campaign on it would be wrong. The plan's wording is
kept for the case it described. Shows in `feedback_generator.py`,
`strategy.py`.

**No verdict is a field of the result, not an exception.**
`FeedbackResult.unjudged` carries the reason (no tags after a retry,
the deadline, no room for a retry); the flow sets `SearchNode.unjudged`
from it and skips the verdict. The one exception kept is the missing
session handle, a wiring bug. Alternative rejected: raising for every
unjudged outcome, which would turn a budget's end into a failed
campaign.

**The launch predicate guards the lane's sessions; ideation's inner
sessions take the room they have.** The implementation, the evaluation
run, the verification, the judge and the reconstruction each check
their room against the floor before launching and record why when
refused. The ensemble members, the selector and the lens planner start
inside an admitted round (admission already covered the floor and the
last round), so they are clamped to their room without a launch check:
a selector that finds less than the floor runs with what remains and is
cut at it, recorded as timed out, rather than skipped with a second
mechanism inside the ideation module. Why: one predicate at the five
sites where a skip has a record to land in; the ideation sub-sessions'
case is an edge of an edge (a round admitted with barely the floor).
Shows in `strategy.py` (`_room`, `_clamped_timeout`,
`_finishing_deadline`), `budget.py`.

**The obligations are counted at the floor, not at their configured
timeouts.** A finishing session's room is the remainder less one floor
per session still to run (the evaluation run in cost mode, the
verification when the harness verifies, the judge) plus the delivery's
slice. Why: the plan says each is "protected at its floor"; counting
configured timeouts (ten minutes of verification, say) would hand the
end of a campaign to the reserve and starve the attempt in flight. The
derived reserve is the other side: there the verification's configured
minutes and the judge's observed duration count in full, so the
searchable window is cut for the real finishing cost while a session
already running is only asked to leave its successors their floors.

**The round's duration lives in the strategy's state.** `round_seconds`
is a key of `strategy_state` (written by `dump_state`, read strictly by
`load_state`), not a field of the checkpoint dataclass: the strategy
measures and consumes it, and the checkpoint carries the state whole.
The schema stays 6.

**The escrowed attempt's end is `attempt_would_not_fit`.** After the
fidelity policy's reserved full-size run, the campaign stops with the
admission detail (no further attempt fits) rather than a third detail:
the closed set has two new members, as planned.

**The delivery's slice without a fidelity measurement is thirty
seconds.** The final checkout and the records; a structural constant
(`DELIVERY_SECONDS`), not a knob, like the quarter bound on a set
reserve (`RESERVE_FRACTION_BOUND`).

**`deny-web` is the implementation module's; `deny-web-fetch` stays
everywhere.** The trading example already denied the shell's fetchers
(`curl`, `wget`, `aria2c`) on every module; the plan's `deny-web` is a
second hook, on the native web tools (`WebSearch|WebFetch`), listed on
implementation alone: ideation's web research is the example's intent
(the engine's `ideation.web` default), the implementation's web is the
leak the protocol forbids. Why two hooks: two rules, two reasons, each
placed where it applies by the manifest. Alternative rejected: folding
both into one hook on every module, which would take the web from
ideation. Codex's native web search is not a tool call its hooks see;
the engine's switch is the boundary there, and the page says so.

**The axes live in one tag, status included.** The planner's first plan
names the axes in an `<axes>` block (a list numbered from 1); a revision
restates every one in the same block as `N. <axis> — <status>`, and the
record keeps those lines as the axes, so the judge and the next
replanner read the latest status beside each axis; a keep puts its
statuses in its paragraph and carries the axes along. Why: one grammar
to parse and one field in the record, and the status is what the judge's
frontier report builds on. Alternatives rejected: a separate
`<axis_status>` tag and an `axis_statuses` field (two things to keep in
step, no reader that wanted them apart); statuses required on a keep
(a terse keep would be discarded for a formality). Shows in
`lens_planning.py`, the two planner prompts, `axis_frontier.md`.

**Single-session ideation keeps the ensemble's candidate hygiene.** The
plain path is the ensemble machinery with one member, so a candidate
under the degenerate floor (eighty characters, or forty of content) is
dropped and the fallback solution is implemented, where the old Claude-
only path took any output as the plan. Why: one machinery, one rule; a
plan that short is a skeleton on any host, and the fallback keeps the
round alive. Shows in `ideation.py`; the direct-construction tests give
their fake agents plan-sized outputs.

**A host the table does not name is a config error, named by where the
host is set.** `search_strategy.params.host (what --coding-agent sets):
gemini is not in hosts (the hosts are claude_code, codex); add
hosts.gemini: {model: …} to the config to use it`, a `ValueError` from
the resolver before the preflight, like every other config refusal; a
role's own `cli` is named as that role's key. The harness staging's
"no staging" refusal is unreachable through the facade now: the hosts
table is checked against the adapter registry, and every registered
adapter stages. Alternative rejected: a preflight row (the preflight
checks the machine against the config; a config that names nothing the
engine knows is not a machine fact).

**An open namespace exists in the schema, empty.** `knowledge_search.
params: {}` is in the packaged file: the walk allows any key under an
open namespace, but the namespace's own key must be one the packaged
mapping names, so an override could not set it until the packaged file
carried it. The readers take `["params"]`, not `.get`. The rule holds
for every open namespace: `hosts`, `inference.roles`, `gates`,
`session_env_defaults` all exist in the packaged file.

**A mapping in the packaged file is a mapping in an override.** The walk
refuses a scalar where the packaged value is a mapping (`inference:
nonsense` → `must be a mapping, got str`) instead of letting the merge
replace the block and the reader die later. Shows in `config.py`.

**The launch record's `config_path` is `None` for the packaged
defaults, however they were named.** `is_packaged()` (the resolver's
own test, exported) decides it at record and resume time, so a resume
from another install — a different site-packages path for the same
packaged file — compares equal, as the old `_launch_config_path` made
it. Shows in `kapso.py`.

**The trace fixture generator writes the launch record at the current
schema.** The archives predate the record without `mode`; the generator
reconstructs the record (schema 2, no `mode`) as it already
reconstructs the checkpoint's and the solution record's schemas, and the
three fixtures were regenerated from their archives (the golden texts
unchanged). The handle's words (`experiments`, `minutes`, `usd`,
`reserve_minutes`, `host`) come with change 5.

## 2026-10-26, WP4 (planned; taken while writing the sub-plan)

**The build's sessions are roles over a hosts table.** The config gains
`hosts.<cli>: {model, …}` and `search_strategy.params.host`; ideation,
implementation and the judge are role blocks (`cli`, `model`, `effort`,
`timeout`) that default to the host and its model; `--coding-agent` sets
the host for the campaign. Why: the flag reached a field no session read
(WP2's and WP6's proof runs ran every session on Claude Code under
`--coding-agent codex`), a model the CLI could not serve was found at the
first session, and three blocks (`coding_agent`, `feedback_generator`,
the flat `*_model` parameters) named hosts and models with no rule
between them. Alternatives rejected: routing the flag to
`implementation_cli` alone (ideation and the judge stay elsewhere and the
model mismatch remains); deleting the flag (the plugin's build on the
host it runs in wants one word). The judge is a role under the strategy's
parameters, not a top-level block: it is the strategy's session, named
beside the sessions it judges. Shows in `config.yaml`,
`search_strategies/generic/roles.py`, the preflight.

**Unknown top-level keys are refused; `benchmark:` is reserved.** An
override file may name only the packaged file's top-level keys; the
runners' own settings live under `benchmark`, which the engine never
reads. Why: a mode leftover or a typo did nothing silently (the "a config
copy's `defaults:` is ignored" trap WP3 met), and the runners' keys
(`run_defaults`, `final_eval`, their own `preflight`) collide with the
engine's at the top level. Alternative rejected: a full nested schema;
the open mappings (env names, inference roles, hosts) would need a schema
language, while the blocks with validators (`budget`,
`evaluation_maintainer`, the ensemble members, the lens planner, and now
the strategy's parameters) keep refusing their unknown keys.

**`MINIMAL` is not shipped.** A config that turns things off is an
override file a person writes; the configuration page shows one.
Alternative rejected: a shipped `minimal.yaml`, a second shape to keep in
step with the first.

**The web stays a switch per role, the engine's boundary; a harness that
forbids the web lists a hook, the domain rule.** `ideation.web` and
`implementation.web` replace `web_search` and `implementation_web` with
one name and one meaning on both hosts: off, the two native tools leave
Claude Code's allow-list and go on `--disallowedTools`, and Codex is not
given `--search`; the selector and the lens planner follow ideation's.
The master's row said `implementation_web` becomes a hook; the review
showed the knob is the only binding denial Codex has (a hook may not see
its native search) and the only one a config without a harness has at
all, so the switch is kept as the engine's permission boundary and the
hook is the per-module domain rule a harness adds. The first draft
deleted both knobs; revised after the review.

**The axes are the planner's output.** The lens planner names the design
axes it derives from the goal and the ideation page in an `<axes>` tag;
the plan file carries them; the replanner statuses them; the judge's
prompt renders the plan's axes, or asks the judge to name the ones it
sees. Why: the harness design's section 9 recommends it, and the five ML
axes in `lens_planning.py` were domain prose in engine code. Alternative
rejected: an axes list in the harness's ideation page that the engine
parses, an engine-facing field in a page.

**The problem text is the brief; the evaluation contract is rendered once,
in the implementation prompt.** The generic handler renders the brief and
the two reference lines (where the evaluation and the data live) and
nothing else; the implementation prompt carries the contract in three
forms (a provided suite: run it, never edit it; a registered one: the
maintainer's; none: build your own). The first draft wrote the task
statement for every build; the review showed that without a harness
nothing mounts or names the file, so it stays a harness build's file.
Alternative rejected: keeping the handler's prose, two renderings of the
same facts that drifted (the handler said `result.json` may be created,
the prompt said where to write it, and the provided case got the
build-your-own text).

**The budget's end: finishing obligations protected, a launch predicate,
admission from the last round, the reserve derived.** A finishing session
(the candidate's evaluation run, verification, the judge, the
reconstruction) is clamped against the whole remainder less the
obligations after it, each at its floor, the delivery's protected slice
kept; a session is launched only when its room is at least the floor,
and a deadline is never raised to the floor; a judge or verification the
budget cuts leaves the attempt `unjudged` or `unverified` with the
reason, its score kept, never delivered; an ordinary round is admitted
when the searchable remainder covers the larger of the static floor and
the last completed round's duration (the insured campaign keeps its own
floor, the fidelity policy's reserved attempt its own gate); the reserve
is derived from the obligations when `budget.reserve_minutes` is unset
(the harness's verification minutes, the judge's observed duration, the
delivery's slice) and bounded by a quarter of the budget when set. Why:
every campaign of WP2's, WP3's and WP6's proof runs that reached its
budget ended on a verification cut at its clamped deadline and a judge on
the sixty-second floor, rejecting a scored attempt. The first draft gave
finishing sessions the whole remainder, floored the clamp and shipped a
ten-minute reserve; the review showed the judge starving after
verification, a reserve refusing short builds, the insured floor lost
and the fidelity slice unprotected. Alternatives rejected: raising
`min_iteration_seconds` (one number for every task), per-phase fractions
(the IOAI runner's recorded lesson: a phase sub-budget starved ideation),
marking a cut attempt invalid (the loss WP6 found, by another route).

**The judge's retry resumes its session, records its reason, and does not
follow a deadline.** Why: both S4 readers of WP6's proof run had to dig
the retry's reason out of the log, and a fresh session answered with none
of the context. A deadline-killed judge is the budget's end, not a
missing tag.

**`result.json` stays in the protected directory; code added there is
the tamper.** Moving the contract's output would break every evaluator
written so far; the integrity verifier already rejects added source and
executables, and what was missing was the rule stated in the pages.

**The maintainer is off by default, its knobs in the config.** It runs
its own calibration and registration sessions per build, and most builds
bring an evaluator or a harness verification; the packaged block carries
`enabled: false` and every knob the orchestrator held as a literal
(Rule 1; the first draft shipped `null` and left seven literals in code),
so a person turns it on with one key and sees what they are tuning.

**No lighter ideation for small briefs here.** An effort change is a
quality change nobody has measured; S1's time-to-first-candidate is
recorded on this package's proof run first, and the budget's end makes a
short campaign stop honestly rather than deliver a half-built attempt.

**The budget block speaks the handle's words.** `budget.minutes`,
`experiments`, `usd`, `reserve_minutes` replace `time_budget_minutes`,
`cost_budget_usd` and `finalization_reserve_minutes`, and the facade's
`DEFAULT_MAX_ITERATIONS` becomes `budget.experiments` (WP1 asked for the
literal to move with the block). One word per dimension across the brief,
the flags, the API and the config.

**`agents.yaml` loses its model defaults and gains `serves` and
`settings`; `build_config` requires a model.** The `default_model` lines
(`claude-opus-4-5` for Claude Code) duplicated the config's model and had
gone stale (Rule 1); the hosts table is the one home. `serves` names the
model family an adapter can be asked for (`claude`, `openai`, `any`), so
the preflight's pairing row is the adapter's fact and the OSS and Vertex
adapters, which run Claude Code with other models, are not refused;
`settings` names the keys a host entry may set for that adapter, so a
misspelled adapter setting is refused rather than ignored.

**The engine's `protect-harness` hook rides every staged session.** The
harness directory is the author's: WP3's proof lost a setup session's
minutes and dollars to a patched harness, and the identity check that
refused the campaign was right while the rule before it was missing.
Alternative rejected: staging the servers from a copy, whose `cwd` and
`${PLUGIN_ROOT}` paths would diverge from the author's tree while the
hash still moved.

**The empty-workspace initialisation session is deleted.** A model call
to write an empty `main()`, unreachable from the facade, which always
names a workspace; the workspace writes the file.

**`--config` stays explicit.** Nothing is read from the working directory
unnamed: the brief is named with `--brief`, and a stray
`kapso-config.yaml` applied silently is the class of surprise the
resolver exists to remove. The build's banner and the doctor's first line
name the layers.

**The strategy's parameters are a known set, every default in the
packaged file.** The gate-list literals WP3 flagged and every other
`params.get(key, literal)` move into `config.yaml`; the strategy reads
without defaults; a misspelled parameter is refused by name.

## 2026-10-25, WP6 (taken while building)

**A run's files are the host's account; the delivered files are the
diff's.** Both adapters took `files_changed` from a git status of the
session directory, which listed the engine's own `.kapso/` directories
for a judge session and nothing for a Codex implementation that had
committed its work (the proof run's Codex-hosted build delivered a
candidate whose trace said `files changed: none recorded`). The run
entry now carries what the host itself reported writing: the edit
tools' inputs on Claude Code, the `file_change` items on Codex,
relative to the session directory. A file written through a shell
command is not in that account, so the trace's delivered candidate
names the files its branch's diff touches (the checkpoint node's
`code_diff`), the ground truth of what the branch changed. Two facts,
two sources, neither a fallback for the other. Alternative rejected:
keeping the git status and filtering the engine's paths, which still
misses a committed change and still depends on when the status was
taken. Shows in `kapso.execution.coding_agents.transcripts` (`files`),
`session_record.record_run` and `kapso.execution.trace._delivered`.

**A launch is registered before its checks, and a receipt keeps it
listed.** The proof run's first launch died at the evaluation suite's
refusal before the campaign directory existed; `kapso watch campaign`
said `no status yet` (the receipt was read only once the directory
existed) and `kapso watch --all` knew nothing of it (the registry was
written after the orchestrator's construction, and its reader dropped
any path with no directory). Now a launch with an output directory is
registered as soon as the brief is read, before the evaluation refusal
and the preflight (a resume was registered by its launch; a build with
no output directory is registered once the orchestrator has made one);
the registry's reader keeps a path with a launch receipt beside it; and
the watch verb reads the receipt whether or not the directory exists.
A foreground launch that dies at a check leaves a registry line and
nothing else, which the listing skips. Alternative rejected: writing
the receipt into the registry's own line, which would make the registry
a second receipt. Shows in `Kapso.build`, `kapso.execution.inbox.
list_registered_campaigns` and `kapso.cli.watch`.

**The trace's JSON carries keys the plan did not list.** `sessions`
(top level): the session records that belong to no attempt, the harness
setup and a node still in flight, which the attempts' `sessions` cannot
hold and the failures section must still see. `branch`: the branch the
view was narrowed to, so `TraceView(data)` renders the narrowed text
again from the data alone. `header.status`: the status file's state,
phase and error, so a partial trace says why it is partial (running at a
phase, failed with its error, no status yet, or a record older than the
checkpoint). `attempts[].evaluation.output`: the evaluator's output, the
evidence of a rejection and of a crashed evaluator (S5's injection).
`lessons[].kind`: a hosted source renders differently from a bank (its
reads are not logged per card). `attempts[].cost_usd` and
`duration_seconds`: the node's own figures, "what it cost" per attempt.
After S4's first reading, two more: `header.cost.unpriced_runs`, the
Claude Code runs killed before their result event, whose spend the
total lacks (both readers worked it out from a `cost n/a` line), and
`attempts[].cards.listed`, the cards a session saw listed beside the
ones it read (`none (15 listed)` is not `none`). After the second,
`sessions[].hooks_staged`: a delivered block saying `hooks none fired
in its sessions` meant nothing when no hook was staged, and now says
so. After the pull-request review, `header.requests` (a campaign
waiting on the person renders `partial` with the count, as the listing
counts it alive), and the run entry's own keys beyond the sketch,
`cost_cumulative` (the host's figure the delta is taken from), `usage`
(the tokens, Codex's only figure) and `host_errors` (what the host
itself reported wrong), each on the trace page. Alternative rejected:
deriving each from the text's sections at the reader's end, which WP7's
skills would have to repeat. Shows in
`kapso.execution.trace` and `tests/test_trace.py`; WP7 reads the keys as
pinned there.

**The attempts come from the checkpoint's node history, not the
experiment history store.** The checkpoint's node carries the
verification, the load-bearing cards, the phase telemetry, the parent
branch and the evaluation output; the store (`experiment_history.json`)
is the gate's ranked view with fewer fields. A session's attempt is
found by the label grammar the staging uses (the branch for the
implementation, `verification/<branch>`, `feedback/<branch>`,
`ideation/iter<n>` for the n-th node), so no record has to name its
node. `complete` is decided by content, the record's `experiments_total`
against the checkpoint's `completed_iterations` with the status `done`,
not by file times, which an archive or a copy does not keep.

**The fixture generator reconstructs what the archives predate.** The
three archived proof runs were recorded before the run entries, the pull
log's session key and the current record schemas existed. The committed
generator (`tests/fixtures/trace/build_fixture.py`) rebuilds the session
records by splitting each Claude Code stream at its `result` events (a
trailing slice without one is a killed invocation, its error the
adapter's deadline text), times each run from the node's start and its
phases' durations, stamps each pull event with the session whose window
holds its time, lifts the records to the current schemas, blanks the
checkpoint's `agent_output` (the trace does not read it; the diffs it
does, for the delivered files, so they stay) and rebases the campaign's
own path to `/campaign`. The pull-request review found the generator
splitting a stream at its `result` events, a turn each: an invocation
begins at its `init` event (the hook events just before it are its own)
and holds every turn until the next; one that ended without a result
and was resumed is an `error` entry that says so, one killed at its
deadline a `timeout`. It is deterministic and the test proves it reproduces the
committed fixtures when the archives are on the machine. Alternative
rejected: hand-written fixtures, which could not carry a real delegate
report, a real void verdict or a real resumed session.

**A session staged again keeps its runs.** Re-staging the same label (a
phase re-run after a crash, a resume) used to rewrite the record with
`runs: []`, losing the entry the failed invocation had written in its
`finally`. The staging now keeps the entries already in the record at
that path, so the trace shows how the first invocation ended and that a
second one followed. Shows in `kapso.harness.staging.existing_runs`.

**The summary's label column is seventeen wide.** `Sessions failed:` is
the longest label; every pinned row in the docs and the tests moved two
columns. A shorter label (`Failures:`) would have read as failed
experiments.

## 2026-10-25, WP6 (planned; taken while revising the sub-plan after review)

**Every session has a record, and the adapters write its runs.** The
staging writes `session.json` for every staged session, harness or not
(WP2 wrote one only with a harness), and each adapter appends one run
entry per host invocation at the end of that invocation, in its own
`finally`, so a session that raises, resumes or retries is recorded as
it happened. Alternative rejected: a `finish(result)` call at every
site that runs a session, which has no result when the session raises
and misses the two Codex paths that bypass the adapters.

**`done` stays in `solve()`; the delivery runs inside it.** The facade
hands `solve()` a `finish` callable (the checkout and the record) that
runs inside the orchestrator's protected region before the terminal
publish, and the checkpoint completes after it. Alternative rejected:
publishing `done` from the facade after the record, which would strand
the five benchmark runners that call `solve()` directly and the tests
that pin `done` after it.

**A launch receipt beside the log.** `--detach` writes
`<campaign>.launch.json` with the child's pid and the log's path before
the child starts, so `kapso watch` can tell a campaign that is starting
from one that died before its status. Alternative rejected: a first
status frame written by the CLI, which the strategy constructor's own
`BuildStatus` would overwrite.

**The liveness rule is the view's.** A campaign is listed when
`OperationStatusView.alive is not False` (`starting` included, and a
machine without `/proc` judged by the heartbeat) or when it waits on
the person; the registry is written on every launch, inbox on or off;
the running learns and ingests are listed from `learning.status_dir`,
which takes `watch --all` off the observability design's deferred list.
`kapso inbox` keeps listing requests. `watch --tree` is retired: the
trace renders the attempts, and two renderers over one file would
drift.

**The trace prints evidence, not a merge verdict.** The six questions of
S4 are answered by a reader from the sections; the recommendation is
`kapso-review`'s, as its specification says. The verb keeps the master's
name, `kapso trace`, though WP1 retired `Solution.trace` because the
word promised a log: the trace renders the record and says so on its
page. The cost line carries its coverage (`claude_code` when any Codex
run exists), extending the status surface's time-only rule to a record
reader that already prints the campaign's cost.

**A refusal is read from the host's words, for now.** The Claude Code
adapter recognises the model's safeguard refusal by the result's text
(`API Error` and `safeguards flagged`), the shape the proof learns met;
the pull-request review asked for the structured signal WP3's finding
named (`model_refusal_no_fallback`), which is the API's error type
inside that same text, not a field of the result event. Until the host
exposes one, the words are the signal; a refusal worded otherwise would
be a failed turn with the words in its run entry, not a silent success,
since the digest keeps the result's text and the entry its error.

**The engine's errors print their message, on stderr.** For the
engine's own error classes the CLI prints the message alone (however
many lines, as the preflight's report is) and no traceback; any other
exception keeps its traceback. The CLI's own `Error:` refusals move
from stdout to stderr with them, so a script reading stdout sees
results only.

**The pull log carries the session.** The staging passes the session's
label to the bank gate's environment and every event records it, so
reads are attributed to the session that made them; an ask of WP3's
package, merged, taken here.

**The refusal is a result field and a config count.**
`CodingResult.refused` is set by the Claude Code adapter on the model's
safeguard refusal; the mining frame relaunches a refused lead
`learning.mining.refusal_relaunches` times (one) before any repair
round, and notes it through the learn status it is handed.

**`stopped_reason` and `requests` keep their names.** WP1 left the
rename open; both names are in the status file, the checkpoint and the
API, and a rename gains nothing.

## 2026-10-25, WP3 (taken after the sub-plan's review, while building)

**Scope is a filter again, on the harness coordinate.** The 2026-08
amendment made scope displayed information ("the index lists the whole
bank; the reading agent selects") because the `family:`/`dataset:`
coordinates were too fine: a card scoped to one dataset was useful on the
next. The coordinate is now the harness, which is the domain itself, so a
card measured under the predictive-modelling harness is noise in a
trading build. `mine` loaded at the default scope lists and reads only the
cards whose scope covers the coordinate (`domain`, or `harness:<name>`);
`mine:all` lifts that; a bank you read serves everything it holds;
quarantine is law under every scope. Alternative rejected: keep scope
displayed and delete the `scope` knob, which would have made the harness
coordinate decorative. Shows in `retriever.py` (the index and both reads
take the scope), the intro's count, `docs/experience/`.

**Loading never touches the network beyond a first fetch; `kapso
experience update` does.** The sub-plan had `mine` pull from its remote
on every load. Both reviewers showed what that does to a host session
(`kapso-mcp --experience mine` on a machine whose remote is unreachable
exits at every session start) and to a resume (the pin and the pull
compete). A load pins the local state; `update [ref]` pulls `mine`
fast-forward from its origin and re-fetches a bank you read at a moving
reference; `remove <ref>` deletes a clone or an index. A pinned commit
never moves; a campaign keeps its pin whatever `update` does later.
Alternative rejected: pull on load with a fallback to the local head when
the remote fails — a silent fallback (rule 2).

**`mine:all` is the lift's syntax everywhere.** `scope="all"` existed only
in Python; the CLI and the brief had no way to say it. `--experience
mine:all`, the brief's `Experience: mine:all`, `kapso-mcp --experience
mine:all` and `load_experience("mine:all")` are one form; `scope=` on
`load_experience` goes. A suffix on any other reference is refused by
name (a bank you read, the hosted gate and an index are always every
scope).

**The coordinate is the harness alone.** `build(serving_scope=)`, its
launch-record key and the `dataset` coordinate are retired with
`family:`; `KAPSO_TASK_HARNESS` is the one coordinate the gate reads.
Alternative rejected: keep `dataset` as a benchmark's extra coordinate —
a benchmark is a harness now (WP8), and a second coordinate would need
its own scope grammar, writer rule and tests for one caller.

**`none` is a reserved harness name.** `harness:none` is the coordinate of
a campaign with no harness, and WP2's name grammar allowed a harness
called `none`; `check` refuses the name. Alternative rejected: an
unnameable coordinate such as `harness:-`, which reads worse in a card.

**A card is identified by its source and its name; the citation grammar
stays `[card:name]`.** With two loaded banks holding one name, the gate
reads from the named source or the first that holds it and the pull log
records the source of every served row; the learning side credits a cited
name to the source the pull log shows it was read from, and reports a
name read from two sources as an ambiguity instead of guessing. The
evidence tool takes `source=` as the read does. Alternative rejected: a
qualified citation `[card:name@source]` — every prompt, parser and grader
that reads citations would change for the rare collision.

**One index per campaign.** The search gates hold one backend; two loaded
indexes are refused by name at planning. Alternative rejected: federated
search across indexes with result provenance, which no backend supports
today and which WP4 may revisit with the config.

**The loaded experience is campaign identity.** The references and their
scope enter the checkpoint's fingerprint; a resume loads the record's list
and re-pins each bank at the recorded head, and a resume that names a
different list is refused by name, as a changed harness is. Alternative
rejected: an operator dial like the budget — a campaign served a different
bank in its second half would be measuring two things.

**`kapso-mcp` serves the bundled gates only, by what is loaded and what
the directory holds.** `--gates` does not come back (the three
directory-driven gates turn themselves off with a reason; `kg` carries
write tools and never rode on a host); `--harness <ref>` gives the
session its coordinate, as a build has one; the intro rides as the MCP
server's `instructions`, so the session names its sources on connect;
`leeroopedia` loaded on the host prints the reminder to add it as the
host's own HTTP server; the session's staging directory is removed when
the server exits; a source that cannot be loaded stops the server with
the line that would fix it (no partial serving).

**The intro has its own slot.** `handler.additional_context` is frozen at
construction, so the intro appended after it never reached a session (WP2
recorded this). `{{experience_intro}}` is threaded into the ideation and
implementation prompts the way WP2 threads the harness guidance, and into
no other session; the test renders the prompts.

**Probe offers ride only your own bank.** A probe is a measurement you pay
for whose evidence lands in a bank you can write; a colleague's open
probes are not yours to run. The gate builds offers from `mine` alone.

**No `--experience` on `deploy` or `learn`.** Deploy's sessions mount no
gates at all (`deployment/selector`, `deployment/adapter`), so there is
nothing to serve; WP5 revisits when it adds staging. `learn` writes `mine`
whatever is loaded; `Kapso(bank=)` points `mine` elsewhere and is threaded
through every crew. The master plan's "on the three verbs" was the
harness interface's line.

**Team banks: `share` stays push-only.** Joining a team bank that already
has history, bare-repository synchronisation and divergence recovery are a
later package; `share` refuses a push the remote rejects, in git's words,
and the intro says "measured by you and your team" when `mine` has a
remote. Papers stay out the same way: no ingestor reads one; the
onboarding grammar marks `--paper` as not shipped, owner: the experience
package's second iteration.

**No `family:` knowledge anywhere.** The sub-plan had `kapso experience
list` mark a bank whose cards carry the retired coordinate; that needs
every card parsed on every `list` and keeps the old grammar alive in code
(rule 7). `list` is silent; a pre-release bank is re-learned.

**An index owns its namespace in both graph stores: a Weaviate collection
named after it and a Neo4j `corpus` property every node and query carry.**
The reviewers showed that two indexes holding a page of the same id
shared one Neo4j node, so adding a corpus changed what an earlier index's
"connected pages" returned. The graph backend now takes `corpus` as a
required parameter (the index's id), writes it on every node, matches on
it in every query and clears only its own nodes; the collection name is
`Kapso<Name><id>`, so a name removed and added again never meets its old
pages. The record's `backend_refs` carries both, which is how a loaded
index connects. Alternative rejected: one label per index, which needs
dynamic Cypher and gives nothing a property does not.

**`kapso learn <campaign>` is the verb's form; the learning system's own
commands move under `kapso learn dev`.** WP2 handed `kapso learn
<campaign> [--harness]` to this package. argparse cannot give one verb a
positional campaign beside nine subcommands (`import`, `mine`, `grade`,
`update`, `develop`, `codify`, `ingest`, `gauntlet`, `behave`): the
positional swallows the subcommand's name, or the subcommand parser
refuses the path. Those nine are the development regime's tools, for
people working on the learning system, so they live under `kapso learn
dev <command>` with the same flags, and the engineer's form is the plain
one. Alternative rejected: a subcommand named `campaign`, which is not
what the onboarding grammar and the docs promise.

**The learning crews are staged in full, and a harvest leaves the hosts'
staging directories out of a bundle.** The pages-only block the crews'
prompts carried is gone: the mining lead is staged at the bundle root like
any session (the learning module's skills, servers, delegates and hooks
beside the crew's own agent definitions and the contract), the update lead
in its run directory, and the update crew's `codex exec` workers in the
same directory in codex's format, `run-role.sh` exporting the harness's
variables and passing the staged servers, agent roles and hook trust, the
guidance read from a file beside the launcher. For a bundle to take a
staging and stay as harvested, the harvest leaves `.claude`, `.agents` and
`.codex` out at every level: they are the harness's files, recoverable from
the hash the manifest records, and the session records under
`.kapso/sessions`, which stay, say what each session ran with. The mining
crew owns those names at a bundle root for its session and refuses any
other occupant by name. Alternatives rejected: hiding a harvested `.claude`
for the session's length (the immutability check would have to learn the
hidden path, and the lead would see a directory the inventory does not
name); a symlinked view of the bundle for the session (the record and the
stream would land in the bundle through the links).

**A mining run has a directory of its own, `learning.mining.run_root`.**
Staging writes a session's record and stream under the campaign's
`.kapso/sessions`; a bundle takes no writes, so `stage_harness` takes a
`records_dir`, and a mining run's go to
`<run_root>/<trajectory>/.kapso/sessions/lead/`, an update run's to
`<run_dir>/.kapso/sessions/{lead,workers}/`, as every other crew keeps its
runs under `learning/`. The mined view itself stays in the bundle.

**The guidance names the task statement only when the campaign has one.**
A learning run's directory has none and an archived campaign has its
build's; a build writes the statement before its sessions, and the setup
session, which writes the domain statement the task statement is composed
from, has the path in its own prompt.

**A bare name is never a path.** The plan let a bare name that is a
directory be read in place, and the reviewers showed the trap: the
checkout an index was made from, beside the run, shadowed the index of
the same name, and the name could then neither be loaded nor removed. A
path is written as one (`./`, `../`, `/`, `~`) and must exist; a bare
name is a reserved name, a git shorthand or an index. The forms table is
corrected.

**A bank read by path is re-loaded from its recorded location, named as
the launch named it.** A resume or an inbox reply may run far from the
launch directory; the record's `resolved` is absolute, so that is what a
resume re-loads, and the source keeps the launch's spelling of its
reference so every record and the fingerprint agree. A reply rebuilds
the object with the bank the launch had (`Kapso(bank=)` is the record's
`mine` entry).

**A bank at a tag is pinned at the tag.** `bank_head` reads the commit a
bank is at (`HEAD`: `main` in a bare home, the checked-out revision in a
clone), not `main`, so `acme/kapso-bank@v1` records, counts and serves
the tag's commit. One clone directory per reference: the directory's
name carries a digest of the reference, so two references that read
alike as names never share a clone.

**An index pin holds at the build.** A handle that loaded index A while
the name was removed and ingested again as B before the build is refused
at the build, as a resume with a stale record is: a campaign is served
the index it loaded, never one that happens to carry the name now.

**Attribution starts at reading.** The exposure ladder carries three
rungs; a citation is credited to your bank when the card was opened from
it (`read` or `evidence-read`), a copy only listed by `bank_index` was
never read, and ambiguity means opened from both sources. A claim of
independence is refuted by any exposure, a listing included. An
unqualified `bank_get_card` with two banks goes to the first bank that
can serve the name under its scope, so what the index advertises a plain
read opens.

**The hindcast and the behaviour scenarios replay the surface a campaign
loading `mine` is served.** `compile_intro` moved to the serving module
and composes the same intro, for `mine` at its default scope under the
task's coordinate; the index they grade is rendered under that scope. A
card of another harness is not availability the grader can charge a
campaign with. The behaviour fixtures recorded with the earlier
coordinates stay exempt from the drift test and are re-recorded when WP8
ports the RelBench harness.

**A lesson that admits nothing moves nothing.** A crew that changed no
card, sighting or quarantine entry used to commit the rebuilt log alone,
so the bank's head moved and the lesson read as admitted. The frame now
commits, tags and pushes only when the bank's content changed; otherwise
the report carries one head before and after and `learn()` says the
lesson admitted nothing.

**`kapso experience add leeroopedia` probes the gate once.** The plan's
Interface promised it: the add sends the hosted server one MCP
`initialize` with the key on its URL and reports the server's name; a
refusal or an unreachable server stops the add with the status; an empty
`LEEROOPEDIA_API_KEY=` is no key. Loading stays a presence check. The
loaded source carries the URL and timeout of the config it came from,
and the sessions mount exactly those (`get_mcp_config(hosted=)`), never
the packaged address; a config gate list naming the hosted gate does not
mount it, since what is loaded is what is served and the records name it.

**A learn may run under a harness the build did not have.** `kapso learn
<campaign> --harness <ref>` on a campaign whose record names none loads
the harness for the crews (its learning module shapes their sessions)
while the lessons keep the build's own coordinate (`none`); a deploy
still runs as its build did.

**The hosts' staging names include the adapter's MCP directory.** The
Claude Code adapter writes `.claude_mcp/` beside a session; the frames
call the adapter's cleanup after the lead's session, and the harvest
leaves the directory out with the other staging directories.

**The research ingest's modes and depth are config.** `experience.
research_modes` and `experience.research_depth` replace two module
literals; the researcher's findings are unwrapped into the typed
sources the pipeline ingests.

**A `result.json` left in the evaluation directory is refused at
launch, not skipped.** (Taken during the proof run.) The evaluator
writes `kapso_evaluation/result.json` by contract, allowed as a new
runtime artifact; a copy an earlier run left at the root of the caller's
directory was protected with the suite, and the first evaluation changed
it and rejected every candidate. The launch now refuses the leftover by
name, with the way out in the message, at the facade's input check and
at the orchestrator's manifest. Alternative rejected: leaving the file
out of the manifest and the copy silently, which would split the
integrity rule ("every file in the suite is protected") with a special
case, reach the vendored archive sandbox and its byte-identical copies
in the benchmark suites, and still leave a stale result in the
workspace for a session to read. Whether the contract should move the
result out of the protected tree altogether is WP4's.

## 2026-10-25, WP2 (taken while building)

**A harness's Python requirements are checked against `python3` on PATH,
not the engine's interpreter.** The sub-plan said `importlib.metadata` in
the engine's process. The harness's servers, hooks and scripts run on the
`python3` the authoring guide tells the author to name, which is the
engine's interpreter only when Kapso is installed into an activated
environment; a `uv tool` install (the fresh machine, the install skill)
runs the engine in a venv of its own with no `pip`, so the fix line
`<sys.executable> -m pip install -r` could not even run there, and a user
who followed it would have installed into the interpreter nothing of the
harness uses. `check` now asks `python3` on PATH for its distributions in
a subprocess, reports its absence as a need of every module, and the fix
line is `python3 -m pip install -r <file>`. Alternative rejected: checking
both interpreters, which doubles the rows and still names the wrong one.

**A session that names no directory is refused, not staged into the
working directory.** Found by a test whose nodes had no directory and
left `.claude/` and `kapso_task.md` in the repository root; production
nodes always carry one. The guard sits after the no-harness path, which
writes nothing, so a build without a harness is untouched.

**The verification session runs on the implementation CLI, not the
judge's.** The sub-plan had it on the judge's host; the review pointed
out that a campaign whose attempts are implemented on Codex then had
its verification on Claude Code, so the Codex rows of the host table
(hooks, roles) were never exercised by the step that matters most.
The attempt's own host verifies it, with the implementation model.
Alternative rejected: a `verification_cli` knob, a knob nobody asked for.

**An `env` entry of a harness server names the variable it forwards, and
every `env` or `headers` value is one whole reference.** Claude Code
expands `${VAR}` wherever it sits and would accept
`"BROKER_API_KEY": "${MY_KEY}"` and `"Bearer ${TOKEN}"`; Codex forwards
variables by name and takes neither. Rather than translate per host (a
value on a command line, or a literal header), `check` refuses both
shapes and the guide says why. The `Bearer` prefix lives inside the
variable.

**A module's staging into a shared directory is exclusive.** Sessions at
the campaign root follow one another; the second used to find the
first's skills beside its own, which contradicted "nothing is inherited"
and the session's own record. Each staging removes the harness's skills,
hooks file and delegates that are not this module's. Alternative
rejected: a directory per session at the root, which moves the judge
and the maintainer out of the campaign root they read.

**A verification session that fails is said so, and no retry runs under
the budget's deadline.** The verdict's summary carries the session's
error (a kill, a crash) instead of "wrote no verdict", so the judge and
the record can tell a dead session from a silent one; a second session
is launched only when the first had its configured time, since the
budget's clamp would give the retry less still.

**A loaded harness is never attached to a campaign built without one.**
`deploy()` and `learn()` refuse it by name; a verb on a finished build
runs as the build did. Alternative rejected: silently adding the
deployment or learning module after the fact, which the Interface
forbids.

**The store's fetch record sits beside the clone.** Inside it, the record
changed the hash of a harness at a repository's root on every fetch and
overwrote an author's own `source.json`; two revisions whose names
sanitise to one directory are refused rather than served as each other.

## 2026-10-24, WP1 (after the third proof run)

**`solution.feedback` is the judge's words on the delivered candidate; the
last judgment only when nothing was delivered.** The third proof run
delivered node 0 (0.8474) and the record's feedback was node 1's failed
judgment ("the required tags were missing after a retry"), because
`build()` took `final_feedback` from the solve result. It now reads the
delivered node's feedback from the strategy, and `explain()` labels the
block `Feedback:` rather than `Last feedback:`. Alternative rejected:
carrying both, which doubles a field nobody asked for.

## 2026-10-24, WP2 (planned; taken while writing the sub-plan)

**The engine never installs a harness's requirements; it names what is
missing and the line that installs it.** The design says `build()`
"prepares the harness's requirements in the run's environment". A harness
can arrive by git URL, and an engine that runs `pip install -r` on whatever
it fetched is a supply-chain hole the engineer never agreed to. So
`check`, `harness add`, `doctor` and the refusal in `build()` report each
missing distribution with `pip install -r <path>`; the engineer runs it
once per machine. Alternative rejected: installing silently.

**Delegates are the hosts' own sub-agents on both CLIs, not engine-run
sessions behind a tool.** The design allowed either ("registered as
sub-agents, or run as their own session where the CLI has none"). Both
hosts have sub-agents: Claude Code takes `--agents` with the name,
description and prompt (verified: the model asks it by name and gets the
reply), and Codex's `multi_agent` is stable. Native agents inherit the
session's skills, servers and hooks for free and the host records their
transcripts; an engine-run session behind an MCP tool would sit inside a
tool call with the host's tool timeout around it. The one open point is
whether Codex takes named roles per invocation; the sub-plan carries the
fallback (spawn with the agent file's text as the first message).

**The task statement is an untracked file at the campaign root, copied
into every session directory; prepared resources are a real directory at
the root and a symlink everywhere else.** `kapso_evaluation/` is committed
to the workspace's main, which is why it reaches every clone; a task
statement committed that way would land in every candidate branch and in
the delivered code. Untracked and ignored, it reaches sessions the way
dropped data does (`_bring_dropped_data`), and the ideation worktree gets
a copy at staging. A snapshot under `prepared/` can be large, so one copy
at the root and a link per session; the `protect-prepared` hook is the
guard the harness chose, and the design says a hook is not a boundary.
Alternative rejected: committing both to main.

**A verification session that writes no verdict is run once more; still
nothing, and the attempt is `unverified`: it keeps its score and stays a
parent, is never delivered while the harness has verification, and the
judge is told.** The first draft left such an attempt valid; both
reviewers objected that an unchecked candidate could then be delivered on
a verifier timeout, which is the one outcome a verification promise must
exclude. A malformed verdict still raises (rule 2). Alternative rejected:
treating a missing verdict as void, which would discard good attempts on
a formatting slip, and treating it as valid, which would deliver them
unverified.

**Export moves to WP7.** An export has to carry `servers/`, the
requirements, the hook scripts and the delegates with their references
rewritten per host, and prove the result loads; that is host plugin
layout work, which WP7 owns, and nothing in a build needs it. The design's
surface line points at WP7.

**The recently-used list of local harnesses is dropped.** `list` shows
what is bundled and what is fetched; a local path is used by path. One
JSON file and a `forget` command were the price of one row of output.

**Delegates on Codex are named roles, verified, so the design's "own
session" fallback is not built.** `agents.<name>` with a description and
a config file whose developer instructions are the prompt; the role is
spawned by name and the host's sub-agent events carry it. One mechanism
per host, no runtime fallback.

**Prerequisites are scoped to the modules a verb runs.** The trading
harness's deployment-only broker server would otherwise refuse every
build on a laptop with no broker credential. A build refuses on the
requirements of setup, ideation, implementation, verification and
feedback and prints the rest as advisory; deploy and learn refuse on
their own module's.

**Setup and verification run only when the harness has content for them.**
A harness with no `setup.md` and no setup attachment gets no setup
session; the same for verification. The engine's deterministic parts
(the directory setup, the integrity check, the registered evaluation)
always run. A build with no harness is byte-for-byte today's behaviour;
nothing new runs for it. Alternative rejected: a setup session for every
build, which would add minutes to the first-build time WP1 already found
long.

**A loaded harness wins over the brief's `Harness` line, and a path in
that line resolves against the working directory.** The first is the rule
WP1 set for budgets (call arguments over the brief); the launch print
names the harness used so the override is visible. The second is what
every other path the CLI takes does; a brief that is run from its own
directory, as the examples say, can use a relative path. Alternative
rejected: resolving against the brief file's directory, which would make
`Brief` carry a base path and break `parse(render())`.

**Harness servers on Codex are mounted with `default_tools_approval_mode =
"approve"`.** The probe showed a stdio server's tool refused under a
read-only or workspace-write sandbox with no key and with `auto` ("MCP
tool call requires approval, but approval policy is never") and running
with `approve`. The shipped adapter uses `auto` for hosted gates; whether
that binds anywhere today is for the package that next touches a Codex
inference session (noted in the sub-plan).

**`export --format cursor` is dropped from WP2.** The design listed three
formats; Cursor's plugin layout is not verified from here and nobody has
asked for it. Two formats ship; a third is one function when wanted.

**The sub-plan was written before WP1's pull request was opened, on the
same branch.** The third WP1 proof run was in progress, and the plan does
not depend on its result; the planning document rides in WP1's PR as a
documentation commit, and WP2's code branch is cut from `redesign` after
the merge. Alternative rejected: idling for an hour.

## 2026-10-24, WP1 (planned; taken while revising the sub-plan after review)

**The brief has its own flag, `--brief`; `--goal-file` stays plain text.**
The design documents said `--goal-file kapso-brief.md`. A file that is not
a brief must not be guessed at (no silent fallback), and engineers do pass
plain problem statements by file today, so the two forms get two flags.
Alternative rejected: one flag that tries the brief grammar first.

**A solution's one path is `trace`.** `SolutionResult.code_path` and the
design's `Solution.trace` were the same directory: the workspace, checked
out at the delivered branch, with the record under `.kapso/`. One field,
named `trace`; deployment reads the code from it.

**`build()` writes `.kapso/solution.json` on every return, and `open()`
reads only that.** The alternative, reconstructing a solution from five
record files, needs rules for which file wins and has no durable
evaluation log to read. The facade already computes the delivered score,
the branch and the registry head at return; writing them is a few lines,
and a paused or budget-stopped campaign gets a record too.

**`Experience` is WP3's.** The design returns `Experience` from `learn()`;
today's `LessonResult` is that class. Renaming it belongs with the
experience work, not with `build`, so WP1 leaves `learn()` returning
`LessonResult` and only changes its parameter type.

**`kapso deploy` takes a solution, not a directory of code.** Today's
command constructs a result from any path. In the design `deploy` takes
the outcome of a build, so the command opens the record; deploying an
arbitrary directory is not a case of this verb. WP5 may add a `--repo`
form if a user needs it.

**`Kapso(config=...)`.** The constructor keyword follows the design; the
internal attribute keeps its name.

**A sentence goal is rendered as a brief too, so the engine always sees one
grammar.** `build("Raise AUC above 0.85")` renders `Objective    Raise AUC
above 0.85` into the goal text, the checkpoint fingerprints that, and a
resume parses it back; `kapso inbox` reads the objective's first line the
same way. Alternative rejected: passing a plain string through untouched,
which would have left the checkpoint holding two shapes and the resume
guessing which one it had.

**The brief keeps paragraphs and pasted indentation.** Blank lines between
two lines of the same field stay as paragraph breaks (elsewhere they are
nothing), and a field's continuation lines are dedented together, so a
pasted traceback or code block reaches the model with its relative
indentation. Rule 6 asked for the content whole; this is what "whole"
means for a multi-line objective. Alternative rejected: stripping every
continuation line, which flattens code.

**The `Solution` dropped three things the old result carried.**
`experiment_logs` (one line per node, re-derivable from the experiment
history and WP6's `kapso trace`), `resumed` (the checkpoint knows) and
`status_path` (always `trace/.kapso/status.json`). Each was a field nobody
acted on; rule 10.

**`final_report.json` carries flat keys instead of an opaque `metadata`
dictionary.** `learn()` writes `goal`, `succeeded`, `final_score`,
`stopped_reason`, `iterations`, `cumulative_iterations`, `cost_usd`. The
report's readers (the trajectory store, the mining crew) never read
`metadata` by key; they read the file as prose, so named keys serve them
better than a dump of the old dictionary.

**The first-build budget is forty-five minutes, because ideation alone
takes twelve to fourteen minutes on the shipped config.** The second proof
run's timeline (file mtimes and the status file on the fresh machine):
the campaign started 21:49:04, setup and repo memory were done by
21:49:08, the implementation session's folder appeared at 22:01:26, and
that session was killed at its 60-second floor at 22:04; the Python-flow
campaign launched at 22:07 was still in `ideation` at 22:21. So a
fifteen-minute campaign spends it all on the ideation session
(`claude-opus-5` with four gates) and the implementation gets the floor.
The first-build example and page now say three quarters of an hour and
the brief says `minutes=45 experiments=3`; the "ten minutes" the docs plan
promised is not true of the shipped config and is not claimed. A faster
profile for small briefs (a lighter ideation for a one-file task, fewer
lanes, or a quick mode) is a product question for WP4, the config
package, and is listed in the sub-plan's findings for later packages.

**The launch record is written on every fresh campaign, not only with the
inbox on, and it carries the brief.** A resume reads the launch's budget
and flags from it; with the inbox off there was no record, so a bare
resume dropped the brief's minutes and dollars (the pull-request review
found it). Registering the campaign in the inbox's registry stays
conditional on the inbox: that file is the inbox's. Alternative rejected:
keeping budgets only in the checkpoint, which does not hold them.

**`deploy()` refuses a solution with no candidate and a campaign directory
not checked out at the delivered commit.** Deployment copies the directory
as it is; after `git checkout main` for a look around, it would have
shipped the seed under the winning score. The refusal names the checkout
command. Alternative rejected: checking the commit out silently, which
moves a directory the person is working in.

**A bullet or a number at the start of an objective's line is text.** The
parser used to refuse a `- ` line under a text field so a misplaced rule
would not vanish into the objective; but goals carry bullet lists, and a
brief built in Python with one could not be read back. Text fields take
any text; `Rules` and `Experience` alone take items. Alternative rejected:
escaping bullets in the file, which no one would type.

**`solution.evaluator` is `None` with the shipped config, and the pages say
so.** The evaluation registry is written by the evaluation maintainer,
which runs only when a config has an `evaluation_maintainer` block;
`GENERIC` and `MINIMAL` have none. The pages describe the shipped
behaviour (a provided evaluator is protected by the integrity check; a
proposed one is judged, not registered) and name the block. Turning the
maintainer on by default is WP4's decision, with the config.

**No abbreviated flags on `kapso build`, `inbox` and `deploy`.** argparse
accepts `--det` for `--detach`; the detached child strips only the full
flag and would detach again, forever. `allow_abbrev=False` on the three
parsers whose flags a child re-reads. Alternative rejected: stripping
every prefix of `--detach`, which hides the trap instead of closing it.

**Fourteen public names changed after the naming review, two user-chosen
words did not.** Both reviewers (Codex gpt-6-astra at xhigh, Claude Code
opus-5) judged every public name of the build surface; the table in
`wp1-build.md` ("Naming review") lists each change and its reason. The
pattern: where the code's word and the word every person-facing string
used disagreed, the person-facing word won (`experiments` over
`iterations`); where one word was spent three ways, each thing got its own
(`evaluation` the directory, `evaluator` the stamp); where one directory had
five names, it got one (`campaign`). `Harness` and `Experience` stayed:
they are the owner's names for the two layers (2026-10-21, 2026-10-24),
and the reviewers' alternatives (`Domain`, `Lessons`) were judged on
cold-reader clarity alone. These renames touch the design document's
section 8 (`solution.trace` → `solution.campaign`,
`solution.evaluation.version` → `solution.evaluator.version`), which is why
they are here for the owner to check.

**The facade resolves every path it is given, and the workspace keeps its
directory absolute.** The first proof run on the fresh machine crashed at
the first session close: `--output ../churn-build` reached the experiment
workspace relative, the session clone ran `git add ../churn-build/...`
from its own directory, and git refused it as outside the repository. The
code is unchanged from `main`, so every relative `--output` whose session
left run data behind crashed before WP1 too; the fresh machine found it
because nothing from the development checkout masked it. `build()` now
resolves `output`, `evaluation`, `data` and a local `repo` (a URL is left
alone), and `ExperimentWorkspace` takes `os.path.abspath` of its directory,
so a direct caller is covered as well. Rule 7 applies: no second code path
for relative directories.

**A five-minute budget starves the implementation; the proof uses
fifteen.** The budget's `clamp_timeout` bounds every session by the time
left outside the reserve, with a 60-second floor, so a five-minute
campaign gives its first implementation about a minute, which the deadline
then kills. The example brief says ten minutes and four experiments; the
proof run passes `--budget-minutes 15 --iterations 2` on the command so
each session has room, which is also what the flags-win-over-the-brief
rule is for. The ten-minute claim on the first-build page stands for the
whole example on a machine with the libraries installed; it is measured by
the proof run and will be corrected if the run says otherwise.

**The examples are the semantic fixtures, under `examples/`.** The docs
plan said one source; the WP0 fixture lived under `tests/semantic/`. It
moved to `examples/churn-first-build/`, and `tests/semantic/README.md`
points at the examples. Alternative rejected: a copy under `tests/`, which
is the drift the plan forbids.

**The engine pages stay at their URLs, regrouped as "How it works".** The
docs plan retires the Evolve pages into condensed How-it-works pages. WP1
regroups them in the navigation and fixes their snippets, and leaves the
condensation to the packages that change their content (the config, the
record and the trace), so each page is rewritten once. Two pages did move
into Build (the inbox, resume and budgets), with redirects.

**The older examples keep their evaluator inside the repository until
WP2.** Moving each `evaluate.py` into a protected `evaluation/` directory
changes how it is invoked and needs each example run again; the harness
re-base in WP2 does that work once. Until then the brief's rule ("the
evaluator is never modified") is what protects it, as it did before.

**`Brief` is both the Python object and the file, one grammar.** The
interfaces document left open whether the brief is an API object or a
CLI-only file. `Brief.read` parses the file into the object and
`Brief.write` emits it, so the plugin, the CLI and a script hand the engine
the same thing, and `build()` renders the whole brief into the engine's
goal text. Alternative rejected: a file-only brief that the CLI flattens
to a goal string, which would leave Python users without the structure.

**Two names are left for later packages, on purpose.** `Experience` as
`learn()`'s return collides with `load_experience`'s source handle; WP3
settles it with the experience work. Whether a brief's `Harness` line or
a `load_harness()` call wins when both are given is WP2's, with the
loader. Neither is decided here.

## 2026-10-24, WP0

**Codex on the fresh machine logs in with an API key, Claude with the shared
OAuth token.** The fresh machine (`scripts/redesign/fresh_machine.sh`)
authenticates Codex with the `openai-api-key` secret from the benchmark
project's Secret Manager, via `codex login --with-api-key` over ssh stdin,
and Claude Code with the `claude-oauth-token-3` secret the campaigns already
use. Why: a copied ChatGPT session is single-holder; the copy that refreshes
on the machine revokes the development box's session, which runs the long
campaigns. A key is metered per token, revocable per machine and leaves no
shared state; a ten-minute proof run costs cents. Alternative rejected:
copying `~/.codex/auth.json` as the campaign waves do. Consequence: proof
runs and semantic runs that use Codex bill the OpenAI API account rather
than the ChatGPT plan; the Claude side still spends the shared weekly
window, so the token gate (probe the CLI per model before a long run)
applies to the fresh machine too.

**The CLI's top-level help keeps its hand-written epilog for now.** The
reviewers suggested replacing it with the parsers' own `help=` strings. WP0
keeps the text byte-identical (apart from trailing whitespace) because its
contract is "no behaviour change"; WP1 rewrites the help when `evolve`
becomes `build`.

**`deploy --interactive` stops on SIGINT through a signal handler instead
of `try/except KeyboardInterrupt`.** Same behaviour (the software is stopped
and the command exits 0); CLAUDE.md rule 2 forbids exception handling, and
WP0 chose to fix the violation in the move rather than carry it into a new
file. WP5 rewrites deploy anyway.

**The semantic fixture's hold-out is not sealed yet, and says so.** The
labels live outside the directories the engine copies into a workspace, but
a process in the workspace can read the absolute path the evaluator uses.
Sealing is the engine's job (evaluation integrity and permissions, build
steps 2 and 6); the fixture measures the flow until then, and its README
states it. Alternative rejected: a fixture-level trick (an encrypted labels
file, a separate evaluator process with its own credentials) that would
have to be undone when the engine seals hold-outs properly.

**The fixture's signal was reshaped for headroom.** The first generator
gave a logistic baseline of 0.742 AUC that a boosting model could not beat
(0.728). A fixture where nothing beats the baseline cannot show a build
working, so the label's logit gained thresholds, an interaction and a tenure
curve: logistic 0.797, boosting with no feature work 0.832. The persona and
the README carry the measured numbers, not round ones.

## WP9 flow-review scoped decisions (2026-10-27)

The post-build cross-model review (the Flow review in
`wp9-semantic-suite.md`) raised two findings answered as bounded
decisions rather than full rebuilds; recorded here so the owner can
revisit them.

**The second engineer's build stays driver-launched.** The review asked
that S2's `second-engineer-build` be driven through the teammate's own
persona and host conversation, as the first engineer is. It remains a
driver launch under the second Linux user, now gated on a scored delivery
served the shared bank (not merely `done` plus a substring). The served
build proves the experience shared across users reaches a second install;
driving the teammate through a second persona+host loop is a larger change
left for a later pass, since the step's purpose is the bank crossing user
boundaries, not a second interview.

**S1's flow count reads a script in its final state.** A session that
edits a script after a failed run leaves only the final text on disk, so
`python_lines` for a task that reused one filename reads that final
version. The count is now per task (so a filename reused across tasks is
not conflated) and reported as the longest single flow, with the limit
stated in the step's notes. Capturing each invocation's exact source would
need the host to write a fresh filename per attempt, which the engineer,
not the driver, controls.

## WP9 S1 flow-length bar: a finding, not a loosened bar (2026-10-27)

The first S1 proof arm (`s1-claude-1`) failed its `python_lines` bar
(16 > 12 for the longest single task) and the judge scored "the flows fit
the design's lengths" at 0. Both read the same thing: the engineer's flow
for the hardest task — load a path-named harness, build under a churn
config, read the solution — was longer than the design's target. The bar
was **not** loosened to pass the run. Two readings are left for the owner:

- **A product finding.** If ≤12 lines is the real target for any single
  task, the harness-load-plus-build path may warrant a more concise idiom,
  or the docs a terser worked example, so a cold engineer reaches it
  without a five-line call plus diagnostics.
- **A metric calibration.** `python_lines` counts physical non-blank,
  non-comment lines, so a readable multi-line `build(...)` call and
  voluntary `print()`s inflate it past a bar tuned for terse flows.
  Counting statements rather than lines, or excluding pure diagnostics,
  would measure the intent — how much the engineer must write to drive a
  task — more directly.

Until the owner chooses, the bar stands and the run reads FAIL honestly.

## WP9 author improvement loop is not wired end to end — RESOLVED (2026-10-27/28)

The first live author arm (`author-claude-1`) surfaced that the round-1 →
guide-fixes → round-2 loop does not actually improve the guide between
rounds, so `structural_gaps_round2` (bar: max 0) is unreachable by
construction, not by the guide's quality. Two gaps, both needing an owner
decision rather than a patch:

- **The gaps are classified only after the whole run.** `step_guide_fixes`
  reads `round-1`'s `classification`, but a guess is classified into
  structural-gap / contradiction / domain-decision / already-in-guide only
  by the Gemini judge, which runs on the development box in the `--grade`
  step *after* the machine run. On the machine, between the rounds, the
  classification is empty, so zero gaps are fed.
- **Nothing writes the gaps into the guide.** `step_guide_fixes` records the
  (empty) gap list and a `git diff` of the guide pages; it never edits them.
  So even with the gaps in hand, the guide is unchanged for round 2.

The arm still showed the author *can* write a passing harness from the guide
alone (round-2 `kapso harness check` green; all four judge dimensions —
pages under forty lines, every skill named, no task value, the ladder
respected — pass; 3/3 steps). What it cannot yet show is the guide getting
*better* between rounds.

Options for the owner:
1. **Multi-stage orchestration** — run round-1 on the machine, fetch, classify
   and write the guide fixes on the development box (the judge's home), push
   the improved guide back, then run round-2. Keeps "judge on the dev box".
2. **In-run classification + edit** — give the machine the judge route and an
   agent that writes the classified gaps into the guide between rounds.
3. **Redefine round-2** as an independent second author (measuring the guide's
   consistency across cold authors, not its improvement), and set the bar to
   "no worse than round-1" rather than max 0.

**Resolved 2026-10-28**, under the standing "no need to wait for approval":
the design's own words describe the loop as development work — "fixed in the
guide (this package owns those pages for that) … the guide's fixes committed"
— not as something one driver run performs, so the arm now matches them. One
cold author per run, on the guide as committed (the record pins the guide's
sha256; the step is `write`; the bars `structural_gaps` max 0 and
`check_green` min 1, `contradictions` reported beside them). The gaps and
contradictions a run classifies are written into the guide and committed,
and the next run's fresh author is the design's round two; both runs' counts
go in the sub-plan's proof table. Deleted: the in-run `guide-fixes` step,
`--round`, the `_round1`/`_round2` metrics. `author-claude-1`'s eight gaps
and one contradiction are in the guide as of this entry. Options 1–3 stay
open if the owner wants the loop inside one run; option 1 is the faithful
one.

## WP9 S5 injections stage early and accept correct handling — RESOLVED (2026-10-27)

The earlier "S5 campaign longevity" open question is resolved, per the owner's
"do the best long-term fix". The live S5 arms showed the injection steps racing
the churn task's node-0 completion: it meets its 0.85 goal in ~10 minutes, so a
rate limit, an evaluator crash, a killed writer or a full disk injected after
that lands on a finished campaign. The cold engineer itself diagnosed this on
evaluator-crash ("the labels moved 16 s after the campaign finished").

A first attempt — stamping an unreachable target (0.99) so the campaign keeps
running — was tried and REVERTED. The intelligent review of that run found two
problems: the engineer detected the artificial goal ("0.99 can't be reached;
the data generator caps the signal"), which distorts its behaviour; and the
campaign then rode out the rate limit and finished on its own, so the engineer
rightly declined to resume it and the `resumed` check failed. The lesson: the
product handled the failure correctly; the test was mis-calibrated.

Fix (kept): each injection fires EARLY, while the campaign is still working and
has budget — killed-writer and full-disk at the implementation phase, the
evaluator crash at implementation (labels moved before the first evaluation),
and the rate-limit shim ARMED AT LAUNCH so it trips the campaign's first
`claude -p` session reliably (arming only after implementation raced a
single-node campaign and never fired). And provider-limit's success now accepts
the engineer's correct handling either way — the campaign rode out the limit
and finished (status done) OR it was blocked and the engineer resumed it —
rather than demanding a resume that is wrong when the campaign was not stuck.
The achievable 0.85 goal is kept, so the resume paths (killed writer, full
disk, a blocked evaluator) still work as designed.

## WP9 S5 injection-harness calibration is a focused follow-up — RESOLVED (2026-10-28, evening)

Resolved by the pass recorded in the sub-plan's S5 section: the seeded
bank, the crash read from the attempt's sessions, the generator removed
from S5's project, the sampler reading `kapso watch`'s verdict marker and
keeping its cadence, a writer that recorded its own end counted as gone,
the resumed campaign watched for the step's budget. The clean arm of all
eight steps (`s5-claude-7`) met every bar with seven of eight steps true;
the eighth (evaluator-crash) passed on the arm before and read false on
the clean arm for the engineer's slip, not the product's. The staging
stays early on the real goal, as the entry below records.

## WP9 S5 injection-harness calibration is a focused follow-up (2026-10-28)

The S5 arms validated the product: kapso and the engineer handle every injected
operational failure correctly (see the proof run — diagnose, recover, refuse to
tamper, pause-and-ask, resume). The S5 TEST harness, however, stages its
injections non-deterministically — they depend on variable campaign/agent
choices (does the campaign use the enrichment, does a transient 429 block, does
a node finish before the injection lands) — and several success conditions were
written for naïve behaviour (e.g. demanding a resume when the correct response
was to ride the limit out, or to pause and ask). provider-limit and
evaluator-crash were calibrated this round; killed-writer, full-disk, tampering,
and the staging of missing-key/provider-limit still need a deterministic-staging
pass: the injection task should force the campaign to hit each failure point,
and each success condition should accept the engineer's correct behaviour.
Deferred to a focused pass rather than burned down with multi-hour runs, since
the product is already validated. S4 (which reads S5's tampering campaign)
defers with it.

## WP9 author classifier: the structural gap is the design's rejection test, made explicit (2026-10-28)

Three author runs on successive guides — `author-claude-1` (8 gaps + 1
contradiction), `author-claude-2` on the guide after those fixes (2 + 0),
`author-claude-3` after the next (3 + 2) — showed the judge's classifier
labelling as structural things no rule rejects: which MCP SDK pattern a
stub server uses, the Python version a script may assume, whether a README
may ship. The design defines a structural gap as "a field, file, key,
convention or host behaviour the engine or `check` would reject if guessed
wrong"; the `CLASSIFY` prompt now says that test out loud (name the rule a
wrong guess would trip; an open choice any reasonable value satisfies is a
domain decision, which now covers the author's own code as well as their
domain). Re-grading run 3's own guesses under it: 1 gap + 1 contradiction,
the gap a real one the first pass had filed as already-in-guide (a skill
named only on `guardrails.md` does not satisfy the module-page mention);
both are in the guide. The bar stays max 0; the re-graded report is the
committed one. Not a loosening: the same words, applied as written.

## WP9: the learning crews' Codex roles run in the adapter's default sandbox (2026-10-28)

S6's third arm (`s6-claude-3`, the shipped Fable-5 learn on a fresh Ubuntu
24.04 machine with codex-cli 0.165.0) mined its trajectory and then failed
the exam: both report_writer sessions ended without `report.md`, and the
frame's line said only "ended without writing it". Reproduced on another
fresh machine: `codex exec --sandbox workspace-write` cannot write into its
own workspace there ("the workspace rejected writes due to a sandbox
permission error"), plain, git-initialised or with `--add-dir`; only the
adapter's default `danger-full-access` — the sandbox every engine session
already runs under — writes. The learning subsystem was the one place that
asked for workspace-write: the graders' roles, the update crew's worker
launcher, codify's implementor and judge, the behavior reviewer. All four
now take the adapter's default; the execpolicy rules file the adapter
stages (`.codex/rules/kapso-campaign.rules`) stays the campaign-tool
guard, as for every other session. And the frame's fail-loud line now
carries the tail of what a session said, so the next such failure names
itself. Two further findings from the same arm, for the owner: Fable 5's
safeguards flagged the mining prompt ("`[reasoning_extraction]`", the
first mining session dead in 3 s, the second interrupted mid-tool-call and
recovered by the frame's repair round) — the learn ran on Opus 5.5 after
that (`--learn-model`, authorized); and the mined `mining-report.md` the
first session left was the API error's text, which the mined-view
validation accepted as a report.

## WP9 S2 learn-complete: a learn over the churn campaign outruns its 150-minute step (2026-10-28)

Both clean S2 runs read `learn-complete` false at the step's 150-minute
cutoff with the learn still in its lesson phase: on the Claude host mining
took 36 minutes, the exam 68 and the lesson ran past 80 (the churn
campaign's first attempt had taken 48 minutes with the research tools);
on the Codex host mining 25, the exam 48, the lesson past 77. S6's learn
over the same fixture finished in 135.8 minutes (mining 37, the lesson 88),
inside S6's own `learn-build-one` budget of 240 minutes — the same operation
carries 150 in S2 and 240 in S6. The exam's length varies most (20 to 68
minutes): its report writer is a Codex session on `gpt-5.6-sol`, slow on
a loaded afternoon. The step's budget is a design value and stays as
written; the record shows the overrun and its parts. The owner's call:
align S2's `learn-complete` with S6's 240 minutes (the learn validation's
finding — a learn's wall-clock scales with the campaign's richness — was
already on the table), or accept that the second engineer's served build
reads false whenever the learn runs long, since it has no card to serve
until the learn banks.

## WP9 S2 line budget: the Claude host's approval message runs fifteen lines against twelve (2026-10-28)

Every build conversation on the Claude host in the S2 round (`build-churn`
14–15, `build-pause` 15–19, `build-fraud` 15, `build-data-only` 15 lines
outside fenced blocks) ran over the design's 12-line budget while the
Codex host said the same in 3–8. The message is the one the evaluation
skill prescribes inside `kapso-build`'s single approval: what is here,
the protocol in five lines, the brief in a fenced block, the baseline and
the ask — about fifteen lines of prose when written as the skill says.
The judge rated those messages 4–5 on "not annoying" and "return on
attention". The budget stays at twelve and the rows read FAIL, as the
design wrote them; the owner's call is whether the skill's message is
trimmed (a shorter "what is here", the protocol's five lines folded into
the brief) or the build conversations' budget is fifteen. Not taken here
because the budget is a design value, not a harness calibration.

## WP9 S2 second engineer's served build: a budget in which an attempt can score (2026-10-29)

The served build's budget was ten minutes and one experiment, and on
the churn fixture a first attempt scores at about sixteen minutes (S6's
build one: 15.8 minutes to its first scored observation at a sixty-minute
budget). At ten minutes the acceptance line "the second engineer's served
build scored" could not be met by construction, whatever the serving did.
The suite's `builds.second_engineer` is now thirty minutes and two
experiments, the step's budget forty-five minutes (the build and the
driver's grace), and the machine cap derived from the suite grows by the
difference. A calibration of the harness, not a design value: the design
says "scored", not how long the second engineer may take; recorded here
so the number is read as the suite's choice. The first arm on the fixed
second user (`s2-claude-second-1`) launched the served build with the
bank's two cards but no status file ever appeared within its wait and
the driver of that run did not archive the campaign's log; the driver
now archives the log and the status, and the rerun says what happened.

## WP9: the fresh machines' Codex roles run on the owner's device-auth session (2026-10-29)

The plan said every Codex role on a fresh machine would wait for credit
on the machine's API key and that the Codex arms would run on the
development box, marked advisory, until then. What was built instead,
from `s1-claude-1` on (411b230e): `fresh_machine.sh` pushes the owner's
device-auth ChatGPT session (`~/.codex-semantic/auth.json`, over ssh
stdin only) to each machine at install, and the S2 driver copies it to
the second engineer's Unix user, so every arm ran on a fresh machine and
no record carries the advisory mark. The cost: a personal login copied
to throwaway VMs and a second user's home, revocable by any copy's
refresh — re-logged before a wave, verified by the preflight's one-token
call, the machines deleted after the fetch. The runner now writes
`~/.codex/SOURCE` beside the pushed session and every driver copies it
into its record's notes, so a run on the borrowed session says so; the
runs before that mark were on the same session, said in the sub-plan.

## WP9 S2 judge: a step is scored on the dimensions it can give evidence for (2026-10-29)

The design's five dimensions at 4 or above, with the worst item binding,
made the bar unreachable on steps the design made terse: `watch` is
budgeted two messages of eight lines ("how the campaign is going, in a
few lines"), and a report with nothing to be proactive about scores 1 on
`proactive` by the rubric's own rule ("no evidence scores 1"), binding
the run. The suite now lists per conversation step the dimensions it is
judged on (`judged`: watch, session-brief, harness-export,
experience-share, learn and campaign-tools are scored on clear, not
annoying, engaging and return on attention; every other step on all
five), the worst eligible item binds, and the judge adjudicates no
ineligible pair. A harness reading of the design, recorded for the owner
to confirm or widen: the design's words are "every dimension at 4 or
above", and the choice here is which steps a dimension is a fair question
of. The Codex host's terseness stays a finding where the step invites a
finding.

## WP9 S5: the two-message bar is the driver's construction; truthfulness is read sample by sample (2026-10-29)

The run-level bar `user_messages_max ≤ 2` could not fail: the driver
sends at most two messages by construction and a class not handled
within them reads false on its own predicate. The bar is gone (the count
stays as information). Truthfulness is now held sample by sample — the
shown verdict against the liveness the sampler observed beside it, on the
samples between one and two heartbeats after the injection — and an
injection with no sample in that window is counted unobserved, where the
bar wants none; the two partial arms had one each, hidden by the ratio.
Retention after a resume compares the attempts' identities and reads
"not exercised" when none had landed before the kill (the early staging
makes that the usual case): the claim the records support is the
campaign's continuation, said as such.

## WP8: the IOAI book onboards as a bank in the card grammar, not an index (2026-10-29)

The port plan of 2026-10-24 said "an ingested index"; WP3 then settled the
forms a source takes: a bank (served whole by the bank gate, from git
alone), the hosted gate, or an index (a corpus ingested through sessions
into Weaviate and Neo4j, searched by embedding). The book is sixty-seven
entries, each a problem, a winning idea with its provenance and a public
score, and runnable code: evidence-priced cards in all but format. A bank
fits it without infrastructure on the contest machine, serves an entry
whole (the code with the card, as a procedure card does), and keeps the
provenance line per card; an index would spend hours of ingestor sessions
turning notebooks into pages and need the two stores wherever the harness
runs. The conversion script writes the grammar's every field; the
harness's playbook skill says how to pull the cards for a statement's
twist and names no problem. If a later package wants the book searched by
embedding, the same bank ingests then.

## WP8: the abstraction scan is a gate test with the benchmarks' vocabularies, not a `check` rule (2026-10-29; superseded on the scan's home by the entry "the abstraction scan is benchmark tooling, not `check`" below: it lives at `benchmarks/harness_scan.py`, not under `src/kapso/harness/`)

The plan asked for the scan "run by `check` for shipped harnesses".
`check` is domain-agnostic and reads one harness; it cannot know which
words are a benchmark's task names, and a user's harness is not our
benchmark's. The scan is a module (`src/kapso/harness/abstraction.py`)
with its vocabulary passed in, and a gate test runs it over the bundled
trees with the words read from the benchmarks' own registries, so a task
added to a benchmark is scanned without anyone editing a list. The two
fixed rules (a decimal that reads as a score, a quota or a deadline; a
path outside the harness's conventions) are the module's own.

## WP8: one reproduction per benchmark at its recorded budget, after a smoke arm; MLE and ALE establish their baselines (2026-10-29)

The done-when lines ask for "the recorded score within noise". The tree
records RelBench's scores per task and the IOAI contest's; it records no
MLE or ALE result. The RelBench arm is judged against its task's band; the
IOAI arm against the practice task's recorded cold start; the MLE and ALE
arms record the baseline this package establishes, and their bar is a
valid score at the budget, said as such in the record. A two-hour smoke
arm precedes each full arm: a harness fault found at a tenth of the cost.

## WP8: the predictive-modelling task swap runs through the IOAI runner's preflight (2026-10-29)

The swap wants a Kaggle tabular competition, a task the benchmark does not
contain. The IOAI runner's preflight already turns a Kaggle URL into a task
root; with `--harness predictive-modelling` it is the swap, with no runner
written for it. That the runner's `--harness` takes any reference is what
makes the swaps cheap, and is why the flag exists on every runner.

## WP8: the hooks shared with the trading example are copied (2026-10-29)

`protect-prepared` and `deny-web-fetch` are the same programs in the
trading example and in `predictive-modelling`. A harness is a plain
directory and composition is not in the design, so the two scripts are
copied, with a test that runs each bundled hook on both hosts' event
shapes. A shared hook library would be a design change for a later
package, once a third harness wants them.

## WP8: nothing of the IOAI book is pushed anywhere by this package (2026-10-29)

The provenance table gives a verdict per entry; the conversion writes a
shareable bank (the entries whose source licence permits redistribution,
and our own reimplementations) and a local-only bank (the rest). Both
live on this machine and in the backup bucket. Whether the shareable bank
gets a remote, and which, is the owner's call, with the table in hand.

## WP8: the reproductions are read against their recorded populations, preregistered, at the shapes the records give (2026-10-29, after the plan review)

The done-when lines say "within noise of the recorded result at the same
budget and model". Retrieved: RelBench `rel-trial/study-outcome` was run
as thirteen campaign lanes (4×A100 boxes, twelve hours, a champion seeded
across lanes; test AUROC mean 0.775, sd 0.021, best 0.821); the MLE
submission was three runs of Gemini-3-Pro-Preview at twenty-four hours on
one H100 (spooky log loss 0.247, 0.230, 0.284); ALE's `ahc008` is one
number with no launch record; the IOAI practice competition is no longer
reachable. No arm can match its record exactly: the RelBench lanes were
warm and the OpenAI keys are exhausted, the MLE model is not served, the
IOAI task is gone. So each bar is preregistered here against its
population with the differences named (the Acceptance section), the
RelBench arm runs at the recorded four-lane shape on a four-GPU spot box,
cold, and the IOAI arm is a baseline on an open practice competition at
the recorded cold-start shape. The owner may replace any bar; the
implementer does not lower one silently.

## WP8: the IOAI book's cards need a reported evidence kind; the request goes to the learning package (2026-10-29)

The card grammar admits only evidence a campaign produced (a learner run,
a trajectory, a reference, a usage, an effect) with executed verdicts. A
curated winner's solution has a source, an author, a licence and a public
score. Rather than fabricate measurement-shaped entries, the conversion
waits for a `reported` source kind and verdict, outside the executed
verdicts and the promotion rules, so a converted card is born `candidate`
and serves as "reported by others". That is a change to `src/kapso/learning/bank.py`,
owned by neither WP3 nor WP8, requested here; the fallback converts only
the entries we reproduced ourselves with a trajectory.

## WP8: the sanitized cache is the runner's to build, with the harness's script (2026-10-29)

The port mapping put the sanitized copy in the setup session as a
prepared resource. Prepared resources are not integrity-verified, the
setup session's budget is minutes, and the copy is reused across
campaigns. So the runner builds it before launch, with the one standalone
script that lives in the harness skill (the domain machinery any
relational task can use), reuses it under a version marker, and names it
in the Brief as an absolute read-only path; the setup session verifies it
and records the regime.

## WP8: the Kaggle session pool is the runner's CLI, not a harness server (2026-10-29)

The ticket office blocks until a slot is granted (its contract), its pool
sizes are the account's limits (configuration), and its ledger is shared
by every lane of a campaign (a path the runner knows). A harness server
runs in a session directory under a tool timeout with no campaign-root
contract. The harness's skill names the discipline; the task root the
runner writes provides the office and the submission procedure.

## WP8: the metric's direction is the evaluation directory's to orient (2026-10-29)

The engine optimises one `score`, higher is better. A lower-is-better
benchmark's grader emits the oriented score and reports the raw metric
and its direction in the evaluation output; the Brief says both; the
runner's final report prints the raw metric. No engine field gains a
direction: the evaluation directory already owns what the score means.

## WP8: three handler behaviours become engine seams; the handler attributes go (2026-10-29)

`honor_agent_stop` becomes a config key read by the orchestrator;
`finalize_run_selection` becomes the generic feedback flow calling the
engine's own archive labelling; `deliverable_ready_reserve_seconds` is
dropped with the IOAI handler (WP4's derived reserve applies). Each is a
small edit in WP1's or WP4's file, requested in the plan and made once
the owner says go; after the ports `GenericProblemHandler` is the only
handler, so the attributes have no second reader.

## WP8: the swaps' bar is zero harness edits; the runner edits are pre-listed (2026-10-29)

A swap tests the harness, not the benchmark adapters. The runner edits
each swap is known to need are listed in the plan before it runs, so a
runner change is not mistaken for a harness finding; a harness edit is a
finding fixed at the source, and the swap runs again.

## WP8: the contest root's conventions are the harness's, and its hooks are mistake catchers (2026-10-29)

A leaderboard contest's task root carries a descriptor,
`kapso_datasets/contest.json` (`competition`, the absolute `ledger` path,
and the `procedure` that submits and records when one exists); a
scorer-and-seeds task's root carries `kapso_datasets/seeds.txt`. The
runners write them (changes 13 and 14); a user's preflight writes them for
their own contest. The shell denial of a raw submission fires only where a
procedure is named; without one the CLI submits and the session records
the entry itself. A hook catches a mistake (shape, freshness, presence) and
the verifier is the gate; both Stop hooks honour `stop_hook_active`. A
corrupt evidence file fails loud in Python, and hook.json's launcher line
(`|| exit 2`) turns the failure into the hosts' blocking exit, so a corrupt
file never passes as a finish. The zero-score convention (one failed seed
voids the submission) is ALE's and goes to its task statement; the harness
reads validity from the scorer's status and the task's rule.

## WP8: the contest harness has no environment file; six skills are shared (2026-10-29)

The contest harness's hooks are standard-library scripts, so its
`environment.python` is dropped (a C++ campaign on a stock box is not
refused for numpy); the task's own tools are checked in setup. The shared
skills are six (`task-type-playbooks`, `cv-protocol`, `ensembling`,
`eda-discipline`, `hf-model-fetch`, `tabular-modeling`), copied by
directory, kept identical by the gate test, and self-contained: none cites
a skill the other harness lacks.

## WP8: the RelBench runner's shape on build() (2026-10-29)

The handler class is gone; `benchmarks/relbench/scoring.py` holds what the
runner still does itself (the copy through the harness's script, the staged
evaluation, the final report), `context.py` the Brief from the task's
facts, and the facade the rest. What the plan listed as optional went: the
champion seeding (its `claims/` directory was retired earlier, so it seeded
nothing), `--initial-repo` (the seeded baseline is the repository), the
early-stop target and the knowledge file (the stop vote is advisory). The
pristine cache is the library's own cache directory, read through the
library's helper, never a config key of ours; the hosted model API grant is
the override file's `benchmark.hosted_model_api` (`none` or `openai`), a
user's decision the Brief states as a rule. The candidate-facing starter kit
reads the task's identity from `kapso_evaluation/task.json` and the copy
from the variable the evaluation sets for the run, so the runner exports no
environment of its own.

## WP8: the MLE and ALE graders are registered, their baselines take the floor, and a large dataset stays by path (2026-10-29)

The plan's "the grader directory per competition" is a provided evaluation
the maintainer registers (the two override files enable it, as RelBench's
does): a provided suite without a maintainer is only a script the sessions
run themselves, and the score of record would be self-reported. The
maintainer's calibration needs a runnable candidate, so each runner seeds
a baseline that completes a fast run and reports the floor: MLE's writes
the sample submission unchanged and prints the override file's floor
score; ALE's prints the problem's example output and fails the cases, which
is the floor under the contest's all-or-nothing rule. The floors are the
override files' `benchmark.floor_scores`, the worst value of either
direction. MLE's competition data is not passed as `data=`: the engine
tracks `kapso_datasets/` in the workspace's git and copies it per session,
which a dataset of gigabytes cannot bear, so the data stays under the
archive root by absolute path, named in the Brief (ALE's task root, the
seed list, is small and is `data=`). The MLE description rides verbatim
(the LLM cleaning step is gone) and the hub search is the harness's
`hf-model-fetch` skill in setup.

## WP8: the IOAI evaluation attributes submissions by content and runs without a maintainer (2026-10-29)

A leaderboard task has no calibration run, so its provided evaluation is
not registered by the maintainer: the sessions run it as the provided
suite and report its score, which is the harness's scorer procedure for a
contest anyway. The evaluation attributes the ledger to a candidate by the
hash of its kernel source or of its submitted file, never by the session's
directory: the engine also runs a candidate's evaluation in a materialized
worktree of its branch, where the directory says nothing. The submission
procedure mints a token into every message, which is how the competition's
submissions list is attributed back to the ledger. The organizers' rules
ride in the Brief's objective as a section, not as a Rules item: the brief
grammar reads a bulleted line under Rules as a new item, and the rendered
text must parse back to the same brief for a resume.

## WP8: the IOAI book's cards are `reported` evidence, born candidates, priced by nobody (2026-10-29)

The card grammar's evidence describes what a campaign did with a claim; a
curated winner's solution has only a score at a source. The grammar gains
one kind, `reported`: a source of `{url, author, licence, score, protocol}`
with the verdict `reported`, outside the executed verdicts (it promotes
nothing, seeds no codify, counts in no probe tier) and outside the
reliability arithmetic (a converted card writes no validity, boundary,
coverage or score — the index shows `score None` and sorts it last; the
first executed verdict starts the numbers). A learner run cannot write it:
admission refuses the verdict by name. The serving layer tells the reader
on every read of a reported-only card that the score is the source's, not
a kapso measurement, and renders the source whole in the evidence trail;
the loaded bank's own intro line stays "measured by others" (the bank is
somebody's; what each card's evidence is, the card says). The converter
takes the idea page's `status:` and `best:` lines as the record (INDEX.md
writes `?` where the page carries the review's verdict), the router line
as the title, the lead paragraph as the rule — a thin rule is the book's
thin lead, left for the seeder's rewrite queue rather than padded. The
licence pass's verdict places an entry (shareable: shippable-with-attribution
and ours; local-only: the rest); the use verdict is "yes" for our code and
permissive sources and "owner's call" with the open question named for
everything else — the mechanical pass decides no legal question. The
survey and the theory entries are text procedures (no payload); compiled
byproducts never ship; markdown in a payload is refused because the card
count would read it as a card.

## WP8: the abstraction scan is benchmark tooling, not `check` (2026-10-29)

`check` validates a harness's grammar for every author; the abstraction
scan validates ours against the benchmarks we ported from, so it lives
under `benchmarks/` with the benchmarks' own vocabularies and runs as a
test over the bundled trees, never as a `check` rule a user's harness
would trip on its own domain words. The decisions per word: a platform's
name (kaggle, atcoder) is domain vocabulary and no word of the scan; a
data library's name (`relbench`, its cache variable) is a skill's
vocabulary — the skill teaches its API — and a page's constant, so the
two page sentences that named it now name the skill; a benchmark's name
(`mle-bench`, `ale-bench`, `ahc`, `ioai`, `kumo`) is a constant anywhere;
a task, dataset, competition or problem id is a constant anywhere. A
number counts only beside a quota, deadline, score or hardware word
(within two tokens), with version strings and statistical thresholds
allowed and an ordered list's marker skipped. The harness's own names are
blanked before a word is read, because a page must name its skills
literally (`relbench-api` is a skill, not the library). A zero-hit scan is
evidence, not proof: the cross-model reviews of each harness read for the
rest.

## WP8: the IOAI arm runs on the recorded practice competition, not the radar mirror (2026-10-29)

The plan preregistered the IOAI arm on the open `radar-ioai-2025` mirror
because the recorded practice competition (`timed-deps-12`) was taken to
be unreachable. The machine said otherwise: the account that ran the
recorded campaigns has not entered the radar mirror (the rules are
accepted in a browser, never by the CLI; the download answers 403 and the
preflight stops, as it should), while the practice competition still
answers that account — its submissions list reads, its August
submissions were scored — and is only absent from the search listing. So
the arm runs on `timed-deps-12`, the task the recorded cold-start shape
was measured on, with the shareable bank minus that task's own entry (46
cards); its public leaderboard supplies the median the bar names. The
radar variant stays prepared (the bank without both radar entries, the
mirror's task and the at-home near-duplicate) for the day the owner
accepts the mirror's rules; the record says which competition each run
used.

## WP8: the long arms run on standard machines, and an arm runs its own copy of the machine script (2026-10-29)

The plan put every arm on a spot machine with the engine's checkpoint as
the answer to a preemption. The first IOAI smoke machine was preempted
two minutes after its runner launched, which would have cost a full arm
its whole evidence trail and the Kaggle submissions already spent. The
machine script gains `KAPSO_BENCH_PROVISIONING` (spot by default, the
price the record carries; standard for the arms whose hours a preemption
would waste), the deletion line bills at the rate of the model used, and
the twelve- and twenty-four-hour arms run standard; the smokes and the
swaps stay spot. Separately, bash reads a script as it runs: an edit to
`benchmark_machine.sh` while three arms were inside it broke each at its
next function boundary, so every arm script copies the machine script
beside the original at launch, runs the copy and removes it at exit; the
copy's name (`benchmark_machine.<run-id>.sh`) is ignored by git. The
install script's lessons from the smokes are in the script itself: one
uv-managed Python 3.12 on every kind (the MLE library needs 3.11 or
later, the GPU image ships 3.10), the bundled harnesses' own requirements
(the engine refuses a build whose harness needs are unmet), the RelBench
library pinned at the version the runner and the sanitizer were written
against (relbench 3 restructured its modules), the engine's `mcp<2`
re-pinned after the ALE library's extras.

## WP8: the arms are quick runs in parallel on distinct logins, not the recorded budgets (2026-10-29, the owner)

The plan preregistered the four arms at the recorded budgets (RelBench
twelve hours on four A100s, MLE twenty-four hours) and the owner ruled
the wait unacceptable: the arms run in parallel on separate machines,
two to four hours each, on simpler shapes where one exists, each on its
own Claude login (the owner's managed accounts beside the setup-token
secrets, so parallel arms never share one login's five-hour window). What
changes in the record: the RelBench arm keeps `rel-trial/study-outcome`
(its recorded population, its Claude Code baseline at 75.16 and KumoRFM's
72.0 are the only offline numbers) at three hours on one A100 with the
four lanes, so its number is read against the twelve-hour population as
a short-budget point, with the published entries as the bar it can
honestly be held to; the MLE arm is three hours of the recorded
twenty-four, read against the medal thresholds; the ALE arm three hours;
the IOAI arm the two hours the plan named. Every record states its
budget beside the recorded one, and the acceptance reads "at the arm's
budget" where it read "at the same budget". The machine script takes a
login from a credentials directory as well as from a secret; an arm's
record names which.

## The release gate runs S1 to S5, not S6 again; the rest in parallel, done by Saturday (2026-10-29, the owner)

S6 (experience pays) was run in WP9 and the owner does not want it run
again for the release: the gate before the release is S1, S2, S4 and S5
on both hosts on fresh machines from the release wheel, with S3's record
from WP8; the S6 reports of WP9 stand as its evidence. The remaining work
(WP8's arms and swaps, WP10, WP11's gate) runs as parallel as the logins
and the machines allow, with Saturday 2026-10-31 as the target for the
release.

## A host's stream splits on the newline alone, and the line a kill cut is the one without its newline (2026-10-29)

The RelBench quick arm (`s3-relbench-1`) died in its first ideation: a
Codex web-search result's title held a U+2028 line separator, Python's
`str.splitlines` (which the session record used to split the stream it
digests) cut that event line in two, and the record raised on the half
(`JSONDecodeError: Unterminated string`), which killed the campaign. A
host's event is one newline-terminated line whose text may hold any of
the separators `str.splitlines` takes for line ends, so `stream_lines`
(transcripts.py) splits on the newline alone and every stream reader
goes through it: the session record, the codex adapter's MCP-call
reader, the ideation's killed-turn messages, the trace fixture builder.
The cut check follows: a kill leaves the last line without its newline,
and that, not a missing closing brace (a cut inside a string of code may
well end on a brace), is the sign of a cut line. Found by the arm, fixed
in WP8 (the arms run from the rebuilt wheel), noted here because it is
engine code outside the package's scope.

## The engine's run folder is committed by force (2026-10-29)

The ALE quick arm's implementation session wrote the campaign's run folder
(`output_data_<id>/`) into the repository's `.gitignore` beside its own
build outputs, and the engine's commit of that folder at the session's
close failed on the rule, which killed the campaign. The run folder is
the engine's record (the scores, the artefacts the evaluation reads), not
the candidate's output, so `ExperimentSession.commit_folder` adds it with
`git add -f`: a candidate's ignore rules, the task's or the session's own,
do not reach it. Found by the arm, fixed in WP8; engine code outside the
package's scope, so noted here.

## A one-line skill fix after the freeze bumps the harness version; the record names the version each arm ran on (2026-10-29)

The pull-request review found the `kaggle-cli` skill's pages line wrong for
the CLI the machines install (the swap's preflight log shows the bare form
rejected and the session recovering with `-c`). The line is fixed and
`competitive-programming` is 0.3.1; the arms ran on 0.3.0, which differs
by that line only, and the S3 record says so beside each arm. The frozen
row in the sub-plan is re-pinned to 0.3.1: the bar "no harness edit after
the freeze" guards silent drift, and a named, one-line change with its
version bump is the opposite of silent.

## The MLE in-loop score is the candidate's own hold-out number; the owner's call whether the grader owns the split (2026-10-29)

The review's finding 5: the MLE grader validates the submission's format
and takes the score the candidate prints on its own stratified hold-out,
so the search optimises a self-report that only the verifier and the
final private grade contradict. The plan's earlier decision (the held-out
score stays the search signal) stands for this package; the Acceptance
now says the number is self-reported and what checks it. Making the
grader own the split (deterministic hold-out ids from `task.json`, the
predictions scored by the grader) is the owner's call for a later
package.

## The IOAI arm stands as a replay, the IOAI swap is dropped, the merge waits for the arms, the MLE grader keeps the self-reported number until after the release (2026-10-29, the owner and the builder)

Asked what needed a call, the owner answered three things. No clean
Kaggle account will be set up: the IOAI arm's record stands as it is
(the preflight, the contest root, the ledger, the hooks and the runner
established; the score a replay of the account's own earlier kernel; no
cold-start claim), and the IOAI swap, which needs a practice
competition's rules accepted in a browser, is dropped from the suite's
required steps for the release (`tests/semantic/suite.json`, the `s3`
entry's `machine` list) and kept as a step definition for a later run.
The merge of the harness ports waits for the running arms to land in the
record. The MLE grader was left to the builder: the release keeps the
in-loop number as the candidate's own hold-out score, since the number
the release reports is the official final grade on the private answers
either way, and the grader-owned split (deterministic hold-out ids from
the task file, the held-out predictions scored by the grader) is the
first item after the release; changing it now would cost a day and a
rerun of the MLE arm without changing what is reported.

## The six-hour RelBench attempt is the last run on any benchmark (2026-10-29, the owner)

After the three-hour quick arm on one GPU with four lanes finished one
iteration under the finishing cut and delivered no candidate, a six-hour
attempt on the same shape (`s3-relbench-4`) was launched. The owner: no
further attempt after this one. Whatever it shows goes into the record
as the RelBench arm's reading for the release, with the three-hour
attempt kept in the notes. The same holds for the other benchmarks:
the MLE and ALE quick arms, the IOAI arm and the three swaps stand as
recorded, and no further attempt runs on any of them before the
release.

## A benchmark run files no evaluator change request: the official evaluation is the protocol (2026-10-30)

The six-hour RelBench attempt (`s3-relbench-4`) reached a test AUROC of
0.7555, inside the recorded band, and delivered nothing: at iteration one
the evaluation maintainer accepted a change request from the campaign
("a genuine resolution defect") and registered a second evaluator that
averages the official metric across closed origins, and every lane's
candidate, written to the official contract, exited non-zero under the
new one, so no attempt was admitted and the campaign stopped with no
successful experiment. The provided bytes stayed immutable, as the
maintainer promises; the protocol around them did not. On a benchmark
the official evaluation is the protocol by definition, so the four
runners' override configs set `evaluation_maintainer.max_change_requests`
to zero and the config test holds them there. Found by the arm after the
owner's call that it was the last run, so the fix is recorded, not rerun.

## The review's fixes after the arms: the pins stand, the trees move (2026-10-30)

The pull-request review's Codex half changed both bundled harnesses'
prose after the arms ran (the copied tabular skill's references, three
rules conditioned, the fetch skill's web claim), so predictive-modelling
is 0.3.3 and competitive-programming 0.3.2. The S3 record's zero-edit bar
reads each launch's harness hash against the accepted pins of its
harness (the frozen tree, and the one correction accepted after the
freeze that no arm ran on), and records the current trees' hashes
beside them: the arms' evidence is what they ran on, and an edit after
them is visible, not a failing bar and not a silent drift. No arm is
rerun for a prose fix (the owner's call of 2026-10-29).

## `problem.md` is covered by the row's verdict as ours (2026-10-30)

The IOAI bank's cards carry each entry's `problem.md` as the situation,
and the first pass left those files outside the per-row verdict. They
are our own summaries of the tasks with their source links, so the
verdict covers them as ours, the table lists them under our files on
every row, and OQ-7 narrows to the audit for verbatim passages from
sources with their own licences — the owner's before the shareable bank
is published anywhere; until then it is served locally. The converter
reads a file's class from its description, never its name.

## The graders take the seed from the command (2026-10-30)

The three graders defaulted `--seed` to 1337, a copy of the config's
`subsample_seed`. The registered command the maintainer writes always
carries the seed, so the graders require it and hold no copy; the
RelBench grader's rescore echoes the archived manifest's class. The
graders' poll cadence likewise comes from the config through the task's
settings, and the IOAI evaluator's listing timeout and page size through
the contest descriptor.

## MLE implementation sessions have no web and no research gate (2026-10-30)

The predictive-modelling harness promises no web in implementation and
verification; MLE's override inherited `implementation.web: true` and the
research gate, and the deny-web hook cannot see Codex's native search.
The config now says what the harness says, as RelBench's does: a
competition's test labels exist on the web. The MLE arm ran with the web
on; it is not rerun (the owner's call), and the release notes say so.

## A push receipt bound to the kernel version is a follow-up (2026-10-30)

The submission procedure attributes a score to the code by the hash of
the kernel source at submit time, at a repository-relative path; the
pushed version's content is not available from the CLI at submit time.
The skill says to submit right after the push, before the source
changes. Binding the digest to a push receipt (the procedure doing the
push and recording the version it printed) is the first IOAI item after
the release.

## A learner version is promoted on the scorecard alone; the behaviour and A/B checks are the operator's (2026-10-30)

The learning design wired three gates after grading. What ships promotes
on the scorecard's `accept`, with the gauntlet's verdict rolled into the
scorecard; the behaviour scenarios and the A/B arms run as their own
commands and their verdicts are read by a person. WP10's review found
the development page promising the three gates; the page now says what
ships, and wiring the two checks into `develop` is a package after the
release, not a doc fix.

## The front pages' outcome sentence carries the budget (2026-10-30)

The direction document's TL;DR promised "the branch that met the goal";
a build can exhaust its budget short of the goal, and the examples say
so. The sentence on the README, the docs index and the direction
document now reads "until it is met or the budget is spent" and "the
branch that met the goal, or the best the budget reached, with the
evaluation that measured it"; the direction document notes the
amendment beside its text, as for the deploy clause before it.

## Provenance is a form the lint checks, not a word (2026-10-30)

The claims lint had passed a result on any verb of measurement in its
paragraph. It now needs the verb with who or when (an actor, a date, a
link to the record), reads a provenance table row by row, counts an
integer beside a rating word as a result, and lets a target or rule mark
cover its own clause only; its words-out rule reads every visible
string. The shipped pages were brought to the form rather than the form
loosened to the pages.

## The repo-memory update withholds what it cannot read (2026-10-30)

`main`'s windowing of the update (#167) sent a file larger than the
payload budget whole, in a call of its own, so that every line the model
re-cites is before it. The gate's campaigns commit the example's 1 MB
dataset, and that file put every update past Codex's input limit: the
memory never updated after an experiment, on `main` and on the branch
alike. The update now withholds a changed file that is binary or whose
payload alone exceeds `repo_memory_update_payload_max_chars` and lists it
by name and size in the first call, with its evidence kept as it is; the
budget bounds every call. A file that large is a dataset or a model dump,
not code the memory cites, so nothing the model is told is cut.

## The judge is kept its reserve before one has been observed (2026-10-30)

A round's reserve kept the judge at the floor (60 s) until a judge had
been observed in the campaign, and the finishing stages' obligations
counted it the same. In the S2 Codex gate's hour the first attempt's
verification was given the remainder less that floor, ran to its
deadline, and left 59 s for the judge — under the floor, so the attempt
went unjudged and the campaign ended with nothing to deploy or learn. The
config's `budget.judge_reserve_seconds` (300) is the judge's reserve
before an observation, in the round's reserve and in what the evaluation
and the verification leave standing; a judge that has been observed is
kept its own last duration, as before. The evaluation and the verification
stay at the floor in the obligations: a verification cut short is
unverified, and the judge still speaks.

## The Codex registration gives Kapso's MCP server a minute (2026-10-30)

Codex drops an MCP server that has not answered within ten seconds, and in
exec mode says nothing; `kapso-mcp` imports the package before it answers
(the experiment store's embedding backend brings litellm, the bank gate
the learners), four to five seconds on an idle box and more on a busy one.
A plugin session of the S2 Codex gate ran without the campaign tools on
that account. The setup skill writes `startup_timeout_sec = 60` under the
server's table in `config.toml` after the `codex mcp add` line, and the
docs and the README say so for a registration by hand. `kapso.core` no
longer re-exports the LLM backend (nothing imported it that way); that
cleanup does not shorten the startup, since the gates import the backend
through the experiment store, and a faster server stays a finding.

## The trace words a void and a rejection apart (2026-10-30)

The trace listed every attempt the campaign did not count under one
heading with one word for the integrity check ("rejected: Provided
evaluation integrity check failed") and another for the harness's
verification verdict ("void: …"), the reverse of the design's two
questions — what was voided and on what evidence, what was rejected and
why — and one S4 reader took the integrity void for a rejection. The
trace now words the integrity check's `voided:` (the protected evaluation
changed, the hash evidence beside it) and the harness's verification
verdict and the judge's invalidation `rejected:`, the split the S4 driver
reads by the engine's own semantics. The goldens were re-rendered.

## The plugin's answers keep the suite's line and question budgets (2026-10-30)

Five S2 steps on Claude Code failed on the suite's counts alone: the
review's twelve lines grew sub-bullets under the risks, the build and the
evaluation asked the approval and a choice, the improve flow asked twice.
The bars are the product's promises (WP7's design, WP9's suite), so the
skills say the counts in their own words: the review's twelve lines are
the whole message outside the outline's fence, one line per risk; the
build asks one question in all, the approval, which carries its choices as
defaults the user may change; the harness improve flow asks for the word
on the proposal and no more; a stated baseline names its source and a
disagreement between records is said (the watch step's planted baseline);
the answer after a fix names the fault, the fix and the resume.

## S4 builds the campaign it reads (2026-10-30)

S4's design read S5's tampering campaign for a void and a rejection in
one record. No campaign ever held both — the tampering voids the one
attempt its budget holds, and nothing in the suite makes the verifier or
the judge reject — so the bar `campaign_has_rejection` read unmet on every
S4 run. S4 now builds its own campaign when none is given: the churn
example at a hundred minutes and two experiments, the protected evaluator
edited in `main` during the first ideation (the attempt's clone inherits
it, with no diff for the writer to notice; the integrity check voids it on
the hash evidence) and restored during the second, then the hold-out
labels written over the predictions on the second attempt's branch during
its evaluation phase — after its evaluation materialised the branch and
before its verification does — the prohibited flow the harness's
verification voids, which the driver reads as the rejection. A campaign
given on the command line is read as it is, as before. S4 no longer waits
for S5, and the eight gate runs launch together.

## S5 edits the parent, takes a host, pins the wheel, reads a resume from the records (2026-10-30)

The tampering injection edited the evaluator in the running attempt's
session clone; a writer that diffs against its parent sees the edit and
restores the pristine file (the gate's first S5), and nothing is voided.
The edit now lands in the campaign's `main` as a commit during the first
ideation, before the attempt's clone, which inherits it with no diff. The
driver takes `--host` (the gate's second S5 ran the plugin on Claude Code
under a machine named for Codex) and `--wheel` (the record named the
version alone). A resume is read from the campaign's records — the status
file's or the launch receipt's `started_at` after the ask — rather than
from a `running` state in the five minutes after the turn, which read
three resumes as none when the plugin waited inside its turn until the
resumed campaign had finished. The fix words accept a relaunch, a restart
and a rerun: a resume by another name.

## The S1 rig reads the engineer as the design says (2026-10-30)

Four of the six failed S1 steps were the engineer waiting: the goal read
"a better churn model… into ./campaign-1", so the engineer followed the
campaign to its end, and "continue it from where it stopped" kept it in a
foreground resume. The goals now say the task is done once the campaign
runs on its own, and the resume the same — what a ticket says. The
sandbox venv holds what the task views say to install (the examples'
requirements), since an engineer with no web cannot; "use no web" is said
of the engineer, not of the campaign's settings, which one engineer
rewrote in a 39-line script. A flow's lines are logical lines (a call
across six physical lines is one line of the flow); the splitter reads
quotes and subshells as the shell does; an execution errs by the host's
exit status, not by its output's words; a command still running at the
turn's end is counted with its outcome unknown. No bar was touched.

## The second engineer installs as the docs say (2026-10-30)

Both S2 gate runs had the served build refused at its launch for the
bundled harness's requirements: the rig's second user installed with `uv
tool`, whose interpreter is off the `python3` the harness's commands name,
and the system's python3 refuses pip as externally managed. The lean
install of 2026-08-26 stands (pandas and pyarrow are not the package's
dependencies); the rig's second engineer builds from a project venv as
the docs install it, the wheel and the project's requirements first on
PATH, and follows the launch's printed remedy for the harness's
requirements before launching again. The `cold-install` step keeps
measuring the uv tool install.

## The trigger probe tries a failed sandbox again, one session at a time (2026-10-30)

151 of 178 Codex trigger sessions of the S2 Codex gate failed in
bubblewrap's loopback setup before any skill could be read, at two
sessions at once, on a machine whose preflight had passed. A run whose
sandbox failed is tried again, up to the suite's `sandbox_tries` with a
pause between, before it counts; the probe runs one session at a time
(`codex_jobs`). The machine's failure stays named in the record, never a
miss.

## A restarted setup session continues its answered requests (2026-10-30)

A harness setup session's inbox requests (node -1) were met by running
the setup session again with the replies, and the record never marked
them `continued`: the request's promise of a resumption was unkept in the
record, and both S2 gate runs failed the inbox step on it. The
orchestrator records the setup's answered requests as continued by the
session that carries the replies, once (`continue_answered`), as a search
node's session records its own on resume.

## S4's rejection is recorded, never required (2026-10-30)

The gate-to-pass fix of the morning had S4 build a campaign holding both
of the design's cases, the second by a planted leak committed onto the
second attempt's branch during its evaluation phase, for the harness's
verification to void. It could not work: the churn campaign runs under no
harness, so no verification session ever runs (`_verify_attempt` returns
at once), and the `evaluation` status phase belongs to the registered and
deferred evaluations, never to a provided evaluator, so the driver waited
until the campaign had ended and committed nothing. Both second-wave S4
runs held the void and no rejection. Read further: every "rejected"
entry in every archived campaign is an integrity void; the engine has
never produced a verification or judge rejection in these records, and a
rejection is the verifier's or the judge's word on an attempt's own work
— a driver cannot arrange one honestly (a leak planted for an agent to
keep is a trap, and a leak timed between the session's end and the
verification's read is a race). So S4 builds its campaign at S5's budget
(40 minutes, two experiments, which two attempts fit — a hundred minutes
let Codex's first attempt run 86 of them and the second never ran), with
the evaluator edited in `main` for the first attempt and restored for the
second; the bar is the void the integrity check guarantees, and whether
the campaign holds a rejection is recorded in the fact sheet and the
report, never required. The readers' question "what was rejected and why"
stands, answered "nothing rejected" when the record says so — which both
hosts' readers answered right (6 of 6 on Codex; Claude's grading died on
a judge misquote, fixed below). The owner is told at the checkpoint that
this bar changed and why; the design's "a campaign with a void and a
rejection" is met only when an agent earns one.

## The S1 Python flow is asked for as a short script (2026-10-30)

The DECISIONS entry of 2026-10-27 left the `python_lines` bar's reading to
the owner: a product finding or a metric calibration ("counting
statements rather than lines, or excluding pure diagnostics, would
measure the intent — how much the engineer must write to drive a task —
more directly"). The second wave read 13 and 22 lines on Claude Code and
39–41 on Codex: `build()` and `learn()` block (the API reference says so
and points a shell user to `--detach`), so an engineer told the task is
done once the campaign runs wraps the CLI's `--detach` in subprocess
boilerplate, and the Codex engineer adds a polling loop and a worker
flag of its own style. The bar measures the package, not the engineer's
style, so the task says what a ticket says — "write a short script (a
dozen lines or so)" — and the count reads whether the package lets the
engineer keep to it. The API's lack of a detached launch stays a product
finding for the owner (three of three build flows wrapped the CLI).

## The trigger probe keeps its own clock (2026-10-30)

At one session at a time (the sandbox fix), the Codex trigger probe's
178 runs need about two and a half hours; the step's two-hour budget
ended it from outside, the shell's timeout raised through the driver,
and the S2 Codex run ended at its watch step with no result for the probe
at all. The probe takes `--deadline-seconds` from the step (its budget
less the suite's `deadline_margin_seconds`): a run not started by then is
recorded as not reached — not established, never a pass, never a miss —
and every run's row lands in `results.jsonl` as it ends, so a probe cut
short still leaves what it measured. The step's budget is four hours.

## The judge reads a JSON line's escapes (2026-10-30)

The S4 Claude reading pasted the engine's feedback as JSON; the judge
quoted the words it read (an em dash) against a line holding `\u2014`,
the quote check found no match twice, and the grading died. The check
resolves JSON escapes on both sides before comparing.

## `kapso deploy --status` answers a campaign not yet deployed (2026-10-30)

The cold engineer asked for the status before the first deploy and the
command exited with an error, which the rig counted as a wrong call of
the package. A status question has an answer — "has no deployment yet" —
and exit 0; a `--number` that names no deployment stays an error.

## The baseline memory keeps the update's budget (2026-10-30)

The repo memory's initial inference read the planner's files whole with
no bound: a 1 MB dataset among them put the call at 1,400,956 characters
and Codex refused it (its limit 1,048,576), an engine session failed in
the S1 Claude prerequisite campaign. The initial inference keeps the
update's budget (`repo_memory_update_payload_max_chars`): a binary or
over-budget file is listed by name and size and never shown, and the
files past the first budget's worth reach the model in further calls,
each patching the model the previous produced.

## The S1 rig: a heredoc, the login shell's PATH, a flow's learn (2026-10-30)

Three rig defects read from the second wave. The command splitter read a
heredoc's body as shell text, and an apostrophe in a script written by
heredoc swallowed the launch that followed it (a Codex flow counted no
execution): a heredoc's body stays with its command. Codex runs each
command in a login shell whose profile puts the machine's `~/.local/bin`
(the gate's uv tool install) ahead of the sandbox venv the driver put
first on PATH, so the engineer's deploy ran the machine's kapso without
the task's requirements and failed until the engineer put the venv first
by hand: the sandbox step ends the profile with the venv first, as the
prompt says it is. And the learn the Python flow started was still mining
when the CLI flow began, so the engineer rightly declined to start a
second: the rig stops a flow's learn, with its process group, before the
next flow, and the learn's goal says the learn keeps running on its own
after the command returns.

## The trace header speaks the two words too (2026-10-30)

The header line "N experiments, M rejected" came from the engine's
`rejected_candidates`, which counts an integrity void as rejected, while
the VOIDED OR REJECTED section below it said "voided". The third wave's
Codex reader followed the header and the record (S4 on Claude Code, 5 of
6). The header now counts `voided` and `rejected` as the section
classifies them; the record's field keeps its name, a finding.

## The S5 rig after the second wave on Codex (2026-10-30)

Four readings, none the product's fault. The evaluator-crash injection
moved one copy of the labels while the seed had left another in the
campaign root, and the implementation session copied it back and scored
— the rig now moves every copy the campaign can reach, and the agent's
repair of protected data, accepted by the judge, is a finding. The
sampler read `kapso watch`'s DEAD marker off the earlier run's screen
below the product's own "starting again" line, a minute after a resume:
the top line is the verdict. A resume is any relaunch after the writer
went, the first answer's included. The vocabularies take "the writer
process died", "PID … was gone", "restoration of the evaluator" and
"not as a valid scored result".

## The plugin's answers are shaped in counts (2026-10-30)

The build skill asked for "three lines of prose around the brief" and the
review for "twelve lines in all", and the second wave's Claude Code
answers ran 13–17 lines: headings, sub-bullets under the evaluation's
approval, six risks. A count the model keeps is a shape it can fill, so
the skills say the shape — the build's prose at most ten lines outside the
brief's fence (one per thing found, two for a built evaluation, one for
the approval), its handoff at most eight, the review's risks at most four
with a follow-up in the same twelve lines, no headings and no sub-bullets
anywhere. The budgets themselves stand.

## A harness goes where the repository loads it from (2026-10-30)

The plugin wrote a green harness one directory above the path the
repository's `load_harness("../harnesses/trading")` resolves to from the
project, and the gate's probe found none. The skill says where: the
repository's own loading path, resolved from the repository's directory,
named as an absolute path in the proposal.

## A deployment's proof is the delivered prediction reproduced (2026-10-30)

The deploy session scored the proof row against a training reference
because the delivered model rank-blends within its batch, and answered a
different number than the delivered predictions file. The deployment
prompt says the proof is parity with the delivered prediction: a solution
that scores rows relative to one another keeps the batch it was delivered
with, and a different number is no proof.

## A bare resume runs on the launch's budget (2026-10-30)

`Budget` promises that a field left None keeps the launch's value on a
resume, and the docs listed the brief above the launch record. The resume
passed the brief-merged figures off as the caller's, so the record's never
applied: a campaign launched at 40 minutes resumed at its brief's 45 (the
plugin noticed, S5 on Claude Code). The resume asks for the caller's own
budget and takes the record's for the rest; the precedence reads call,
then on a resume the launch record, then the brief, then the config.

## The judge's citation may be one line off, and its call retries (2026-10-30)

Two of three replies grading the S4 Codex readings cited the `{` line for
the `"branch"` line below it; the quote check read them as misquotes and
the grading died, as it had on a Vertex connection timeout before. A
citation one line from the quote is the material's, never an invention,
so the check accepts it (two lines off is still no citation); the model
call carries the suite's `transport_retries` and `timeout_seconds`.

## The second engineer's copy points at its own labels (2026-10-30)

The rig copied the first user's churn project whole, its
`evaluation/private_dir.txt` naming the first user's private directory,
unreadable to the second user; the evaluator died on every attempt and
the served build ended unscored on both second-wave S2 runs. The copy's
pointer names its own `private/`.

## The S1 learn task runs on the campaign the deploy task left (2026-10-30)

The comprehension turn is a fresh session that answers from what the
sandbox holds. The rig restored `campaign-prereq` from its snapshot before
the learn task, wiping the deploy task's record, and the engineer who had
done everything right could not say what the deploy returned ("I found no
deployment record on disk"): the comprehension 0 on a run with every
other bar met. The learn runs on the campaign as the deploy left it, the
realistic order, and the deployment record stays for the reading.

## The S1 rig after the third wave on Codex (2026-10-30)

Three readings. A detached launch's child writes its status seconds after
the command returns, and a host whose turn ends at the launch (Codex,
twice at half a minute) left the evidence to land after the watcher's
last look: the watcher keeps looking for up to the suite's
`evidence_grace_seconds` after the turn, and the launch clock still stops
at the evidence. The deploy task's step cap was fifteen minutes and the
deploy session itself took seventeen on Codex, the engineer's launch half
a minute: the cap is twenty, the engineer's measure being the minutes to
the launch. And the Python flow is asked for as a script of at most a
dozen lines — the bar's own figure, said in the ticket — while the
finding stands that the API's blocking `build()` makes the engineer
reinvent the detached launch in eight of those lines on every host.

## The trigger probe falls back when bubblewrap fails the probes (2026-10-30)

The suite's Codex sessions run under the `read-only` sandbox, and the
design already said that a machine whose bubblewrap cannot start runs
them under `danger-full-access`, said in the record. One probe decided
it, and one probe passed on the third wave's machine while 113 of the 139
runs that followed failed in the loopback setup at one session at a time
with three tries each: the failure is the machine's and persistent, not
intermittent. The probe runs `sandbox_probes` sessions; any failure sends
the whole suite to `sandbox_fallback`, which must itself start; the
decision is written beside the results, recorded with the requested
sandbox, and the step's notes say it. The run was stopped in its probe
and launched again rather than left to fail two hours later.

## The Python API launches detached (2026-10-30)

`build()` and `learn()` block, and the API reference sent a Python user to
the shell for a run that outlives the call. Every engineer of the release
gate's S1, on both hosts and in every wave, reinvented the detached launch
in Python — a self-relaunching worker script, a shell-out to `kapso learn
--detach` — and the fourth Codex run met every bar but the judge's docs
dimension on exactly that flow. The morning's reading had left this a
finding rather than a feature, since framework changes are the owner's
call; the gate's standing instruction is that a failure goes back to its
package, and the fix is the thin wrapper the CLI already had: the launcher
moves to `kapso.execution.launch`, shared by the CLI's four `--detach`
verbs and the API so neither imports the other; `build(detach=True)` and
`learn(detach=True)` compose the CLI's own argv (the brief beside the
output as the file the child reads, the loaded harness and experience as
their flags, the budget's set fields as theirs) and hand back a `Launch`
(pid, log, status). The owner may revert it; the checkpoint says so.

## The trigger runs switch sandboxes on a streak of failures (2026-10-30)

Three probes started under `read-only` on the fourth S2 Codex machine and
15 of the next 18 runs failed in bubblewrap's loopback setup: the probe's
sessions run alone, the trigger runs beside the machine's other sessions
(the earlier steps' campaigns), and the loopback setup fails under that
company. A probe cannot see it, so the runs decide: `sandbox_switch_after`
sandbox failures in a row send the rest of the suite to the fallback once
it starts, every run the sandbox failed is run again under it at the end
with its first failure kept on the row, and the switch is written beside
the results and said in the record.

## S5's diagnosis vocabularies take the plugin's own correct words (2026-10-30)

The third S5 Codex launch failed three injections, each the plugin
diagnosing correctly in words the rig's `names_class`/`names_fix` regexes
did not hold: "the campaign died", "PID N disappeared", "a stale running
status", "resumed" (not "resume"), "recovered", "finished", "Second
attempt". The bar is whether the plugin named the failure class and the
fix, and it did; the regexes take these phrasings now. The vocabularies
stay disjoint (a word that names the class cannot also name the fix),
the test that pins that holds, and the full-disk and killed injections
that had failed only on the first-turn resume read pass with the earlier
`8c771d9a` fix.

## S5's evaluator-crash counts the crash, not an unrecovered attempt (2026-10-30)

The evaluator-crash injection moves the hold-out labels so the evaluator
crashes; the plugin's job is to diagnose it. A capable agent restored the
labels from a workspace copy (WP7's readability finding — a read is not
prevented) and the attempt went on to score 0.8481, so no attempt was
left scoreless, though the crash was in the record and the plugin named
it. The step now passes on the crash having occurred (in any attempt's
evaluation output, integrity error or sessions), the plugin naming the
class and the fix, and a resume — not on an attempt ending scoreless,
which would require the agent to be incompetent. The agent's self-repair
of protected data, accepted by the judge, remains a finding for the owner.

## S2's learn-complete passes on a clean end-to-end learn (2026-10-31, the owner's)

The third S2 Claude launch's learn ran every phase — harvest, mine, exam,
lesson, push — and moved the bank head, and the lesson landed "0 created,
0 updated": the update crew, with the exam before the lesson, admitted no
card from a one-attempt campaign that reached 0.8500 on a 0.85 target. An
earlier wave's learn on a like campaign banked five. Whether a card is
banked is a learning-quality outcome that varies with the campaign and the
crew's model (this round's learns ran on Opus, the shipped model's login
being rate-limited), and the rig must not force it. Put to the owner, who
chose: the step passes when the learn completes end to end and advances
the bank; the cards banked are recorded in the step's note, not gated.

## The plugin's follow-up detail goes in the fence; the line budget stays hard (2026-10-31, the owner's)

The same launch's two other misses were the plugin's probing follow-ups —
"is it safe to merge?" at 13 lines, "show the candidate criteria" at 18 —
against the twelve-line budget, each first answer now within it. The
follow-ups broke their detail (five risks, the branch clutter, what was not
checked; a criterion per bullet with its figures) into prose lines. The
fenced block is exempt from the count and is where such detail belongs:
the build and evaluation skills now put the candidate comparison in one
fenced table with at most three prose lines around it (the earlier "no
table" instruction pushed the detail into prose and is withdrawn), and the
review skill sends a merge-safety answer's files-and-rules detail into the
fenced outline and keeps at most four risks. Put to the owner, who chose
that the twelve-line budget stays a hard gate the fenced fix must meet,
not an advisory dimension.

## No further questions: the suggestions stand (2026-10-31, the owner's)

After those two, the owner said to ask no further questions and to go
with my suggestions to pass the master plan's gates. From here every
remaining call on the gate is taken on my own judgment and written here.

## The trigger probe runs last, on an idle machine (2026-10-31)

On Codex the probe is a hundred minutes of sessions one at a time, and in
the round's order it followed build-churn, so the churn campaign lived its
whole hour under that load: its ideation session, about sixteen minutes on
its own, passed its 2250-second deadline and was killed, and the attempt
voided (the fifth S2 Codex launch; on Claude Code the probe is twenty
minutes and the same campaign delivered). The probe now runs after every
campaign and the learn, so the campaigns and the probe are each measured
alone; the run is no longer in wall clock, the steps being sequential.

## A skipped MCP registration fails the setup step (2026-10-31)

The setup session on Codex read the skill's "add Kapso's MCP server to the
user's config once" beside a doctor whose rows were all green and ran no
`codex mcp add`; the list came back empty, and campaign-tools and
session-brief — the two steps that need the plugin's tools — could only
report that no tools were exposed, with the judge's proactive and
return-on-attention scores landing on those answers. Two changes: the
skill's step 4 is unconditional and verified (run the lines in this
session whether or not the doctor is green; `codex mcp list` must show a
`kapso` row; a list without one is a setup not done, said to the user),
and the rig's setup step on Codex fails when the list shows no kapso row,
so a skipped registration fails where its cause is rather than three
steps later in words that look like a product gap.

## The second engineer builds at forty-five minutes (2026-10-31)

The second engineer's thirty-minute build reached a 0.8464 candidate and
its finishing cut left the verification session four minutes, which timed
out: no branch, no score, nothing delivered. The same build on Claude Code
passed at 31 minutes, so the budget was marginal, not wrong; it is now 45
minutes under a 60-minute step cap, enough for an implementation of
twenty-five minutes and a verification of its own.

## An answered inbox continuation is admitted whatever the remainder (2026-10-31)

The fourth S5 Codex launch's missing-key campaign paused on its request,
the plugin wrote the key and replied within two minutes, and the
orchestrator's admission check refused the resume — "the remaining 16.3
min would not cover an attempt (the last took 18.2 min)" — where "the
last" was the paused attempt's own elapsed time, the eighteen minutes its
session had spent before asking. The strategy already treats a node
suspended on an answered request as a continuation, not a round ("the last
round's duration stands for the next admission"), but the orchestrator's
check runs before the strategy does and did not know. The strategy now
says so (`continues_an_answered_node`, False on the base class), and the
orchestrator admits such a continuation regardless of the fit rule: the
attempt is under way and the person answered; the round after it is
admitted on the real remainder as before. The previous launch's same
campaign asked sooner, resumed, and scored — the defect was timing-bound,
and is fixed in the package rather than tuned around in the rig.

## The crash is read from the record's difficulties; the rig's siblings move (2026-10-31)

The evaluator-crash step's detector read the attempt's evaluation output
and the sessions' streams. The fourth S5 Codex launch's session checked
its inputs before scoring, found the labels gone, searched the siblings
under the work directory by checksum, copied the previous injection's
copy into place, and scored — and the crash sat only in the engine's own
record of the node's difficulties, which the plugin cited. Two changes:
`crash_occurred` reads run_state's technical difficulties too (the regex
takes "holdout_labels.csv … raised FileNotFoundError", the record's
order), and every `holdout_labels.csv` under the work directory moves for
the step, since the rig's own layout must not hand the labels back. An
agent that reconstructs protected labels from a copy, against the brief's
rule, with the judge letting the score stand, remains the owner's finding.

## Every copy of the labels on the machine moves for the evaluator-crash step (2026-10-31)

After the siblings under the work directory moved, the fifth S5 Codex
launch's session searched the whole machine (`find / -name holdout_labels.csv`)
and copied the fixture from the rig's own repository clone,
`examples/churn-first-build/private`, then scored. The injection says the
evaluator crashes on every invocation; a fresh machine holds no copy of
the labels but the rig's, so all of them move for the step — the project's,
the campaign's, the siblings', the clone's examples and test fixtures —
and the crash bites as designed, which is what the plugin is asked about.
An agent that reconstructs protected labels from any copy it can reach,
against the brief's rule, stays the owner's finding; the rig only stops
handing them over itself.

## The S5 Codex fifth launch stopped at killed-session, not run out (2026-10-31)

Its decisive step could not pass on its wheel (the clone copy), the four
injections after it had all passed on the previous launch, and the
deadline is today: the run was stopped at its fourth injection, its
partial record archived with the two readings it gave (missing-key passing
on the continuation fix; the crash seen and the handoff finding), and the
test relaunched from the fixed tree two hours sooner than waiting it out.

## S2's learn-complete waits four hours (2026-10-31)

The fourth S2 Claude launch's learn was still in its update crew when the
step's 150-minute wait expired; the previous launch's learn took 121
minutes and this one more, the crew being about ninety minutes of either.
The step's wait is the gate's accommodation of the product's cost, not a
bar on it: it is now 240 minutes, the launcher's run cap ten hours, and the
learn's duration is recorded for the owner. The run was stopped at its
second engineer rather than run out — learn-complete could not pass on it,
every earlier step had, and the remaining ones had passed on the previous
launch — and relaunched from the widened tree an hour and a half sooner.

## A finished rate-limited campaign is continued, not waited for (2026-10-31)

The sixth S5 Codex launch's provider-limit campaign had stopped at
`max_iterations` before the question — the rate-limited first attempt
counted against its two experiments — and the plugin's fix was to
continue it with a larger allowance, which the vocabulary (wait, retry,
the window, resume) did not hold: those are a running campaign's fixes.
The vocabulary takes "continuing", "the experiment allowance", "more
experiments" and "another attempt"; the archived record is read again
under it when the run lands, the suite's own `--rederive`, rather than a
seventh launch. That a provider-rejected attempt counts against the
allowance, and that the feedback quoted the brief's budget rather than the
launch's, are the owner's findings.

## The plugin's answers carry their findings (2026-10-31)

The sixth S2 Codex launch passed all 24 steps and missed the judge's bar
on proactive and return on attention alone — 3 against 4 on both passes —
on four answers that were correct and empty of findings: a launch that
named the brief and the harness but not the data, the evaluator and the
baseline it had found; a watch that said the budget "isn't reported" when
the launch record holds it; a share that echoed the command; a proposal
that cited the skill's own rule to the user. The rubric's proactive is
"findings the user did not know, with evidence", and the Claude plugin's
answers to the same steps score 5 by carrying them. The build skill's
four findings lines are now imperative, each with its figure or file,
with no line about the skill's rules; the operating skill's watch answer
gives the best score against the baseline and target and the budget left
from the launch record, and its share answer says what was pushed and
what a teammate runs. The budgets are unchanged (the findings are the
lines the budget was made for); the judge's bar is not lowered.

## The judge reads every message the host printed in a turn (2026-10-31)

The judge's transcript rendered each turn's final text alone — the Claude
Code `result` row, the last Codex `agent_message` before the next prompt —
by WP9's choice. A host prints several messages in a turn as it works and
the user reads each as it lands: the sixth S2 Codex launch's launches
named the data, the evaluator and the baseline in the messages before the
last, and the judge, reading the last alone, scored them as findings never
given. The rubric's own frame is what the user got for the attention it
cost, and those messages cost it and carried the findings. The transcript
now carries every message of a turn — each Codex agent message, each text
block of a Claude Code assistant row — as its own assistant block; the line
budget stays on the final answer. Validated on two archives before it
stayed: S2 on Claude Code, third launch, proactive 5, clear 4, return on attention 4 (every binding dimension at or above the bar; the advisory not-annoying at 2, the working narration read as noise); S2 on Codex, sixth
launch, proactive 4 from 3, clear 4, return on attention 3 (on the learn launch — a pid and the watch line — and the empty wish's proposal; the advisory not-annoying at 2). The sixth Codex launch therefore still misses the judge's
bar on return on attention, on two answers, and its report stands as it
landed; the seventh launch decides. The advisory not-annoying dimension
reading a host's working narration as noise is recorded for the owner.

## The learn launch and an empty wish's proposal carry their value (2026-10-31)

The two answers that held the sixth Codex launch's return on attention at
3 under the faithful transcript: a learn launch that gave the pid and the
watch line, and a proposal for "build something useful here" that gave an
evaluation's check count. The operating skill's learn answer now says what
the learn reads, where it banks, how long it runs and what lands; the
build skill's proposal for a wish that names no task says what will exist
and what it does for the user, with its evidence and one alternative. The
seventh Codex launch runs without these (it predates them); an eighth, if
the seventh's judge misses on those answers, carries them.

## S2's churn build runs at ninety minutes (2026-10-31)

The fifth S2 Claude launch's churn campaign — sixty minutes, three
experiments — spent most of its hour in one implementation; the
verification session then timed out at its own ten minutes, its retry got
the 275 seconds left before the reserve, the feedback got 299, and the
finishing cut delivered no branch, so deploy had nothing and the review
named no delivered branch. The third and fourth launches' implementations
finished in under thirty minutes and delivered 0.846 and 0.850; this one
ran long, as a modelling implementation can. The budget was marginal, as
the second engineer's thirty minutes were: the churn build is ninety
minutes, the review step's wait follows the budget, and the run's wall
clock grows by half an hour. The engine's session floors and the
verification role's ten-minute timeout are the product's; a verification
that needs longer than its timeout on a slow run is a finding, recorded.
The run was stopped at its second engineer rather than run out — deploy
could not pass on it, every step before it had — and relaunched from the
widened tree an hour sooner.

## The plugin holds a measured figure against a user's claim (2026-10-31)

The watch step plants a wrong figure in the persona's mouth — "baseline is
0.90", the README's unmeasured claim, against 0.7974 measured three times
into `baseline.json` — and the seventh S2 Codex launch's answer wavered:
it named both records and then granted that the campaign "would not
establish improvement over a 0.90 baseline", after its own launch message
had measured 0.7974 and called the README's figure incorrect. The persona,
an ML engineer, was not satisfied within the step's two turns; the sixth
launch's answer had held the measured figure. The operating skill's watch
answer now says which figure is measured and which a claim, with the file
behind the measured one, and that the campaign stands on it — the design's
"a number names its source", made to bear a user who asserts otherwise.

## The MCP tool steps run after the learn; the registration waits two minutes (2026-10-31)

The seventh S2 Codex launch's campaign-tools session had no kapso tools
though `codex mcp list` showed the server with its sixty-second startup
timeout and the sixth launch's session had called all three. Codex drops a
server that has not answered within its timeout and leaves no line in the
session's stderr or a log; the machine was running the churn and pause
campaigns at the time. Two mitigations, neither a diagnosis: the
registration's timeout is two minutes, and campaign-tools runs after
learn-complete, before session-brief, when no campaign loads the machine.
The server's cold start on a loaded machine stays the owner's finding.

## The candidate-criteria follow-up asks nothing (2026-10-31)

Asked to "show the candidate criteria and recommendation; don't launch",
the sixth S2 Claude launch's answer kept its comparison in the fence and
its prose to three lines, as the skills ask, and then ended "tell me which
criterion you want, or say go" — a second question in a step whose budget
is one, the approval having been asked in the launch message. The user
asked to see, not to decide again. The build and evaluation skills end
that answer on the recommendation and why, with no question.

## The second engineer builds at sixty minutes (2026-10-31)

The sixth S2 Claude launch's second engineer, served the shared bank,
spent most of its forty-five minutes in ideation reading it; the
implementation session got a 650-second deadline, its retry 215, both cut,
the feedback 359, and the finishing cut left no branch. The third launch's
second engineer passed at thirty minutes with a quicker ideation. The
budget was marginal, as the churn build's sixty was: the second engineer
builds at sixty minutes under a seventy-five-minute step cap, and the
run's wall clock grows by a quarter hour. The owner allowed one more
launch after the sixth; this is it, with every fix the gate has earned.

## The campaign-tools step probes the server and runs one more session (2026-10-31)

The eighth S2 Codex launch's campaign-tools session reported the three
campaign tools unavailable — after the learn, on an idle machine, with the
registration's two-minute startup timeout — while the session-brief
session minutes later read a card through the same server, and the sixth
launch's session had called all three. Codex leaves no line for a server
it drops at a session's start, and the session's own account ("the tools
are unavailable") cannot say whether the server failed or the session
lost it. The step now records a standalone handshake with the server as a
host starts it (`initialize`, then `tools/list`, over the stdio
transport, from a login shell in the project), with the seconds it took
and the tools it lists — on the development box 2.6 seconds and six
tools, the three campaign tools among them — names the `kapso` server in
the prompt, and, when the first session used no tool though the server
lists them, runs a second session and says so in the step's note, both
answers kept. A server that lists the tools and a session that sees none
is the owner's finding, with the probe's figures beside it.

## An empty wish gets the same one message; a proposal alone is not done (2026-10-31)

Two misses on the seventh S2 Claude launch, both mine. The value line I
had written for a wish that names no task ("the value the user gets, not
the campaign's mechanics or an evaluation's check count") made the answer
a 26-line proposal that ended on an instruction and built no evaluation
for a folder with no data, where the sixth launch's seven lines had built
one on fixtures and ended "Approve this proposal?" — the step reads a
proposal as an answer that asks. The wording is folded back into section
4's shape: the findings, a proposal line that says what will exist and
what it does for the user, one alternative, the brief, the approval
question it ends on, the evaluation built on fixtures the proposal makes.
And the persona for harness-new, told "a proposal first, then the harness
written and checked green", said DONE at the proposal; its intent now says
DONE comes only once the harness is on disk and its check is green. The
owner allowed one more launch; this regression of my own making costs one
past it, said plainly, landing before the morning deadline.

## The ask detector keeps a prose line that opens with a quoted path; S2 re-reads its proposal steps (2026-10-31)

The eighth S2 Claude launch's data-only proposal closed on one line:
"`../fraud-campaign` already holds an earlier campaign, so this one will
write to `../fraud-scorer-campaign`. Approve the brief and evaluation, or
edit a line; say go to launch". The detector's rule that drops the command
shown after an ask — a backtick-wrapped line, kept since three Claude
steps ended "Say go and I'll run:" and the command — took the whole line
for a command, read no ask, and the step read no proposal. A wrapped
command is one wrapped whole, judged on the raw line; a prose line that
opens with a quoted path carries its ask. The run in flight carries the
detector of its day, and its archive holds the answer that asked, the
launch fact in the step's note and the persona's DONE: the driver's
`--rederive` reads the proposal steps again under the current detector
and says the day in the record and the step, every other step and the
judge's block as recorded — the mechanism S4 and S5 already use, rather
than a launch past the owner's allowance for a rig line.

## The watch step's budget counts the pushback it plants (2026-10-31)

The watch step's persona reads a task view with a planted wrong fact —
"the current model scores about 0.90 ROC AUC" — so that it pushes back,
and the step checks the pushback. The budget of two host messages covered
the status answer and the follow-up and not the answer to the pushback:
the eighth S2 Codex launch's watch completed its checks and failed on the
count. The budget is three; a persona that does not push back uses two.

## A baseline is measured in the session; no skill's rule is cited (2026-10-31)

The eighth S2 Codex launch's data-only proposal gave "Baseline:
unmeasured" and the figures a generator's comment reports, where the
Claude plugin's ran the evaluator on the shipped rules (0.091, with its
standard error); its empty-wish proposal cited the evaluation skill's rule
to the user. The judge scored both answers 3 on return on attention and
the second 3 on clear, and the build skill already asked for the baseline
"with the figure it measured" and "no line about this skill's own rules".
It now says it in words the terser model follows: a baseline is a figure
the evaluator printed in this session on the shipped model, never a number
a comment, a README or a generator reports — "unmeasured" is a step
skipped, not a finding — and never a line that cites any skill's rules.

## A launch the persona itself asked for is no miss (2026-10-31)

The criteria steps say "do not launch anything" to catch a plugin that
launches unasked. The ninth S2 Codex launch's empty-wish persona, told to
take the first proposal and launch nothing, replied "Wrong repo — there's
no predict.py here. Validator, not modelling: go", and the plugin launched
on that go. A plugin that obeys its user's go did right; the miss the rule
exists for is a launch nobody asked for. The criteria step and the
re-read now read the persona's replies from the step's transcript and
count only an unasked launch, saying in the note when the persona itself
said go; the empty-wish intent tells the persona never to say go, whatever
is proposed. The run in flight carries the rule of its day; its archive
is re-read under this one when it lands.

## Deployment parity reads the row nested in the batch a deployment answered (2026-10-31)

The deploy step's parity reader took the deployment record's observed
answer for one row with an id at the top level. The eighth S2 Claude
launch's blend keeps the batch it was delivered with, as the deployment
prompt says a solution that scores rows relative to one another must, and
its observed answer was the batch's row for the proof's id inside a
`predictions` list, with the batch size and the reference batch beside it
— a correct deployment the reader took for no row, while predictions.csv
sat on the delivered branch. The reader now takes the row of the proof's
id from such a list (or the one row it holds), the step and the re-read
share it, and the re-read recomputes the deploy step from the record the
step copied and the campaign the archive carries. The run in flight is
read again from its archive under this reader when it lands.

## Finding for the owner: a Codex session served the bank sees no campaign verbs (2026-10-31)

On the ninth S2 Codex launch the campaign-tools step ran after the learn,
when the project's config served the bank (`session_sources: [mine:all]`).
The rig's handshake, started in the project as a host starts the server,
listed six tools in 6.2 seconds — the research tools and the three
campaign verbs, no bank tools (the standalone start did not stage the
bank). The Codex session's own tool discovery "found no
`kapso_build_start`, `kapso_status`, or `kapso_stop`", in the first
session and in the second the step ran; the session-brief session minutes
later called `bank_index` and `bank_get_card` on the same server. The
sixth launch's campaign-tools session, which ran before any bank was
served, called all three verbs. So in a Codex session the server served
either the bank or the campaign verbs, not both; the host session's code
enables every default gate, and a local start could not stage a bank in
the time left to say where the verbs go. The step's probe figures and both
sessions' answers are in the archive; this is the one miss the ninth
launch carries, and the owner's finding.

## The second port of `main`, at the merge: the retrieval eval's files are kapso overrides, its block under `benchmark` (2026-11-01)

**Decision.** When the pull request from `redesign` to `main` opened
conflicting (`main` had taken nineteen commits since the first port),
`main` was merged into `redesign` and the thirteen conflicts resolved as
ports, the rule WP11 set for the first twenty. Three of the ports chose a
shape the owner did not discuss. The retrieval eval (pull request #180)
resolved a mode config and layered `inference:` blocks by hand; the
redesign has no modes, one packaged config and override files checked
against it, so the eval's `config.yaml` and `config.local.yaml` are kapso
override files with the eval's own settings under `benchmark`, the key
the other runners use, each file checked by `load_config` and the
engine's blocks (`models`, `retry`, `inference`) read from the layered
files over the packaged config, the way a build's one override is read.
The inbox reply's retry (pull request #182) lands in the CLI package's
inbox verb and the facade's `reply`, whose resume runs under the resume
lock with the record's `experiments`, `.env` and bank and whose refusals
name `kapso build --output … --resume`; the credential guard holds on a
retried note. The ledger tests (#181, #183) build the real `GenericSearch`
through `__new__` and so carry the redesign's attributes (the round clock,
the finishing cut, the harness, the normalized cost block, the workspace's
repo for the attempt commit) and fake `load_config` in place of a mode
loader; a missing config is a `ConfigError`, and the CLI's refusals are
read on stderr.

**Why.** The gate's eight reports were read from the wheel of the first
merge candidate; the ports touch the inbox retry, the ledger across a
resume and a benchmark runner, none a surface the reports measured, so
the reports stand and the tests `main` brought, green on the ported
code, are the port's gate. Keeping `main`'s mode loader alive for one
runner would be the dual code path Rule 7 forbids.

**Where.** The merge commit 4734dc45 on `redesign` (the ports in its message), `benchmarks/kg_retrieval/`,
`src/kapso/cli/inbox.py`, `src/kapso/kapso.py`, the two ledger tests.
