# The task runner

The runner is a public conversation benchmark for customer support over a bank's knowledge
base and account tools. Each task is one customer with a goal. A simulated customer talks; the
agent side replies and calls tools: searches over the documents (keyword, dense and a sandboxed
shell) and reads and writes on the account database. The runner drives the loop: it builds the
prompt each turn (the policy, the conversation so far, the tool definitions), sends it to a
chat-completions endpoint, executes the tool calls that come back, feeds the results in, and
repeats until the customer ends the conversation or a limit is hit.

## How a conversation is scored

Deterministic given the conversation. For most tasks the final state of the account database
must match the expected state; for a few, the required write actions must have been made. No
model judges anything. A conversation that ends in an infrastructure error (the model answered
with nothing, or the context window overflowed) counts as failed.

## The plug-in

The runner is pointed at the harness service instead of the model endpoint. The service is
an OpenAI-compatible chat-completions endpoint; the runner cannot tell the difference. The
harness may change what the model sees and what comes back; it cannot change the loop, the
step limits, the tools or the simulator. Those, with the model and its serving settings, are
the fixed conditions in `fixed_conditions.json`.

## What the runner publishes per conversation

Its results file holds, per conversation: the task id and trial, the outcome and its basis,
the termination reason, an error when one ended it, the messages with their roles, tool calls
(names and arguments) and token usage, the duration, and for each task a reference route: the
actions the task expects, each marked as made or not. The adapter in `bench/adapter.py`
converts this to `results/episodes.jsonl`: outcome, termination, the error as a category
(`overflow`, `empty_reply`, `timeout`, `endpoint_error`), counts and tokens, the calls as
names, tool names, kinds and argument names, and the reference route as names, kinds, matched
flags and the names of the arguments that differ. No expected value is ever written.

## What is private

The task definitions with their expected final states, the simulations directory where the
runner writes its raw results, and the split file. They live in the runner's own directory,
outside this repository; `protection.json` lists them by absolute path, so a tool that must keep
them out of reach can. The run script's runner reads them; nothing else does.

## The run script

`python bench/run_dev.py [--fast]` starts the service from the repository root, waits
for the model endpoint, runs the development tasks through the service at the fixed settings,
converts the results, writes `results/` and prints `pass_1` last. One full run is about an
hour and a half; `--fast` is five tasks once, about fifteen minutes, not comparable with a full score.
`reference/fable-5.1/` is the current agent's run: Fable 5.1 through this same runner on the
development conversations, converted the same way, with the transcripts beside it;
`reference/opus-5.5/` is Opus 5.5's run, recorded the same day.
