# Harness examples

A harness is the domain layer of Kapso: one directory per domain that tells
the engine's agents how work is done there, with what, under which rules,
and nothing about any particular task. The interface is described in
[`docs/research/harness-design.md`](../../docs/research/harness-design.md)
and the authoring guide in
[`docs/research/harness-authoring-guide.md`](../../docs/research/harness-authoring-guide.md).

| Example | What it shows |
|---|---|
| [`trading/`](trading/) | the complete shape: seven instruction pages, seven skills (one with a script, one a deployment workflow), two delegate agents, two local tool servers, seven hooks listed per module |
| [`predictive-modelling/`](predictive-modelling/) | the manifest only, to show the same shape on a second domain |

The trading harness's scripts and servers run as written: the hook scripts
read a hook event on stdin and print a decision, the lookahead scan reads a
diff on stdin, and the two servers are FastMCP servers over a local store of
CSV or parquet bars. They are examples of the interface, not a trading system.

The harnesses that ship inside the package are under
[`src/kapso/harness/bundled/`](../../src/kapso/harness/bundled/) and load by
name; `kapso harness list` prints them. Two of them carry an example here:
`data-engineering`, which the [dbt on Snowflake example](../dbt-snowflake/README.md)
runs on, and `agent-engineering`, which the
[support agent example](../support-agent-harness/README.md) runs on.
