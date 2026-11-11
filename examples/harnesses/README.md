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
| [`data-engineering/`](data-engineering/) | data-engineering changes delivered with measured evidence: fourteen skills, five scripts (lineage, invariants, fingerprints, the evidence a change's label is computed from, a pull-request proof), two delegate reviewers, four hooks, and two delivery targets of its own (one pull request per ticket, a reversible warehouse release); dbt on a SQL warehouse as the worked stack. The [dbt on Snowflake example](../dbt-snowflake/README.md) runs on it |
| [`agent-engineering/`](agent-engineering/) | building the harness around a fixed model (the done-check, the tool layer, the turn cycle and its bounds, the context policy, the guards) so that it does a job a task runner scores: eleven skills with three scripts (pass^k over paired runs, episode triage, an endpoint probe) and two delegate auditors; customer support on a local model as the worked bias. The [support agent example](../support-agent-harness/README.md) runs on it |

The trading harness's scripts and servers run as written: the hook scripts
read a hook event on stdin and print a decision, the lookahead scan reads a
diff on stdin, and the two servers are FastMCP servers over a local store of
CSV or parquet bars. They are examples of the interface, not a trading system.

The data-engineering and agent-engineering harnesses were written from
researched expert mental models and revised after independent reviews. Each
has since carried a full build; the two examples linked above describe the
setup.
