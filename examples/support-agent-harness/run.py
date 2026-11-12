"""The support agent's harness, built on the agent-engineering harness.

Run from this directory after `python runner/configure.py <runner dir>`; the campaign lands in campaign/."""
from kapso import Kapso, Brief

kapso = Kapso(config="repo/kapso-config.yaml")
kapso.load_harness("agent-engineering")
solution = kapso.build(
    Brief.read("kapso-brief.md"),
    evaluation="evaluation", data="data", repo="repo", output="campaign",
)
print(solution.explain())
