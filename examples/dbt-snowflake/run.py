"""The backlog build on the data-engineering harness, from Python.

Run from this directory once repo/ is assembled and pushed and the platform answers on
localhost:8765 (README, "Set it up"); the campaign lands in campaign/."""
from kapso import Kapso, Brief

kapso = Kapso(config="repo/kapso-config.yaml")
kapso.load_harness("../harnesses/data-engineering")
solution = kapso.build(
    Brief.read("kapso-brief.md"),
    evaluation="evaluation", repo="repo", output="campaign",
)
print(solution.explain())
