"""Pathology tiles on the predictive-modelling harness, scored on new patients.

Run from this directory, after `python generate.py`; the brief's Harness line
names the harness, and the campaign lands in campaign/."""
from kapso import Kapso, Brief

solution = Kapso().build(
    Brief.read("kapso-brief.md"),
    evaluation="evaluation", data="data", repo="repo", output="campaign",
)
print(solution.explain())
