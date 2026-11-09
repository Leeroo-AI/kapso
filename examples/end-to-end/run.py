"""The full loop on Spaceship Titanic: build, the review, learn, build again served the lesson, deploy.

Run from this directory with the data in repo/data/; the campaigns land in campaign/ and campaign-2/."""
from kapso import Kapso, Brief

kapso = Kapso()
brief = Brief.read("kapso-brief.md")
first = kapso.build(brief, repo="repo", output="campaign")
print(first.explain())                       # the review: the score, the stop reason, the cost
lesson = kapso.learn(first)                  # what the campaign taught, banked with its evidence
print(lesson.explain())
kapso.load_experience("mine")                # the bank, served to the next build
second = kapso.build(brief, repo="repo", output="campaign-2")
print(second.explain())
software = kapso.deploy(second, target="local")
print(software)
