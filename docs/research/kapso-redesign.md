# Kapso redesign: positioning and interfaces

*Started 2026-10-21; the positioning reset on 2026-11-06 to the launch
story: an experiment-to-delivery engine, with learning, and on 2026-11-09
to the final post: the premise (two things learned by experiment), the
engine paragraph, and the vocabulary (the word "loop" is out). The
positioning is settled first; the interface and architecture changes
follow from it.*

## 1. One-liner (settled 2026-11-06)

> Kapso is an open-source engine that runs experiments, deploys the best solution, and learns from every run.

Why this one: it names the three things the engine does, in the order a
run does them, and nothing else. The previous one-liner (2026-10-21,
"turn coding-agent output into verified outcomes, not review debt") put
Kapso beside the coding agents; this one puts it where the founders' own
work was: the experimentation that ends in a delivered, measured result,
and the lessons kept for the next run. It still names no category:
"applied research", "auto-research", "iteration engine", "software
factory" and "long-running agents" stay rejected.

## 2. TL;DR (settled 2026-11-09 from the final launch post; the README's "What is Kapso?" and the docs index carry it)

The premise. Delivering an AI or data outcome inside a company means
learning two things by experiment. First, the environment: no individual
or AI agent knows how a data estate is connected and operated, and the
only way to learn it is to investigate the company's artifacts (repos,
pipelines, documentation, data sources), propose hypotheses, run
experiments and learn from them, until each system's behaviour and its
conditions are understood. Second, the solution: the best one cannot be
picked in advance, whether the job is optimizing a warehouse or
predicting churn, so you experiment and learn until one fits the
environment. The know-how that fits the environment accumulates, much as
in RL. RAG does the opposite: it feeds the agent general knowledge that
was never tested in this environment, all of it carrying the same weight,
and that knowledge has little value until it has been tried there.

The engine. Kapso starts from a domain harness and an empty bank of
lessons. Given an objective and success criteria, it runs a tree of
experiments, proposing hypotheses, then implementing and verifying each
one, and deploys the best solution on the target stack. It then learns
from the run, assigning credit to each hypothesis and implementation by
its impact on the outcome and updating the bank, where every lesson
carries a credence score and a boundary on where it applies. Over time
the domain harness carries a bank of lessons that accumulated
experimented best practices in that environment. The system is
model-agnostic and coding-agent-agnostic.

Four parts, in the order a run meets them:

1. **Smart experimentation.** For each experiment, an ideation module
   proposes ideas from earlier experiments and the bank of lessons,
   re-ranks them by expected return against the budget each would use,
   and sends the top-ranked ones to the implementor. Because the bank
   keeps learning what works in that exact environment, the ideas become
   specific to it, rather than the generic ones a bare LLM or RAG would
   offer. Results are verified by a fingerprinted evaluator that no
   experiment can access, and the evaluation evolves too: the
   experimentation modules can propose updates to the metrics and
   criteria for the verifier to approve.
2. **Learning.** Everything in the bank comes from experiments: each
   lesson is induced or deduced from the traces of experimentation runs,
   carries a credence score, boundary conditions on where it applies and
   the evidence behind it, and after every run `learn()` updates those
   from the lesson's effect on the final outcome. Changes to the bank are
   experiments too: every updated lesson is A/B tested by running the
   same build with and without it, so an improvement is measured, not
   guessed. The bank is built on git, so you can share your experience
   with others (`kapso experience share`), list the banks available to
   you (`kapso experience list`) and update them (`kapso experience
   update`).
3. **Deployment.** `deploy()` runs a shorter experimentation of the same
   kind to adapt the best solution to the target stack (Docker, Modal,
   BentoCloud, LangGraph Platform, or a target a harness adds): a session
   adapts a copy of the node, deploys it, and demonstrates health checks
   and a working call.
4. **Domain adaptation.** Harnesses are Kapso's domain packages:
   knowledge and workflows for an area such as predictive modelling, so a
   build does not experiment and learn from scratch. Your initial domain
   harness (instructions, code, guardrails and checks) is written with
   `/kapso:harness` in the plugin or `kapso harness` in the CLI.

## 3. Tags (settled 2026-11-09)

Smart experimentation · Learning · Deployment · Domain adaptation

## 4. Rules for every public sentence

- Words out: "loop" (readers hear a coding-agent loop; say experimentation,
  campaign, cycle or learning), "with a number", "judge" (say evaluation),
  "trust", "applied research", "iteration engine", "long-running agents",
  "self-improving", "software factory", "team of agents", "always-on".
  (The word autonomous left the list on 2026-11-09: the IOAI track's own
  term for its format is the one fact that needs it.)
- Every number travels with who ran it, on what evaluator, and when.
- Learning is told as the launch post tells it (2026-11-09): everything in
  the bank comes from experiments, the lessons carry a credence score and
  a boundary, and every change to the bank is A/B tested, so an
  improvement is measured, not guessed; a number for the gain travels with
  the A/B that measured it.
- The engineer stays in charge: their goal, their evaluation when they have
  one, their merge.

## 5. The product shape the one-liner implies

- **Engine**: the four parts as one run: the tree of experiments and the
  modules that read and write it (ideation, implementation, verification,
  feedback), the isolated and versioned evaluation, the trace, deployment
  by a shorter experimentation of the same kind, learning into the bank of
  lessons.
- **Harness**: the domain package, a plug-in for one kind of work: the handler (scoring
  direction, stop semantics, task context), per-module instructions
  (propose, build, judge, deploy, the entrance's evaluation templates),
  extra tools, default budgets and gates. Shipped harnesses are today's
  benchmarks and examples; anyone can write one (trading, science). The
  trust boundaries stay in the engine; a harness never reaches the
  evaluation guard, the budgets, the inbox or the checkpoint.
- **Experience**: what the engine knows. Learned itself (the lesson bank,
  evidence-priced) or through others (the hosted knowledge base, imported
  repositories and papers), with provenance shown.

Interface sketch from the whiteboard (to be designed in §7):

```python
kapso = Kapso()
kapso.load_harness("predictive-modelling")      # or a path, or a git URL
kapso.load_experience("mine")                   # the bank; or "leeroopedia", a repo, findings
solution = kapso.build(goal, evaluation=...)    # evolve becomes build
software = kapso.deploy(solution)
experience = kapso.learn(solution)
```

## 6. Open items

- The TL;DR and the tags: settled 2026-11-06 from the launch post and
  carried to the README, `docs/index.mdx`, the PyPI summary, the CLI's
  help line, the plugin's manifests and the Context7 file in one change;
  reset 2026-11-09 to the final post (the premise and the engine paragraph
  above, "Learning" for the second part, "loop" out of every public
  sentence) and carried the same way, with the hero figure's captions and
  the README translations regenerated. The claims lint still holds every
  public sentence to section 4.
- The entrance when there is no evaluation (the engine proposes one, the
  judge checks it is fair): keep the three-case design from
  `docs/plans/kapso-plugin-user-flow.md` §4.
- The harness contract and the `load_harness` / `load_experience` /
  `build` / `deploy` / `learn` interface: design next, after the story.

## 7. Interface redesign

To be written once the story is locked.
