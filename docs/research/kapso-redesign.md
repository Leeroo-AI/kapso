# Kapso redesign: positioning and interfaces

*Started 2026-10-21; the positioning reset on 2026-11-06 to the launch
story: an experiment-to-delivery engine, with learning. The positioning
is settled first; the interface and architecture changes follow from it.*

## 1. One-liner (settled 2026-11-06)

> Kapso is an open-source engine that runs experiments, deploys the best solution, and learns from every run.

Why this one: it names the three things the engine does, in the order a
run does them, and nothing else. The previous one-liner (2026-10-21,
"turn coding-agent output into verified outcomes, not review debt") put
Kapso beside the coding agents; this one puts it where the founders' own
work was: the experiment loop that ends in a delivered, measured result,
and the lessons kept for the next run. It still names no category:
"applied research", "auto-research", "iteration engine", "software
factory" and "long-running agents" stay rejected.

## 2. TL;DR (settled 2026-11-06; the README's "What is Kapso?" and the docs index carry it)

Given an objective, initial success criteria and evaluation metrics,
Kapso investigates, runs experiments by choosing the best ideas, sometimes
provides feedback to improve the evaluation metrics and success criteria,
deploys the best solution on the desired stack, and learns from each run.
It keeps a bank of lessons, each with a credence score and a boundary on
where it applies, so the knowledge compounds rather than only accumulates
over time.

Four parts, in the order a run meets them:

1. **Smart experimentation.** Specialised modules share one data
   structure, a tree of experiments: each node records the hypothesis, the
   implementation (the branch and the technical difficulties), the budget
   used and the distance to the objective. Ideation draws on previous
   experiments and the bank of lessons to propose ideas, each saying how it
   differs from specific earlier experiments and which lessons it adopts; a
   reranker ranks them by expected return against the target, weighed
   against time and budget. The implementor builds on a branch forked from
   the parent node; a verifier checks the node against the safety rules and
   the success criteria, decides whether the score is valid and whether the
   objective is met. Success criteria and tests can evolve during a run:
   modules propose justified updates, and the verifier approves or rejects
   them.
2. **Learning loop.** Each lesson records a piece of knowledge, a credence
   score, boundary conditions on where it applies and the evidence behind
   its updates. After each run, `learn()` updates the credence and
   boundaries of the lessons used from their effect on the result, induces
   lessons from similar traces or deduces one from a single piece of
   evidence and its mechanism, merges overlapping lessons, and A/B tests the
   updated lessons by running the same build with and without them. The
   next build's ideation and implementation read each lesson's credence and
   boundaries to decide when to rely on it, rather than simply RAGing the
   knowledge.
3. **Deployment.** `deploy()` turns the best solution in the tree into
   running software on the target stack, by a shorter version of the
   experimentation loop: a session adapts a copy of the node, deploys it,
   and demonstrates health checks and a working call. Docker, Modal,
   BentoCloud and LangGraph Platform ship; a harness adds a target.
4. **Domain adaptation.** Harnesses are Kapso's domain packages: knowledge
   and workflows for an area such as predictive modelling, so a build does
   not experiment and learn from scratch. Your own domain knowledge goes in
   through the harness interface (instructions, code, guardrails and
   checks), authored with `/kapso:harness` in the plugin or `kapso harness`
   in the CLI.

## 3. Tags (settled 2026-11-06)

Smart experimentation · Learning loop · Deployment · Domain adaptation

## 4. Rules for every public sentence

- Words out: "with a number", "judge" (say evaluation), "trust",
  "applied research", "iteration engine", "long-running agents",
  "self-improving", "autonomous", "software factory", "team of agents",
  "always-on", "AI and data systems".
- Every number travels with who ran it, on what evaluator, and when.
- Learning is told as the launch post tells it (2026-11-06): the lessons
  carry a credence score and a boundary, and the knowledge compounds
  rather than only accumulates; a number for the gain travels with the
  A/B that measured it.
- The engineer stays in charge: their goal, their evaluation when they have
  one, their merge.

## 5. The product shape the one-liner implies

- **Engine**: the four parts as one loop: the tree of experiments and the
  modules that read and write it (ideation, implementation, verification,
  feedback), the isolated and versioned evaluation, the trace, deployment
  by a shorter loop of the same shape, learning into the bank of lessons.
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
  help line, the plugin's manifests and the Context7 file in one change.
  The claims lint still holds every public sentence to section 4, so the
  front pages say "decide when to rely on it" where the post says "judge
  when to trust it".
- The entrance when there is no evaluation (the engine proposes one, the
  judge checks it is fair): keep the three-case design from
  `docs/plans/kapso-plugin-user-flow.md` §4.
- The harness contract and the `load_harness` / `load_experience` /
  `build` / `deploy` / `learn` interface: design next, after the story.

## 7. Interface redesign

To be written once the story is locked.
