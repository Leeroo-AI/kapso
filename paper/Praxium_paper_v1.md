**Praxium: A Knowledge-Grounded AI Framework for Iterative Experiment-Driven Software Synthesis and Optimization**

**Authors**: _[TODO: add authors and affiliations]_

**Date**: 2026-01-02

---

### Abstract

We introduce **Praxium**, a modular framework that converts a natural-language goal into runnable software through an iterative loop of **proposal**, **implementation**, **execution**, and **evaluation**. Praxium targets long-horizon failures common in coding agents—lost experimental state, brittle debugging, and weak reuse of domain expertise—by integrating three tightly coupled components. First, a **git-native experimentation engine** isolates each attempt as a branch, producing reproducible artifacts and preserving provenance across iterations. Second, a **knowledge system** ingests heterogeneous sources—including repositories, benchmark artifacts (starter code and evaluation scripts), internal playbooks, and curated external resources such as documentation, scientific papers, and web search results—and organizes them into a structured representation that supports hybrid retrieval over workflows, implementations, and environment constraints. This knowledge can be provided via offline ingestion and, when enabled, via tool-based web search during builds, with retrieval provenance recorded. Third, a **cognitive memory layer** performs **tiered retrieval** and maintains an **episodic store** of reusable lessons distilled from prior experiment traces (run logs, diffs, and evaluator feedback), enabling the system to reduce repeated error modes and accelerate convergence. We evaluate Praxium on **MLE-Bench** (Kaggle-style ML competitions) and **ALE-Bench** (AtCoder heuristic optimization), which stress ML engineering under long runtimes and algorithmic optimization under strict time limits, respectively, and report end-to-end performance, cost, and ablations of the search, knowledge, and memory components.

---

### 1. Introduction

Domain experts often know what they want to build, but turning that intent into reliable, runnable software still requires repeated experimentation. In practice, successful development is an iterative process: propose an approach, implement it, run it in the real environment, inspect outcomes, and refine. This loop is especially visible in Data and AI programs, where progress depends on many measurable improvements and on careful management of code, data, and evaluation contracts. Importantly, iterations often succeed in the narrow sense of producing a working artifact, yet still fall short on quality, accuracy, robustness, or efficiency. Practical progress therefore requires repeated evaluation and targeted improvement, not only error fixing.

LLM-based coding agents reduce the cost of writing code, but they remain unreliable in long-horizon execution loops. Common failure modes include losing state across iterations, repeatedly triggering the same integration errors, and failing to reuse relevant engineering expertise even when it is available in repositories, documentation, internal playbooks, or prior attempts. In many real settings, the decisive advantage is not raw code generation, but the ability to consistently apply expert-grade best practices and high-leverage engineering workflows, including environment setup, data contracts, evaluation harnessing, debugging procedures, and performance tuning. A second practical limitation is reproducibility. Without explicit experiment isolation and provenance, it is difficult to compare approaches, debug regressions, or reuse successful solutions as building blocks.

We present **Praxium**, a framework that turns knowledge into executable software through an execution-grounded loop of iterative improvement. Given a natural-language goal and an evaluation interface, Praxium repeatedly generates candidate solution specifications, applies code changes, executes the resulting artifact under a task-defined evaluator, and uses measured outcomes to guide subsequent iterations. The system integrates three tightly coupled components. First, a **git-native experimentation engine** isolates each attempt as a branch, capturing code changes, logs, and evaluation outputs as reproducible artifacts. Second, a **knowledge system** ingests heterogeneous sources, including repositories, benchmark artifacts, internal playbooks, documentation, scientific papers, and web-derived material, and organizes them into a structured representation that supports retrieval of workflows, implementations, heuristics, and environment constraints. This knowledge is hosted in **MediaWiki**, providing a familiar interface for human review, curation, and human-in-the-loop iteration. We release a complete knowledge package consisting of a MediaWiki dump, Neo4j and Weaviate snapshots, and Docker-based deployment scripts that bring up the MediaWiki instance and all indices in a reproducible configuration. Third, a **cognitive memory layer** performs tiered retrieval and maintains an episodic store of reusable lessons distilled from experiment traces, such as run logs, diffs, and evaluator feedback, to reduce repeated error modes and accelerate convergence.

Praxium is intentionally modular. It supports pluggable evaluators, knowledge backends, and coding agents, enabling the same iteration loop to be applied across domains where progress is defined by executable outcomes and measurable objectives. We instantiate this design and evaluate it on two complementary benchmarks, MLE-Bench and ALE-Bench, and we use these instantiations to study end-to-end performance, cost, and ablations of the search, knowledge, and memory components. Beyond these benchmarks, the same interfaces generalize to additional tasks by swapping the evaluator and the knowledge sources.

#### Contributions

This paper makes the following contributions:

1. An end-to-end framework that converts heterogeneous knowledge into executable software via evaluator-grounded experimentation and iterative improvement.
2. A git-native experimentation engine that represents each attempt as an isolated, reproducible branch with explicit provenance.
3. A knowledge acquisition and representation pipeline hosted in MediaWiki that converts heterogeneous sources into a typed, workflow-oriented knowledge base usable at runtime.
4. A cognitive memory system that combines tiered retrieval with episodic learning from experiment traces to reduce repeated failures and accelerate iteration.
5. A modular architecture with pluggable evaluators and knowledge sources, demonstrated through benchmark instantiations and ablations, together with a released knowledge package containing a MediaWiki dump, Neo4j and Weaviate snapshots, and Docker-based deployment scripts for reproducing the full stack. The released knowledge base is populated from over 2,000 widely used Data and ML repositories, with selection criteria defined later.

---

### 2. Framework Overview

At a high level, Praxium exposes a user-facing `Expert` API:

- **learn**: ingest knowledge sources (e.g., repos) into the knowledge system.
- **build**: run an experiment loop to produce a working solution.
- **deploy**: adapt a solution to a target deployment strategy and return a unified runtime interface.

Internally, the build process is orchestrated by an `OrchestratorAgent` that composes four pluggable subsystems:

- **SearchStrategy**: how we generate and select candidate solutions (e.g., LLM-steered tree search).
- **ContextManager**: what context we pass into solution generation and implementation (legacy vs. cognitive).
- **KnowledgeSearch**: how we retrieve domain knowledge at runtime (legacy LLM-navigation vs. hybrid KG search).
- **CodingAgent**: which code generator applies edits (e.g., Aider, Claude Code, Gemini).

This modular design is intentional. It allows Praxium to swap out components without changing the solve loop or the benchmark harnesses.

---

### 3. Formalization (Notation and Algorithms)

This section introduces notation and formal operators for the full Praxium framework. The goal is to make the system comparable to prior work in program synthesis, agentic search, and black-box optimization by stating (i) the objects Praxium manipulates, (ii) the objective it optimizes, and (iii) the algorithms it runs.

#### 3.1 Task model and objective

We model each benchmark instance as a **task** \(t \in \mathcal{T}\) with a black-box execution-and-scoring interface. In the codebase, this is implemented by a `ProblemHandler` (and optionally an `Evaluator`), but for formalization we treat it as an environment:

- **Goal / objective**: \(g\) is a natural-language goal (optionally with constraints). Praxium may also represent this as a structured `Objective` / `Goal`.
- **Budget progress**: \(\beta_i \in [0,1]\) is the normalized progress fraction at iteration \(i\) (derived from time / iteration / cost budgets).
- **Problem context**: \(P_t(\beta)\) returns the task description string shown to the agent at a given budget progress (budget-aware in MLE).
- **Execution + scoring**: \(\mathrm{Run}_t(c)\) executes a code artifact \(c\) (a repository state in an experiment branch) and returns a result tuple

\[
r = (\text{had\_error},\; y,\; o,\; m,\; f)
\]

where \(y \in \mathbb{R}\) is the score, \(o\) is runtime output, \(m\) is an error message/details (if any), and \(f\) is optional evaluator feedback.

Many benchmarks define either a maximize or minimize objective. We normalize this with a sign \(s_t \in \{+1,-1\}\) and define the **normalized utility**:

\[
J_t(c) = s_t \cdot y_t(c)
\]

so Praxium can always be described as maximizing \(J_t\).

Finally, we define an **experiment history** after iteration \(i\) as:

\[
H_i = \{e_1,\dots,e_{i-1}\},\quad e_j = (b_j,\; u_j,\; y_j,\; \text{had\_error}_j,\; m_j,\; f_j)
\]

where \(b_j\) is the branch identifier and \(u_j\) is the natural-language solution specification used to generate code for that branch.

#### 3.2 Knowledge grounding as retrieval operators

Let \(\mathcal{G}=(\mathcal{V},\mathcal{E})\) be a typed knowledge graph where each page/node \(v \in \mathcal{V}\) has:

- an ID and title,
- a type in \(\{\text{Workflow},\text{Principle},\text{Implementation},\text{Environment},\text{Heuristic}\}\),
- overview/content text (and sometimes code snippets),
- typed edges in \(\mathcal{E}\) (e.g., `STEP`, `IMPLEMENTED_BY`, `USES_HEURISTIC`, `REQUIRES_ENV`).

Praxium uses a retrieval backend that supports at least:

- \(\mathrm{Search}(q, F)\rightarrow \langle (v_k,\mathrm{score}_k)\rangle\): semantic/graph search for a query \(q\) under filters \(F\),
- \(\mathrm{GetPage}(id)\rightarrow v\): fetch full content for a page by ID/title,
- optional graph traversal \(\mathrm{Traverse}(v)\) to collect linked knowledge.

The output of retrieval is a **knowledge packet** \(K\) (implemented as `KGKnowledge`) containing:

- tier label (Tier 1 / 2 / 3),
- confidence (retrieval score),
- provenance metadata (`query_used`, `source_pages`),
- structured workflow (if any), otherwise a list of relevant principles,
- optional error-recovery attachments (heuristics + alternative implementations).

#### 3.3 Core solve loop (Orchestrator)

At the top level, Praxium repeatedly builds context and executes experiments until a stop condition fires. In the codebase this is `OrchestratorAgent.solve()` plus a chosen `SearchStrategy`.

```text
Algorithm 1: Praxium solve loop (orchestrator-level)
Inputs:
  - task handler H_t (problem context + run + stop)
  - search strategy S (linear or tree)
  - context manager M (legacy or cognitive)
  - budgets (iterations/time/cost)

Initialize i = 0
while i < N:
  β_i = budget_progress(time, i, cost)
  if Stop_t(β_i) or β_i >= 1: break

  x_i = M.get_context(β_i)         # includes P_t(β_i), knowledge, and history
  if M.should_stop(): break        # only meaningful for cognitive mode

  S.run(x_i, β_i)                  # executes ≥1 experiments and updates its history
  i = i + 1

return best experiment in S.history (maximizing J_t)
```

This formalization makes explicit that Praxium is a **search over code artifacts** driven by repeated experiments and task-defined scoring.

#### 3.4 Implement-and-debug loop (SearchStrategy)

Both linear and tree search ultimately execute the same inner loop: create an isolated branch, implement a solution, run it, and optionally debug it for a bounded number of tries. In the codebase this is `SearchStrategy._implement_n_debug()`.

```text
Algorithm 2: Implement-and-debug loop (branch-level)
Inputs:
  - solution spec u
  - context x (problem + knowledge + history)
  - debug budget D
  - branch name b and parent branch p

session = create_experiment_session(branch=b, parent=p)
r = implement_solution(u, x, session)          # codegen + Run_t

for k in 1..D:
  if r.had_error and r.continue_debugging:
    r = debug_solution(u, x, r.error_details, session)
  else:
    break

finalize_session(session)                      # commits + push + cleanup
return r
```

#### 3.5 LLM-steered tree search (one concrete instantiation)

For completeness, we formalize the LLM-steered tree search used in MLE/ALE configurations. Let \(T=(\mathcal{N},\mathcal{A})\) be a tree of nodes \(\mathcal{N}\), where each node stores a solution spec \(u(n)\) and an optional experiment result \(e(n)\).

At each outer iteration, the strategy:

- \(\mathrm{Prune}\): optionally terminates some leaf nodes using an LLM conditioned on \((P_t(\beta_i),K_i,H_i)\).
- \(\mathrm{Expand}\): chooses nodes to expand (exploration vs. exploitation) and generates new child solution specs via an LLM ensemble.
- \(\mathrm{Select}\): selects top-\(k\) leaf nodes to execute as experiments (again using an LLM conditioned on context).

This can be viewed as a learned proposal distribution over solution specs \(u\), combined with black-box evaluation through \(\mathrm{Run}_t(\cdot)\).

#### 3.6 Cognitive memory (tiered retrieval + episodic learning + decisions)

When the cognitive context manager is enabled, Praxium augments the loop with a controller state \(C_i\) containing:

- the goal \(g\),
- knowledge packet \(K_i\),
- last experiment result \(e_{i-1}\),
- episodic insights \(Z_i\),
- meta statistics (e.g., consecutive failures).

The controller defines:

- \(\mathrm{TieredRetrieve}(g, e_{i-1})\rightarrow K_i\): Tier 1/2 retrieval, plus Tier 3 augmentation after failures,
- \(\mathrm{UpdateEpisodic}(e_{i-1})\rightarrow E\): store generalized lessons from errors/successes,
- \(\pi(C_i)\in\{\textsc{Retry},\textsc{Pivot},\textsc{Complete}\}\): an LLM policy over iteration-level actions.

```text
Algorithm 3: Cognitive controller step (per experiment)
Inputs: goal g, current knowledge K, episodic store E, last result e

if e.had_error:
  E.add(ExtractError(g, e.error_message))
  K = Tier3(g, e.error_message, K)                 # add error heuristics + alternatives
else if e.feedback is non-empty:
  E.add(ExtractSuccess(g, e.feedback))

Z = RetrieveEpisodic(E, g, e.error_message, e.feedback)
a = DecideAction(g, K, e, Z)                        # RETRY / PIVOT / COMPLETE

if a == PIVOT:
  K = TieredRetrieve(g, exclude_current_workflow=True)

return (a, K, Z)
```

The key property is that both the coding agent and the decision maker are grounded in the same rendered context, and Tier 3 explicitly records provenance via `query_used` and `source_pages`.

---

### 4. Experimentation Engine

#### 4.1 Git-native experiment isolation

Praxium models each experiment as a **git branch**. Concretely, an `ExperimentWorkspace` initializes a git repo and creates isolated `ExperimentSession`s. Each session:

- clones the workspace,
- checks out a parent branch (to inherit code),
- creates a new experiment branch,
- calls a configured coding agent to implement the candidate solution,
- runs the benchmark handler, and
- commits outputs and pushes the branch for reproducibility and downstream reuse.

This makes experiments debuggable. It also enables tree search over solutions without losing provenance.

#### 4.2 Search strategies

Praxium includes two search strategies:

- **Linear search**: generate a single solution per iteration and refine based on history.
- **LLM-steered tree search**: maintain a tree of solution candidates, expand nodes with new ideas, prune unpromising leaves, and run selected leaves in parallel.

The tree search has three LLM-mediated decisions:

- **Generation**: propose diverse candidate solutions.
- **Selection**: pick which nodes to run next based on expected improvement and diversity.
- **Pruning**: remove leaves that appear unlikely to improve given history and knowledge.

This design is practical. It allows fast exploration early and exploitation later, while staying compatible with strict evaluation loops (e.g., ALE’s fixed iteration budget).

---

### 5. Knowledge System

Praxium’s knowledge system is designed to turn unstructured codebases into structured, searchable expertise that can be reused during builds.

#### 5.1 Knowledge acquisition: Repo → typed wiki pages

The `KnowledgePipeline` coordinates a two-stage flow:

- **Stage 1 (Ingestors)**: parse a source (primarily a git repo) into a set of `WikiPage`s.
- **Stage 2 (Merger)**: merge the proposed pages into the main KG (create new pages or merge into existing ones).

For repositories, `RepoIngestor` runs a two-branch, multi-phase pipeline:

- **Branch 1 (workflow-based extraction)**: find workflows from READMEs and examples; then trace code paths to write paired Principle+Implementation pages; then extract environments and heuristics; finally run an audit step.
- **Branch 2 (orphan mining)**: identify useful files not reached by workflow tracing, triage them deterministically, ask an agent to review ambiguous cases, create pages for approved orphans, and verify coverage.

This design reduces “knowledge holes” that arise when only top-down workflows are extracted.

#### 5.2 Knowledge representation: a typed wiki schema

Praxium uses a typed schema with five page types:

- **Workflow**: an ordered recipe (steps).
- **Principle**: theory-level concept.
- **Implementation**: concrete API / code reference (often includes code snippets).
- **Environment**: dependencies and system constraints.
- **Heuristic**: tips and optimizations.

The schema is a directed structure (Workflow → Principle → Implementation) with heuristics attachable at multiple levels and environment requirements attached to implementations. This structure matters because it bounds context and makes retrieval compositional.

#### 5.3 Runtime retrieval backends

Praxium ships two knowledge-search backends:

- **Legacy: KG LLM Navigation (`kg_llm_navigation`)**  
  Stores a graph in Neo4j and uses an LLM to navigate neighbor nodes for a query. This backend can be seeded from a compact JSON graph (e.g., `benchmarks/mle/data/kg_data.json`).

- **Hybrid: KG Graph Search (`kg_graph_search`)**  
  Uses Weaviate embeddings for semantic retrieval and Neo4j for graph enrichment. It optionally applies LLM reranking. This backend is designed for typed wiki pages and supports richer workflow traversal.

These backends allow a continuum: fast legacy graph navigation for lightweight setups vs. hybrid semantic+graph retrieval when a full wiki KG is available.

---

### 6. Cognitive Memory System (Optional, Workflow-Aware)

Praxium includes an opt-in cognitive memory layer implemented as a context manager (`context_manager: cognitive`). This subsystem is designed to eliminate sharp edges in long-horizon solving: lost feedback, opaque fallbacks, and untraceable retrieval behavior.

#### 6.1 Tiered knowledge retrieval

The cognitive controller retrieves knowledge in three tiers:

- **Tier 1 (exact workflow)**: find a matching workflow via semantic search, then traverse the workflow’s ordered steps to collect linked principles, implementations, heuristics, and environments.
- **Tier 2 (relevant principles)**: if no workflow match exists, return relevant principles and their linked implementations/heuristics (without fabricating a workflow).
- **Tier 3 (error recovery)**: after a failure, generate error-targeted queries, retrieve error-specific heuristics and alternative implementations, and add them to the existing knowledge packet.

Critically, Tier 3 records provenance: it appends the exact error queries into a `query_used` field and adds the retrieved source page IDs into `source_pages`. This supports auditability and reproducibility.

#### 6.2 Episodic memory: learning from experiments

In addition to KG retrieval, the cognitive controller maintains an episodic store of reusable insights. After each experiment:

- on **failure**, an LLM generalizes the error into a reusable lesson with trigger conditions and suggested fixes;
- on **success**, an LLM extracts best-practice insights from evaluator feedback.

These insights are stored in a vector database (Weaviate) with a JSON fallback. On subsequent iterations, an LLM generates an episodic retrieval query conditioned on the current goal and the most recent error/feedback, then ranks candidate insights for applicability.

#### 6.3 LLM-based decision making

After each experiment, the controller asks an LLM to decide one of:

- **RETRY**: try again with the current knowledge/workflow,
- **PIVOT**: re-retrieve knowledge while excluding the current workflow, or
- **COMPLETE**: stop early because the goal is achieved.

The decision is grounded in the same unified context blob that is sent to the coding agent (goal, knowledge packet, last experiment state, and episodic insights). This keeps decisions interpretable.

---

### 7. Evaluation

We evaluate Praxium on two benchmarks that capture distinct real-world software-building regimes.

#### 7.1 MLE-Bench (Kaggle-style ML competitions)

**Task**: produce a Python solution that trains on provided competition data and writes a `final_submission.csv` in the expected format.

**Execution protocol** (as implemented in the benchmark handler):

- run **debug mode** first (`python main.py --debug`) with a strict runtime cap,
- validate the submission file format,
- run **full mode** (`python main.py`) with a longer runtime budget,
- grade the output submission on the competition’s hidden/private split.

**Stop condition**: stop early if the run achieves **any medal** according to the MLE-Bench grading library.

**Reported metrics**:

- **Private score**: returned by the benchmark grader.
- **Medal rate**: fraction of competitions where any medal is achieved.
- **Cost**: cumulative LLM usage cost tracked by the framework.

**Results**: _[TODO: insert your MLE-Bench results table: competitions, private scores, medals, cost]_.

#### 7.2 ALE-Bench (AtCoder heuristic contests)

**Task**: produce a C++ (`cpp23`) solution in `main.cpp` to maximize/minimize a contest-defined score under a strict per-run time limit.

**Execution protocol**:

- compile and run in the benchmark’s Docker environment,
- if the solution is accepted, run it multiple times and average the score to reduce variance,
- evaluate on a private test set to obtain final rank.

**Reported metrics**:

- **Final rank** on the private evaluation.
- **Rank percentile**: final_rank / number_of_contestants.
- **Private absolute score** (contest objective value).
- **Cost**: cumulative LLM usage cost.

**Results**: _[TODO: insert your ALE-Bench results table: problems, rank, percentile, score, cost]_.

---

### 8. Discussion

#### 8.1 Why MLE-Bench and ALE-Bench are complementary

MLE-Bench stresses data engineering, model training, file I/O contracts, and long runtimes. ALE-Bench stresses algorithmic design, tight time limits, and stochastic optimization. Together they expose different “sharp edges” in agent frameworks: brittle file outputs, poor runtime reasoning, lack of iterative improvement, and inability to reuse domain knowledge.

#### 8.2 What we expect memory to change

In Praxium, memory is not a monolith. The system supports:

- **short-horizon memory** (recent/top experiment summaries),
- **domain memory** (KG retrieval), and
- **self-memory** (episodic insights from past experiments).

We expect the cognitive memory layer to reduce repeated failure modes and shorten time-to-success on tasks with recurring errors (dependency issues, common API misuse, evaluation contract errors). We also expect provenance tracking to make benchmark runs auditable.

---

### 9. Limitations

- **External services**: the full KG graph search relies on Neo4j + Weaviate and may require infrastructure setup.
- **LLM dependence**: performance depends on the underlying LLMs used for idea generation, coding, reranking, and decisions.
- **Benchmark mismatch**: some real-world constraints (security policies, proprietary APIs, long-term maintenance) are not captured by MLE/ALE.

---

### 10. Conclusion

Praxium is a framework for building software by running experiments, using structured knowledge, and optionally learning from experience. Its core contributions are a git-native experimentation engine, a scalable knowledge acquisition and retrieval system, and a workflow-aware cognitive memory layer with tiered retrieval and episodic learning. The framework is designed to be modular and auditable. We recommend reporting benchmark results with both performance and cost, and we provide a clear execution protocol for MLE-Bench and ALE-Bench to enable reproducible evaluation.

---

### Appendix A. Reproducibility checklist

- **Configuration**: record the exact mode configs (search strategy params, coding agent model, knowledge search backend).
- **KG snapshot**: record the wiki snapshot / KG index version used for the run.
- **LLM versions**: record exact model strings (including provider prefixes) for all LLM calls.
- **Randomness**: record MLE seed and ALE run counts.

---

### Appendix B. Suggested tables to include in v1

- **Table 1**: MLE-Bench results per competition (private score, medal, cost, iterations).
- **Table 2**: ALE-Bench results per problem (rank, percentile, private score, cost, iterations).
- **Table 3**: Ablation: KG on/off; cognitive context manager on/off.


