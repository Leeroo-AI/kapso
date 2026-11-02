# kg_graph_search retrieval eval

Measures how well `kg_graph_search` ranks the right wiki page for a query:
recall@k and MRR on labelled queries, with the LLM reranker off and on, and
with the reranker prompt's overview cap (`reranker_overview_max_chars`) at its
shipped value of 300 characters and removed. Tracked in
[#179](https://github.com/Leeroo-AI/kapso/issues/179).

## What it measures

- **Corpus** (`corpus/`): 50 synthetic wiki pages in the layout
  `parse_wiki_directory` reads, in five clusters of ML training and inference
  engineering. Each cluster holds near neighbours a careless retriever would
  confuse. Six Implementation pages have 600-1000 character overviews that open
  with a generic class description and put their distinguishing options after
  the first 300 characters; the others are 100-200 characters.
- **Labels** (`labels/queries.jsonl`): 30 plain-language queries, one object per
  line: `id`, `query`, `relevant` (page ids), and `stress` when the target page
  is one of the long-overview pages. Queries never use a technique, page-title
  or library name.
- **Arms** (`config.yaml`): `off`, `on_cap300`, `on_nocap`. For each query the
  runner fetches the candidate pool once and scores every arm on that same pool,
  so only the rerank prompt differs. An LLM rerank varies from call to call, so
  each reranking arm is sampled `rerank_runs` times on that pool.
- **Reported, overall and on the stress subset**: recall@k for each k in
  `ks`, MRR, the mean number of results returned, the share of candidate
  overviews the arm's cap cut (`clipped_share`), and the pool's own
  recall@`candidate_pool_size`, which is the most any reranker could reach.
  Reranking arms show the mean over the runs with the min-max range, and the
  report lists the queries whose target rank changed between runs.

## Running it

Run everything from the repository root.

1. Start Weaviate and Neo4j: `scripts/start_infra.sh`.
2. Start a second Neo4j for the eval. Neo4j has no namespace or database option
   in `kg_graph_search`, and `KGGraphSearch.clear()` deletes every `WikiPage`
   node, so the eval needs its own instance:

   ```bash
   docker run -d --name kapso-eval-neo4j -p 7475:7474 -p 7688:7687 \
     -e NEO4J_AUTH=neo4j/password neo4j:5.18.0
   ```

   Weaviate can be shared: the eval writes only to the collection named in
   `config.yaml` (`KapsoRetrievalEval`). The runner refuses to start if that
   collection or the Neo4j address is the shipped default.
3. Embedder and reranker. `config.yaml` uses the production models: the shipped
   `text-embedding-3-large` (needs `OPENAI_API_KEY=sk-...` in a gitignored
   `.env` in the directory you run from, which the runner loads like the CLI
   does) and the reranker of the packaged `kg_rerank` role, which is codex with
   its own login. Codex refuses the packaged `gpt-5.6-sol` on a ChatGPT-login
   account; `kapso doctor` checks the rest.

   `config.local.yaml` holds the overrides behind the numbers in the PR: a local
   `bge-m3` embedder served by Ollama (`ollama pull bge-m3`) and `gpt-6-luna` as
   the reranker. The OpenAI SDK still demands a non-empty `OPENAI_API_KEY`, so
   put any value (Ollama ignores it) in `.env`. `--config` can be repeated, and
   later files deep-merge over earlier ones.
4. Run, with the production models:

   ```bash
   python -m benchmarks.kg_retrieval.runner \
       --config benchmarks/kg_retrieval/config.yaml
   ```

   or with the local overrides that reproduce the posted numbers:

   ```bash
   python -m benchmarks.kg_retrieval.runner \
       --config benchmarks/kg_retrieval/config.yaml \
       --config benchmarks/kg_retrieval/config.local.yaml
   ```

It indexes the corpus with the repo's own indexer, scores the arms and writes
`results/results.json` and `results/results.md` (gitignored). A rerank error
aborts the run: `kg_graph_search` normally falls back to the unranked order,
which would otherwise be scored as the `off` arm.

Cost: one rerank session per query, per reranking arm, per run, so 180 sessions
for the current 30 queries, two reranking arms and `rerank_runs: 3` (about ten
minutes with `gpt-6-luna`). The `off` arm makes no LLM calls.
Remove the eval Neo4j with `docker rm -f kapso-eval-neo4j`.

## Honest limits

- The corpus is synthetic and small. Numbers compare arms against each other;
  they are directional and not an absolute quality score for a real graph.
- The posted numbers use `config.local.yaml`: a local `bge-m3` embedder, not the
  shipped `text-embedding-3-large`, and `gpt-6-luna` as the reranker, not the
  packaged `gpt-5.6-sol`. The arms rerank a shared pool, so their comparison
  stays valid, but absolute recall and the size of the reranker's gain may
  differ with the production models.
- Indexing the same corpus twice can move the `off` arm by a query: two indexings
  gave recall@1 of 0.433 and 0.400. The runs inside one invocation share one
  index and one pool, so the arms compare cleanly within it.
- The indexer embeds a page's Description. The first live run showed the six
  long pages ranking badly because their descriptions never named the page's
  topic, so each now opens with what the class is. In the historical corpus the
  long text is the description itself, so a real index embeds all of it; this
  corpus embeds only the opening, which is slightly pessimistic for retrieval
  on those pages.
- Each rerank is one LLM sample; three runs give a range, not a confidence
  interval. The reranker also returns few pages (about 2.6 on average in the
  runs recorded for pull request #180, 2026-11-01), so recall@k flattens past
  k=3; MRR and recall@1 carry the signal.
- The cap binds on only the six long pages, about 12% of the corpus (the real
  historical corpus has about 6% over 300 characters, all Implementations), so
  `on_cap300` and `on_nocap` can only differ where a target or a competitor is
  one of them. `clipped_share` shows how often that happened.
- Five stress queries are too few for a confident difference; read that subset
  as a pointer, not a verdict.
- Queries avoid technique names on purpose, while agents often name them. The
  eval measures meaning-based retrieval, not keyword matching.
- Page text was checked against library signatures and sources when written;
  those libraries change, and two of them (AutoGPTQ, AutoAWQ) are archived.

## Follow-ups

- A named-technique query subset (agents often name techniques; this set
  deliberately does not).
- A windowed rerank, if `on_nocap` clearly beats `on_cap300`: split the
  candidates into complete batches instead of clipping their overviews.
