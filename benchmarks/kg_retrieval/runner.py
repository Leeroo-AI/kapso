"""Retrieval-quality eval for kg_graph_search: recall@k and MRR per arm.

Indexes the synthetic corpus into a dedicated Weaviate collection (and a
dedicated Neo4j instance) with the repo's own indexer, fetches each
query's candidate pool once, then scores every arm on that same pool: the
reranker off, and the reranker on with each configured overview cap. The
reranker arms are sampled `rerank_runs` times on that same pool, since an
LLM rerank varies from call to call. All metrics are reported over every
query and over the stress subset (queries whose target page has an overview
long enough for the cap to cut), as a mean with its range across the runs.

Run from the repository root; --config may be repeated, later files
deep-merging over earlier ones:
    python -m benchmarks.kg_retrieval.runner \
        --config benchmarks/kg_retrieval/config.yaml
"""

import argparse
import copy
import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from benchmarks.kg_retrieval.labels import load_labels
from benchmarks.kg_retrieval.metrics import aggregate
from kapso.core.config import RUNNER_KEY, deep_merge, load_config, read_yaml
from kapso.knowledge_base.search import (
    KGIndexInput,
    KGSearchFilters,
    KnowledgeSearchFactory,
)
from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory

SEARCH_BACKEND = "kg_graph_search"

# Provider keys (the embedder's OPENAI_API_KEY) live in a gitignored .env, as
# for the CLI and the other benchmark runners; no setting is read from it.
load_dotenv(find_dotenv(usecwd=True))


class RerankFailed(RuntimeError):
    """A rerank call failed; scoring it would silently measure no reranker."""


class FailLoudReranker:
    """Wraps the rerank LLM backend; remembers if its last call finished.

    KGGraphSearch._rerank_results swallows any error from the backend and
    returns the original order, which would score a failed run as the
    "reranker off" arm. The wrapper cannot stop that swallow, but it leaves
    a flag the caller checks right after the rerank.
    """

    def __init__(self, backend):
        self.backend = backend
        self.last_call_completed = False

    def llm_completion_with_system_prompt(self, **kwargs):
        self.last_call_completed = False
        response = self.backend.llm_completion_with_system_prompt(**kwargs)
        self.last_call_completed = True
        return response


class KGRetriever:
    """The retrieval operations the arm loop needs, over a KGGraphSearch."""

    def __init__(self, search):
        self.search = search
        self.reranker = FailLoudReranker(search._llm_backend)
        search._llm_backend = self.reranker

    def top(self, query, top_k):
        """What the product returns with the reranker off."""
        filters = KGSearchFilters(top_k=top_k, include_content=False)
        return self.search.search(
            query, filters=filters, use_llm_reranker=False
        ).results

    def pool(self, query, pool_size):
        """The candidates the reranker would be shown."""
        filters = KGSearchFilters(top_k=pool_size, include_content=False)
        return self.search.search(
            query, filters=filters, use_llm_reranker=False
        ).results

    def rerank(self, query, pool, top_k, overview_cap):
        """Rerank a copy of the pool with this arm's cap on the prompt."""
        if not pool:
            return []
        self.search.reranker_overview_max_chars = overview_cap
        candidates = copy.deepcopy(pool)
        results = self.search._rerank_results(query, candidates, top_k)
        if not self.reranker.last_call_completed:
            raise RerankFailed(f"rerank call failed for query {query!r}")
        return results


def check_isolated(search_params):
    """Refuse to run against the stores a real knowledge graph lives in."""
    defaults = KnowledgeSearchFactory.get_defaults(SEARCH_BACKEND)
    for key in ("weaviate_collection", "neo4j_uri"):
        if search_params[key] == defaults[key]:
            raise ValueError(
                f"{key} is {search_params[key]!r}, the shipped default; the "
                "eval needs its own so it never touches a real graph"
            )


def clipped_counts(pool, overview_cap):
    """Pool overviews the prompt cap cuts, and the pool size."""
    if overview_cap is None:
        return 0, len(pool)
    return sum(len(item.overview) > overview_cap for item in pool), len(pool)


def summarize(indices, rankings, queries, ks, extras):
    """Metrics for the queries at `indices`; extras carry per-query counts."""
    relevants = [queries[i].relevant for i in indices]
    scores = aggregate([rankings[i] for i in indices], relevants, ks)
    counts = [extras["returned"][i] for i in indices]
    scores["mean_results"] = sum(counts) / len(counts)
    if extras["clipped"] is not None:
        clipped = sum(extras["clipped"][i][0] for i in indices)
        total = sum(extras["clipped"][i][1] for i in indices)
        scores["clipped_share"] = clipped / total
    return scores


def target_rank(ranking, relevant):
    """1-based position of the first relevant page, None when absent."""
    for position, page_id in enumerate(ranking, start=1):
        if page_id in relevant:
            return position
    return None


def spread(per_run):
    """Per-metric mean, min and max over the runs' metric dicts."""
    stats = {"runs": per_run}
    for name, reduce in (("mean", None), ("min", min), ("max", max)):
        stats[name] = {}
        for metric in per_run[0]:
            values = [run[metric] for run in per_run]
            stats[name][metric] = (
                sum(values) / len(values) if reduce is None else reduce(values)
            )
    return stats


def empty_run():
    return {"rankings": [], "returned": [], "clipped": []}


def ranks_by_query(queries, runs):
    """Each query's target rank in every run."""
    ranks = {}
    for i, query in enumerate(queries):
        rankings = [run["rankings"][i] for run in runs]
        ranks[query.id] = [target_rank(r, query.relevant) for r in rankings]
    return ranks


def evaluate(queries, arms, retriever, ks, top_k, pool_size, rerank_runs):
    """Score every arm on one candidate pool per query.

    A reranking arm is sampled `rerank_runs` times on that pool; the off
    arm is deterministic and runs once.
    """
    pool_rankings = []
    arm_runs = {}
    for arm in arms:
        count = rerank_runs if arm["rerank"] else 1
        arm_runs[arm["name"]] = [empty_run() for _ in range(count)]
    for query in queries:
        pool = retriever.pool(query.query, pool_size)
        pool_rankings.append([item.id for item in pool])
        for arm in arms:
            for run in arm_runs[arm["name"]]:
                if arm["rerank"]:
                    cap = arm["reranker_overview_max_chars"]
                    results = retriever.rerank(query.query, pool, top_k, cap)
                    run["clipped"].append(clipped_counts(pool, cap))
                else:
                    results = retriever.top(query.query, top_k)
                run["rankings"].append([item.id for item in results])
                run["returned"].append(len(results))
    subsets = {
        "overall": list(range(len(queries))),
        "stress": [i for i, query in enumerate(queries) if query.stress],
    }
    report = {
        "rerank_runs": rerank_runs,
        "queries": {name: len(idx) for name, idx in subsets.items()},
    }
    report["pool_ceiling"] = {
        name: aggregate(
            [pool_rankings[i] for i in idx],
            [queries[i].relevant for i in idx],
            [pool_size],
        )
        for name, idx in subsets.items()
    }
    report["arms"] = {}
    report["target_ranks"] = {}
    for arm in arms:
        runs = arm_runs[arm["name"]]
        report["arms"][arm["name"]] = {}
        for name, idx in subsets.items():
            per_run = []
            for run in runs:
                extras = {
                    "returned": run["returned"],
                    "clipped": run["clipped"] if arm["rerank"] else None,
                }
                rankings = run["rankings"]
                per_run.append(summarize(idx, rankings, queries, ks, extras))
            report["arms"][arm["name"]][name] = spread(per_run)
        report["target_ranks"][arm["name"]] = ranks_by_query(queries, runs)
    report["unstable"] = sorted(
        {
            query_id
            for ranks in report["target_ranks"].values()
            for query_id, per_run in ranks.items()
            if len(set(per_run)) > 1
        }
    )
    return report


def cell(stats, metric):
    """A mean, with the min-max range when the runs disagree."""
    low, high = stats["min"][metric], stats["max"][metric]
    text = f"{stats['mean'][metric]:.3f}"
    return text if low == high else f"{text} ({low:.3f}-{high:.3f})"


def render_markdown(report, ks, pool_size):
    """One table per subset, one row per arm."""
    columns = [f"recall@{k}" for k in ks] + ["mrr", "mean_results"]
    lines = [f"Reranking arms sampled {report['rerank_runs']} times.", ""]
    for subset, count in report["queries"].items():
        ceiling = report["pool_ceiling"][subset][f"recall@{pool_size}"]
        lines += [
            f"## {subset} ({count} queries)",
            "",
            f"Candidate pool ceiling, recall@{pool_size}: {ceiling:.3f}",
            "",
            "| arm | " + " | ".join(columns) + " | clipped_share |",
            "|---|" + "---|" * (len(columns) + 1),
        ]
        for arm, subsets in report["arms"].items():
            stats = subsets[subset]
            cells = [cell(stats, column) for column in columns]
            if "clipped_share" in stats["mean"]:
                cells.append(cell(stats, "clipped_share"))
            else:
                cells.append("n/a")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Target rank by run, queries that changed")
    lines.append("")
    for query_id in report["unstable"]:
        arm_ranks = report["target_ranks"].items()
        shown = ", ".join(f"{a}: {r[query_id]}" for a, r in arm_ranks)
        lines.append(f"- {query_id}: {shown}")
    if not report["unstable"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def validate_arms(arms):
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a non-empty list")
    for arm in arms:
        if not isinstance(arm.get("name"), str) or not isinstance(
            arm.get("rerank"), bool
        ):
            raise ValueError(f"arm needs a name and a boolean rerank: {arm!r}")
        if arm["rerank"] and "reranker_overview_max_chars" not in arm:
            raise ValueError(
                f"arm {arm['name']!r} reranks, so it must set "
                "reranker_overview_max_chars (null for no cap)"
            )


def load_layers(config_paths):
    """The override files, later ones deep-merged over earlier ones; each
    is checked as a kapso override, the eval's own settings under
    `benchmark` (the runners' key)."""
    merged = {}
    for path in config_paths:
        load_config(path)
        merged = deep_merge(merged, read_yaml(path))
    return merged


def build_search(layers, eval_config):
    """A KGGraphSearch with the reranker on, so a call can switch it off;
    the engine's blocks come from the layered files over the packaged
    config, the way a build's one override does."""
    config = deep_merge(
        load_config(),
        {key: value for key, value in layers.items() if key != RUNNER_KEY},
    )
    params = dict(eval_config["search"])
    params.update(
        include_connected_pages=False,
        use_llm_reranker=True,
        models=config["models"],
        retry=config["retry"],
        inference=config["inference"],
    )
    check_isolated(params)
    return KnowledgeSearchFactory.create(SEARCH_BACKEND, params=params)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="config.yaml; repeat to layer overrides over it",
    )
    args = parser.parse_args(argv)

    layers = load_layers(args.config)
    eval_config = layers[RUNNER_KEY]
    validate_arms(eval_config["arms"])
    ks, top_k, pool_size = (
        eval_config["ks"],
        eval_config["top_k"],
        eval_config["candidate_pool_size"],
    )
    if max(ks) > top_k or pool_size < top_k:
        raise ValueError("need max(ks) <= top_k <= candidate_pool_size")
    rerank_runs = eval_config["rerank_runs"]
    if not isinstance(rerank_runs, int) or rerank_runs < 1:
        raise ValueError(f"bad rerank_runs {rerank_runs!r}")

    corpus_dir = Path(eval_config["corpus_dir"])
    page_ids = {page.id for page in parse_wiki_directory(corpus_dir)}
    queries = load_labels(eval_config["labels_path"], page_ids)
    if not any(query.stress for query in queries):
        raise ValueError("no stress queries in the labels")

    search = build_search(layers, eval_config)
    if eval_config["index_corpus"]:
        search.index(KGIndexInput(wiki_dir=corpus_dir))
    report = evaluate(
        queries,
        eval_config["arms"],
        KGRetriever(search),
        ks,
        top_k,
        pool_size,
        rerank_runs,
    )
    report["config"] = eval_config
    search.close()

    results_dir = Path(eval_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(report, indent=2))
    table = render_markdown(report, ks, pool_size)
    (results_dir / "results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
