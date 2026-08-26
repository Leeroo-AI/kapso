# Knowledge Base Module
#
# Handles knowledge storage, learning (ingestion), and search (retrieval).
#
# Submodules:
#   - types: Unified source types (Source.Repo, Source.Idea, etc.)
#   - search: Unified search backends (KG Graph Search, etc.)
#   - learners: Modular knowledge ingestion pipeline
#   - wiki_structure: Wiki page templates and definitions (markdown, loaded
#     by path from the ingestors)
#
# Import from the submodules directly — this package re-exports nothing.
# (The old eager re-exports + a lazy-loading table had zero importers and
# put search imports on every consumer's cold-start path; stale-code
# audit 2026-08-26.)
