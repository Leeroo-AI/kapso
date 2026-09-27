# CLI entry point: learn from a repository through the facade.
#
#   python -m kapso.knowledge_base.learners https://github.com/user/repo [--branch B]
#       [--extract-only] [--wiki-dir DIR] [--config FILE] [--verbose]
#
# Kapso.learn_knowledge() runs the preflight, the extraction phases and the
# merge, and builds the wiki's index when it has none, all from the config.

import argparse
import logging
import sys

from kapso.kapso import Kapso
from kapso.knowledge_base.learners.sources import Source
from kapso.knowledge_base.search.base import DEFAULT_WIKI_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kapso.knowledge_base.learners",
        description="Learn from a Git repository: extract wiki pages and merge them into the knowledge graph.",
    )
    parser.add_argument("url", help="Repository URL")
    parser.add_argument("--branch", "-b", default=None, help="Branch to learn from (default: the repository's default branch)")
    parser.add_argument("--extract-only", "-e", action="store_true", help="Extract pages without merging them into the graph")
    parser.add_argument("--wiki-dir", "-w", default=str(DEFAULT_WIKI_DIR), help=f"Wiki directory (default: {DEFAULT_WIKI_DIR})")
    parser.add_argument("--config", "-c", default=None, help="Config file (default: the packaged config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    result = Kapso(config_path=args.config).learn_knowledge(
        Source.Repo(args.url, branch=args.branch),
        wiki_dir=args.wiki_dir,
        skip_merge=args.extract_only,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
