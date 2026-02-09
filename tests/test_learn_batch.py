# Test: Learn from Multiple Repositories (Batch, No Merge)
#
# Loops over repos from repos.json and learns from each one.
# Skips the merge step (Stage 2) — only extracts wiki pages.
# Runs repos in parallel batches using ThreadPoolExecutor.
#
# Usage:
#   python tests/test_learn_batch.py
#   python tests/test_learn_batch.py --batch-size 4
#   python tests/test_learn_batch.py --batch-size 1   # sequential (same as original)
#
# Required environment variables:
#   - AWS_BEARER_TOKEN_BEDROCK or AWS credentials for Bedrock
#   - AWS_REGION (defaults to us-east-1)

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from kapso import Kapso, Source


# ─── CLI arguments ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Batch-learn repos (extract only, no merge)")
parser.add_argument(
    "--batch-size", type=int, default=3,
    help="Number of repos to process in parallel (default: 3)",
)
parser.add_argument(
    "--wiki-dir", type=str, default="data/wikis",
    help="Directory to save wiki pages (default: data/wikis)",
)
parser.add_argument(
    "--github-org", type=str, default="leeroopedia",
    help="GitHub org for workflow repos (default: leeroopedia)",
)
parser.add_argument(
    "--private", action="store_true", default=False,
    help="Create private workflow repos (default: public)",
)
parser.add_argument(
    "--log-dir", type=str, default="data/logs",
    help="Directory for log files (default: data/logs)",
)
args = parser.parse_args()


# ─── Set up logging (writes to both console and file) ────────────────────────

log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

# Timestamped log file, e.g. data/logs/learn_batch_20260207_143012.log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"learn_batch_{timestamp}.log"

# Create a logger that writes to both stdout and the log file
logger = logging.getLogger("learn_batch")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log(msg: str):
    """Helper: logs to both console and file."""
    logger.info(msg)


# ─── Load repos ──────────────────────────────────────────────────────────────

repos_file = Path(__file__).parent.parent / "repos.json"
with open(repos_file) as f:
    repos = json.load(f)

# Process repos from end to start
repos = list(reversed(repos))

log(f"Found {len(repos)} repositories to process")
log(f"Batch size: {args.batch_size}")
log(f"Merge: SKIPPED (extract only)")
log(f"Wiki dir: {args.wiki_dir}")
log(f"Log file: {log_file}")
log(f"{'='*60}\n")


# ─── Worker function ─────────────────────────────────────────────────────────

def learn_repo(repo_url: str, index: int, total: int) -> dict:
    """
    Learn from a single repo. Returns a summary dict.
    Each thread creates its own Kapso instance to avoid shared state issues.
    """
    log(f"[{index}/{total}] Starting: {repo_url}")
    repo_start = time.time()

    # Each thread gets its own Kapso instance for thread-safety
    kapso = Kapso()

    try:
        result = kapso.learn(
            Source.Repo(repo_url),
            wiki_dir=args.wiki_dir,
            skip_merge=True,            # <-- Skip merge (Stage 2)
            github_org=args.github_org,
            is_private=args.private,
        )

        repo_elapsed = time.time() - repo_start
        log(f"[{index}/{total}] Done: {repo_url} ({repo_elapsed:.1f}s)")
        log(f"  Sources processed: {result.sources_processed}")
        log(f"  Pages extracted: {result.total_pages_extracted}")
        log(f"  Success: {result.success}")

        if result.errors:
            log(f"  Errors ({len(result.errors)}):")
            for error in result.errors[:3]:
                log(f"    - {error}")
            if len(result.errors) > 3:
                log(f"    ... and {len(result.errors) - 3} more")

        return {
            "repo": repo_url,
            "success": result.success,
            "pages": result.total_pages_extracted,
            "elapsed_s": round(repo_elapsed, 1),
            "errors": result.errors,
        }

    except Exception as e:
        repo_elapsed = time.time() - repo_start
        log(f"[{index}/{total}] ERROR: {repo_url} — {e} ({repo_elapsed:.1f}s)")
        return {
            "repo": repo_url,
            "success": False,
            "pages": 0,
            "elapsed_s": round(repo_elapsed, 1),
            "errors": [str(e)],
        }


# ─── Run in batches ──────────────────────────────────────────────────────────

start_time = time.time()
results = []

# Process all repos using a thread pool with the given batch size
with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
    futures = {
        pool.submit(learn_repo, repo_url, i, len(repos)): repo_url
        for i, repo_url in enumerate(repos, 1)
    }

    # Collect results as they complete
    for future in as_completed(futures):
        repo_url = futures[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            log(f"Unexpected error for {repo_url}: {e}")
            results.append({
                "repo": repo_url,
                "success": False,
                "pages": 0,
                "errors": [str(e)],
            })

elapsed = time.time() - start_time


# ─── Final summary ───────────────────────────────────────────────────────────

total_success = sum(1 for r in results if r["success"])
total_failed = sum(1 for r in results if not r["success"])
total_pages = sum(r["pages"] for r in results)
failed_repos = [r["repo"] for r in results if not r["success"]]

log(f"\n{'='*60}")
log("FINAL SUMMARY")
log(f"{'='*60}")
log(f"  Total repos:    {len(repos)}")
log(f"  Successful:     {total_success}")
log(f"  Failed:         {total_failed}")
log(f"  Total pages:    {total_pages}")
log(f"  Elapsed time:   {elapsed:.1f}s")
log(f"  Batch size:     {args.batch_size}")
log(f"  Merge:          SKIPPED")

if failed_repos:
    log(f"\nFailed repos:")
    for repo in failed_repos:
        log(f"  - {repo}")

log(f"\nWiki pages saved to: {args.wiki_dir}")
log(f"Log file: {log_file}")

# ─── Save JSON results file (alongside the .log) ────────────────────────────

results_file = log_file.with_suffix(".json")
results_data = {
    "timestamp": timestamp,
    "batch_size": args.batch_size,
    "wiki_dir": args.wiki_dir,
    "total_repos": len(repos),
    "successful": total_success,
    "failed": total_failed,
    "total_pages": total_pages,
    "elapsed_s": round(elapsed, 1),
    "repos": results,
}
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)

log(f"Results JSON: {results_file}")
log(f"{'='*60}")
