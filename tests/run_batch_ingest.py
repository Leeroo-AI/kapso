#!/usr/bin/env python3
"""
Parallel Batch Repository Ingestion Script

Runs the repo ingestor for multiple repositories in PARALLEL with AWS Bedrock.
Each repo's output is logged to a separate file in logs/batch_ingest/

Usage:
    python tests/run_batch_ingest.py              # Run all repos in parallel
    python tests/run_batch_ingest.py --workers 3  # Limit to 3 parallel workers
    python tests/run_batch_ingest.py --limit 50   # Only process first 50 repos
"""

import argparse
import json
import os
import sys
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Path to repo URLs JSON file
REPO_URLS_FILE = "data/llm_finetuning.json"

# Default number of repos to process (top N from the JSON file)
DEFAULT_REPO_LIMIT = 2

# Ingestor settings (AWS Bedrock mode)
INGESTOR_PARAMS = {
    "use_bedrock": True,
    "aws_region": "us-east-1",
    "timeout": 1800,  # 30 minutes per phase
}

# Output directories
WIKI_DIR = "data/wikis_llm_finetuning_merge_test"
LOG_DIR = "logs/llm_finetuning_merge_test"

# Default number of parallel workers
DEFAULT_WORKERS = 5  # Run 5 repos in parallel by default


def load_repos(limit: int = DEFAULT_REPO_LIMIT) -> list:
    """
    Load repository URLs from the JSON file.
    
    Args:
        limit: Maximum number of repos to load (top N from the file)
    
    Returns:
        List of repository URLs
    """
    repo_file = Path(REPO_URLS_FILE)
    if not repo_file.exists():
        print(f"ERROR: Repo URLs file not found: {REPO_URLS_FILE}")
        sys.exit(1)
    
    with open(repo_file, "r") as f:
        all_repos = json.load(f)
    
    # Return top N repos
    return all_repos[:limit]


# ============================================================================
# Helpers
# ============================================================================

def get_repo_name(url: str) -> str:
    """Extract repo name from GitHub URL (e.g. 'vllm-project_vllm')."""
    parts = url.rstrip("/").split("/")
    return f"{parts[-2]}_{parts[-1]}"


def setup_file_logging(repo_name: str, log_dir: Path):
    """Set up file-only logging for a repo (used in worker processes)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{repo_name}_{timestamp}.log"
    
    # Configure root logger for this process to write to file
    # Clear any existing handlers first
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler only (no console to avoid mixed output)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    return log_file


def run_single_repo(args: tuple) -> dict:
    """
    Run ingestion for a single repository (worker function).
    
    Args:
        args: Tuple of (url, log_dir, wiki_dir)
    
    Returns:
        dict with: repo, success, pages, duration, error, log_file
    """
    url, log_dir, wiki_dir = args
    
    # Import here to avoid issues with multiprocessing
    from dotenv import load_dotenv
    load_dotenv()
    
    from src.knowledge_base.learners import KnowledgePipeline, Source
    
    repo_name = get_repo_name(url)
    log_file = setup_file_logging(repo_name, Path(log_dir))
    
    logger = logging.getLogger(repo_name)
    logger.info(f"=" * 70)
    logger.info(f"Starting ingestion: {url}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 70)
    
    start_time = datetime.now()
    result_info = {
        "repo": url,
        "repo_name": repo_name,
        "log_file": str(log_file),
        "success": False,
        "pages": 0,
        "duration": 0,
        "error": None,
    }
    
    try:
        # Configure pipeline
        pipeline = KnowledgePipeline(
            wiki_dir=wiki_dir,
            ingestor_params=INGESTOR_PARAMS,
        )
        
        # Run ingestion (skip_merge=True to keep wikis separate)
        result = pipeline.run(Source.Repo(url), skip_merge=True)
        
        result_info["success"] = result.success
        result_info["pages"] = result.total_pages_extracted
        
        if result.errors:
            result_info["error"] = "; ".join(result.errors)
            logger.error(f"Errors: {result.errors}")
        
        logger.info(f"Completed: {result.total_pages_extracted} pages, success={result.success}")
        
    except Exception as e:
        result_info["error"] = str(e)
        logger.exception(f"Failed with exception: {e}")
    
    duration = (datetime.now() - start_time).total_seconds()
    result_info["duration"] = duration
    logger.info(f"Duration: {duration:.1f}s")
    
    return result_info


# ============================================================================
# Main
# ============================================================================

def main():
    """Run parallel batch ingestion for all configured repos."""
    parser = argparse.ArgumentParser(description="Parallel batch repo ingestion")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--limit", "-l", type=int, default=DEFAULT_REPO_LIMIT,
                        help=f"Number of repos to process from repo_urls.json (default: {DEFAULT_REPO_LIMIT})")
    args = parser.parse_args()
    
    # Load repos from JSON file (top N based on limit)
    repos = load_repos(limit=args.limit)
    num_workers = min(args.workers, len(repos))  # Don't use more workers than repos
    
    print("\n" + "=" * 70)
    print("PARALLEL BATCH REPOSITORY INGESTION")
    print(f"Source: {REPO_URLS_FILE}")
    print(f"Repos: {len(repos)} (top {args.limit})")
    print(f"Parallel workers: {num_workers}")
    print(f"Wiki dir: {WIKI_DIR}")
    print(f"Log dir: {LOG_DIR}")
    print("=" * 70)
    
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare worker arguments
    worker_args = [(url, str(log_dir), WIKI_DIR) for url in repos]
    
    print(f"\nStarting {len(repos)} repos with {num_workers} parallel workers...")
    print("-" * 70)
    for url in repos:
        print(f"  • {get_repo_name(url)}")
    print("-" * 70 + "\n")
    
    # Run in parallel
    results = []
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_url = {
            executor.submit(run_single_repo, args): args[0] 
            for args in worker_args
        }
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            repo_name = get_repo_name(url)
            
            try:
                result = future.result()
                results.append(result)
                
                status = "✓" if result["success"] else "✗"
                print(f"{status} {repo_name}: {result['pages']} pages in {result['duration']:.1f}s")
                if result["error"]:
                    print(f"  Error: {result['error'][:100]}")
                    
            except Exception as e:
                print(f"✗ {repo_name}: Worker exception - {e}")
                results.append({
                    "repo": url,
                    "repo_name": repo_name,
                    "log_file": "N/A",
                    "success": False,
                    "pages": 0,
                    "duration": 0,
                    "error": str(e),
                })
    
    total_wall_time = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r["success"])
    total_pages = sum(r["pages"] for r in results)
    total_cpu_time = sum(r["duration"] for r in results)
    
    print(f"Successful: {successful}/{len(repos)}")
    print(f"Total pages: {total_pages}")
    print(f"Wall time: {total_wall_time/60:.1f} minutes")
    print(f"Total CPU time: {total_cpu_time/60:.1f} minutes")
    print(f"Speedup: {total_cpu_time/total_wall_time:.1f}x" if total_wall_time > 0 else "")
    print()
    
    # Sort results by repo name for consistent output
    results.sort(key=lambda r: r["repo_name"])
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['repo_name']}: {r['pages']} pages ({r['duration']:.1f}s)")
        print(f"    Log: {r['log_file']}")
    
    print("\n" + "=" * 70)
    
    # Return exit code based on success
    return 0 if successful == len(repos) else 1


if __name__ == "__main__":
    sys.exit(main())

