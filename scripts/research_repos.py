#!/usr/bin/env python3
"""
Repository Research Script with OpenAI Web Search

Uses OpenAI's web search tool to deeply research GitHub repositories and extract:
- Short description
- Number of stars
- List of relevant tags (e.g., machine learning, data engineering, etc.)
- Relevance score (0-10) for data/ML domains

Usage:
    python scripts/research_repos.py                    # Process all repos
    python scripts/research_repos.py --limit 10        # Process first 10 repos
    python scripts/research_repos.py --output results.csv   # Custom output file
    python scripts/research_repos.py --workers 3       # Limit concurrent requests
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================

# Path to repo URLs JSON file
REPO_URLS_FILE = "data/repo_urls.json"

# Default output file for results
DEFAULT_OUTPUT_FILE = "data/repo_research_results.csv"

# Default number of repos to process (None = all)
DEFAULT_REPO_LIMIT = None

# Default number of concurrent workers (to respect rate limits)
DEFAULT_WORKERS = 5

# OpenAI model to use (must support web search)
OPENAI_MODEL = "gpt-5.2"

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def get_repo_name(url: str) -> str:
    """
    Extract repo name from GitHub URL.
    
    Example: 'https://github.com/owner/repo' -> 'owner/repo'
    """
    parts = url.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def load_repos(limit: Optional[int] = None) -> list:
    """
    Load repository URLs from the JSON file.
    
    Args:
        limit: Maximum number of repos to load (None = all)
    
    Returns:
        List of repository URLs
    """
    repo_file = Path(REPO_URLS_FILE)
    if not repo_file.exists():
        logger.error(f"Repo URLs file not found: {REPO_URLS_FILE}")
        sys.exit(1)
    
    with open(repo_file, "r") as f:
        all_repos = json.load(f)
    
    # Return all repos or top N based on limit
    if limit is not None:
        return all_repos[:limit]
    return all_repos


def build_research_prompt(url: str) -> str:
    """
    Build the prompt for researching a repository.
    
    The prompt instructs the model to use web search to gather
    comprehensive information about the repository.
    """
    return f"""Research the GitHub repository at: {url}

Please search the web to find accurate and up-to-date information about this repository.

Provide your findings in the following JSON format ONLY (no other text):
{{
    "url": "{url}",
    "name": "<repository name>",
    "description": "<concise description of what the repo does, 1-2 sentences>",
    "stars": <number of GitHub stars as integer, or null if unknown>,
    "tags": ["<tag1>", "<tag2>", ...],
    "ml_data_score": <score from 0-10>,
    "score_reasoning": "<detailed justification for the score>"
}}

Guidelines for each field:
- description: Write a clear, concise summary of the repository's purpose and main features.
- stars: The current number of GitHub stars. Use null if you cannot determine this.
- tags: Generate relevant topic tags that describe the repository's domain and technologies.
  Examples: machine learning, data engineering, web development, DevOps, NLP, computer vision, etc.
  Be specific and include 3-8 tags that accurately represent the repository.
- ml_data_score: Rate from 0-10 how valuable this repo is for data science and machine learning.
  Consider these factors when scoring:
  * Primary purpose: Is it built specifically for ML/data tasks?
  * Direct applicability: Can data scientists/ML engineers use it directly in their workflows?
  * Community adoption: Is it widely used in the ML/data community?
  * Educational value: Does it help learn ML/data concepts or techniques?
  * Integration potential: Does it integrate with ML/data tools and frameworks?
  
  Scoring scale:
  - 0-1: Completely unrelated to ML/data (e.g., gaming, mobile apps, general web dev)
  - 2-3: Tangentially related (e.g., general utilities that could be used anywhere)
  - 4-5: Indirect relevance (e.g., databases, APIs, infrastructure tools sometimes used in ML)
  - 6-7: Moderately relevant (e.g., data visualization, data processing, statistics libraries)
  - 8-9: Highly relevant (e.g., ML frameworks, data pipelines, model training tools, MLOps)
  - 10: Core ML/data tool with major industry adoption (e.g., PyTorch, TensorFlow, Pandas, Spark)

- score_reasoning: Provide a detailed justification (2-4 sentences) explaining:
  * What the repository does and its primary use case
  * How it relates to ML/data workflows (or why it doesn't)
  * Why you assigned this specific score based on the factors above

Return ONLY the JSON object, no markdown formatting or additional text."""


def parse_research_response(response_text: str, url: str) -> dict:
    """
    Parse the model's response into a structured dictionary.
    
    Handles cases where the response might have markdown formatting
    or extra text around the JSON.
    """
    # Try to extract JSON from the response
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        result = json.loads(text)
        # Ensure required fields exist
        result.setdefault("url", url)
        result.setdefault("name", get_repo_name(url))
        result.setdefault("description", "Unable to retrieve description")
        result.setdefault("stars", None)
        result.setdefault("tags", [])
        result.setdefault("ml_data_score", 0)
        result.setdefault("score_reasoning", "")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response for {url}: {e}")
        # Return a default structure with error info
        return {
            "url": url,
            "name": get_repo_name(url),
            "description": "Failed to parse response",
            "stars": None,
            "tags": [],
            "ml_data_score": 0,
            "score_reasoning": "",
            "parse_error": str(e),
            "raw_response": response_text[:500]  # First 500 chars for debugging
        }


# ============================================================================
# OpenAI Web Search Research
# ============================================================================

def research_repository(client: OpenAI, url: str) -> dict:
    """
    Research a single repository using OpenAI's web search tool.
    
    Args:
        client: OpenAI client instance
        url: GitHub repository URL
    
    Returns:
        Dictionary with research results
    """
    repo_name = get_repo_name(url)
    logger.info(f"Researching: {repo_name}")
    
    prompt = build_research_prompt(url)
    
    try:
        # Call OpenAI API with web search tool enabled
        # Using the Responses API with web_search tool
        response = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            input=prompt
        )
        
        # Extract the output text from the response
        output_text = response.output_text
        
        # Parse the response into structured data
        result = parse_research_response(output_text, url)
        result["research_status"] = "success"
        
        logger.info(f"✓ {repo_name}: {result.get('stars', 'N/A')} stars, score={result.get('ml_data_score', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ {repo_name}: {str(e)}")
        return {
            "url": url,
            "name": repo_name,
            "description": "Research failed",
            "stars": None,
            "tags": [],
            "ml_data_score": 0,
            "score_reasoning": "",
            "research_status": "error",
            "error": str(e)
        }


async def research_repository_async(client: OpenAI, url: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Async wrapper for research_repository with rate limiting via semaphore.
    """
    async with semaphore:
        # Run the sync function in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, research_repository, client, url)


async def research_all_repos_async(client: OpenAI, urls: list, max_workers: int) -> list:
    """
    Research all repositories concurrently with a limit on concurrent requests.
    
    Args:
        client: OpenAI client instance
        urls: List of GitHub repository URLs
        max_workers: Maximum number of concurrent requests
    
    Returns:
        List of research results
    """
    semaphore = asyncio.Semaphore(max_workers)
    
    tasks = [
        research_repository_async(client, url, semaphore)
        for url in urls
    ]
    
    results = await asyncio.gather(*tasks)
    return list(results)


def research_all_repos(client: OpenAI, urls: list, max_workers: int = DEFAULT_WORKERS) -> list:
    """
    Research all repositories with concurrent requests.
    
    Args:
        client: OpenAI client instance
        urls: List of GitHub repository URLs
        max_workers: Maximum number of concurrent requests
    
    Returns:
        List of research results
    """
    return asyncio.run(research_all_repos_async(client, urls, max_workers))


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point for the repository research script."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Research GitHub repositories using OpenAI web search"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_REPO_LIMIT,
        help="Number of repos to process (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Max concurrent requests (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already researched repos"
    )
    args = parser.parse_args()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load repository URLs
    repos = load_repos(limit=args.limit)
    
    # Handle resume functionality
    existing_results = {}
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert tags back from pipe-separated string to list
                row["tags"] = row["tags"].split("|") if row["tags"] else []
                # Convert numeric fields
                row["stars"] = int(row["stars"]) if row["stars"] and row["stars"] != "None" else None
                row["ml_data_score"] = int(row["ml_data_score"]) if row["ml_data_score"] else 0
                existing_results[row["url"]] = row
        logger.info(f"Resuming: found {len(existing_results)} existing results")
        # Filter out already processed repos
        repos_to_process = [url for url in repos if url not in existing_results]
        logger.info(f"Remaining repos to process: {len(repos_to_process)}")
    else:
        repos_to_process = repos
    
    # Print header
    print("\n" + "=" * 70)
    print("REPOSITORY RESEARCH WITH OPENAI WEB SEARCH")
    print("=" * 70)
    print(f"Source file: {REPO_URLS_FILE}")
    print(f"Total repos in file: {len(repos)}")
    print(f"Repos to process: {len(repos_to_process)}")
    print(f"Max concurrent requests: {args.workers}")
    print(f"Output file: {args.output}")
    print(f"Model: {OPENAI_MODEL}")
    print("=" * 70 + "\n")
    
    if not repos_to_process:
        logger.info("No repos to process. All repos already researched.")
        sys.exit(0)
    
    # Research all repositories
    start_time = datetime.now()
    new_results = research_all_repos(client, repos_to_process, args.workers)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Merge with existing results if resuming
    if existing_results:
        # Start with existing results
        all_results = list(existing_results.values())
        # Add new results
        all_results.extend(new_results)
    else:
        all_results = new_results
    
    # Sort by URL for consistent ordering
    all_results.sort(key=lambda r: r["url"])
    
    # Calculate statistics
    successful = sum(1 for r in all_results if r.get("research_status") == "success")
    avg_score = sum(r.get("ml_data_score", 0) for r in all_results) / len(all_results) if all_results else 0
    
    # Save results to CSV file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    csv_columns = ["url", "name", "description", "stars", "tags", "ml_data_score", "score_reasoning", "research_status"]
    
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        
        for result in all_results:
            # Convert tags list to pipe-separated string for CSV
            row = result.copy()
            row["tags"] = "|".join(result.get("tags", []))
            writer.writerow(row)
    
    logger.info(f"Results saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {len(repos_to_process)}")
    print(f"Successful: {successful}/{len(all_results)}")
    print(f"Average ML/Data Score: {avg_score:.2f}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # Show top 10 repos by ML/data score
    print("\nTop 10 repos by ML/Data relevance score:")
    print("-" * 70)
    sorted_results = sorted(all_results, key=lambda r: r.get("ml_data_score", 0), reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        score = r.get("ml_data_score", 0)
        stars = r.get("stars", "N/A")
        name = r.get("name", "Unknown")
        print(f"{i:2}. [{score}/10] {name} ({stars} stars)")
        print(f"    {r.get('description', 'No description')[:70]}...")
    
    print("\n" + "=" * 70)
    
    return 0 if successful == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
