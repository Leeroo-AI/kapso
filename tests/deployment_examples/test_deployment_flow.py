#!/usr/bin/env python3
"""
Deployment Flow Test

This script demonstrates the full deployment pipeline:
1. Takes repositories from input_repos/ (original code, no deployment files)
2. Runs Selector Agent (Claude Sonnet) to choose deployment strategy
3. Runs Adapter Agent (Claude Code with Opus 4.5) to generate deployment files
4. Outputs adapted repositories to output_repos/

Structure:
    input_repos/           <- Original repositories (2 files each)
        sentiment/         sentiment.py, requirements.txt
        image/             processor.py, requirements.txt
        embeddings/        embedder.py, requirements.txt
        qa/                qa_model.py, requirements.txt
        classifier/        classifier.py, requirements.txt (BentoML)
  
    output_repos/          <- Adapted repositories (with deployment files)
        sentiment_docker/  + Dockerfile, app.py, main.py, .dockerignore
        image_docker/      + Dockerfile, app.py, main.py, .dockerignore
        embeddings_modal/  + modal_app.py, main.py, README.md
        qa_modal/          + modal_app.py, main.py, README.md
        classifier_bentoml/ + service.py, bentofile.yaml, main.py

Usage:
    python test_deployment_flow.py

Requirements:
    - ANTHROPIC_API_KEY in .env file
    - Claude Code CLI: npm install -g @anthropic-ai/claude-code
"""

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = Path(__file__).parent
INPUT_DIR = EXAMPLES_DIR / "input_repos"
OUTPUT_DIR = EXAMPLES_DIR / "output_repos"

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not found in .env file")
    sys.exit(1)

sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.selector.agent import SelectorAgent
from src.deployment.adapter.agent import AdapterAgent


# ANSI colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}\n")


def success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def info(text):
    print(f"{Colors.BLUE}  {text}{Colors.END}")


def warn(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


# Test repositories
# Only 'name' and 'goal' are used in the deployment flow
# The selector analyzes the code + goal to choose strategy
REPOS = [
    {
        "name": "sentiment",
        "goal": "Sentiment analysis API using TextBlob",
    },
    {
        "name": "image",
        "goal": "Image processing REST API with Pillow",
    },
    {
        "name": "embeddings",
        "goal": "Text embeddings API using sentence-transformers with GPU acceleration",
    },
    {
        "name": "qa",
        "goal": "Question answering service using transformers pipeline",
    },
    {
        "name": "classifier",
        "goal": "Production ML text classification service with batching support using BentoML",
    },
]


def run_test():
    """Run the deployment flow test."""
    header("DEPLOYMENT FLOW TEST")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selector: Claude Sonnet 4")
    print(f"Adapter:  Claude Code with Opus 4.5")
    
    # Verify input repos exist
    for repo in REPOS:
        input_path = INPUT_DIR / repo["name"]
        if not input_path.exists():
            print(f"\nERROR: Input repo not found: {input_path}")
            print("Run this script from the project root.")
            sys.exit(1)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Initialize agents
    selector = SelectorAgent(use_llm=True, model="claude-sonnet-4-20250514")
    adapter = AdapterAgent(coding_agent_type="claude_code", model="claude-opus-4-5")
    
    results = []
    
    for repo in REPOS:
        input_path = INPUT_DIR / repo["name"]
        
        header(f"PROCESSING: {repo['name']}")
        
        # Show input
        input_files = [f.name for f in input_path.glob("*") if f.is_file()]
        print(f"INPUT:  input_repos/{repo['name']}/")
        for f in input_files:
            print(f"        {f}")
        print(f"\nGOAL:   {repo['goal']}")
        
        # Step 1: Selector
        print(f"\n{Colors.BOLD}[1] SELECTOR AGENT{Colors.END}")
        setting = selector.select(str(input_path), repo["goal"])
        strategy = setting.strategy
        success(f"Selected: {strategy.upper()}")
        info(f"Resources: {setting.resources}")
        info(f"Reasoning: {setting.reasoning[:80]}...")
        
        # Step 2: Copy to output
        output_name = f"{repo['name']}_{strategy}"
        output_path = OUTPUT_DIR / output_name
        shutil.copytree(input_path, output_path)
        print(f"\n{Colors.BOLD}[2] COPY TO OUTPUT{Colors.END}")
        success(f"Copied to: output_repos/{output_name}/")
        
        # Step 3: Adapter
        print(f"\n{Colors.BOLD}[3] ADAPTER AGENT{Colors.END}")
        result = adapter.adapt(
            solution_path=str(output_path),
            goal=repo["goal"],
            setting=setting,
            validate=False,
        )
        
        if result.success:
            output_files = [f.name for f in output_path.glob("*") if f.is_file()]
            new_files = [f for f in output_files if f not in input_files]
            success(f"Adapted successfully!")
            info(f"New files: {new_files}")
            info(f"Deploy command: {result.deploy_script}")
        results.append({
            "name": repo["name"],
            "strategy": strategy,
            "output": output_name,
                "new_files": new_files,
                "deploy": result.deploy_script,
        })
        else:
            warn(f"Adaptation failed: {result.error}")
            results.append({
                "name": repo["name"],
                "strategy": strategy,
                "error": result.error,
            })
    
    # Final summary
    header("SUMMARY")
    
    print("INPUT (original repos):")
    for repo in REPOS:
        files = [f.name for f in (INPUT_DIR / repo["name"]).glob("*") if f.is_file()]
        print(f"  input_repos/{repo['name']}/  →  {files}")
    
    print("\nOUTPUT (adapted repos):")
    for r in results:
        if "error" not in r:
            print(f"  output_repos/{r['output']}/")
            print(f"    Strategy:  {r['strategy'].upper()}")
            print(f"    New files: {r['new_files']}")
            print(f"    Deploy:    {r['deploy']}")
    print()
    
    header("DONE")
    print("Compare input_repos/ vs output_repos/ to see the changes.")
    print("\nTo deploy:")
    print("  DOCKER:  cd output_repos/sentiment_docker && docker build -t app . && docker run -p 8000:8000 app")
    print("  MODAL:   cd output_repos/embeddings_modal && modal deploy modal_app.py")
    print("  BENTOML: cd output_repos/classifier_bentoml && bentoml serve service:SolutionService")


if __name__ == "__main__":
    run_test()
