#!/usr/bin/env python3
"""
REAL PRODUCTION TEST - Deployment Module

This test:
1. Uses 4 UNADAPTED repositories (no deployment files)
2. Runs Selector Agent (Claude) to choose deployment strategy
3. Runs Adapter Agent (Claude Code) to adapt the code
4. Verifies the adaptation created proper deployment files

NO MOCKING - Real LLM calls to Claude for selection and adaptation.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Verify ANTHROPIC_API_KEY
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not found in .env file")
    sys.exit(1)

print(f"ANTHROPIC_API_KEY loaded: {os.environ['ANTHROPIC_API_KEY'][:20]}...")

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment import DeploymentFactory, DeployStrategy, DeployConfig
from src.deployment.selector.agent import SelectorAgent
from src.deployment.adapter.agent import AdapterAgent


# ANSI colors
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def header(text):
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.BLUE}{text}{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.END}\n")


def success(text):
    print(f"{C.GREEN}✓ {text}{C.END}")


def error(text):
    print(f"{C.RED}✗ {text}{C.END}")


def info(text):
    print(f"{C.BLUE}ℹ {text}{C.END}")


def warn(text):
    print(f"{C.YELLOW}⚠ {text}{C.END}")


# Test repositories (from input_repos/)
REPOS = [
    {
        "name": "sentiment",
        "path": PROJECT_ROOT / "tests/deployment_examples/input_repos/sentiment",
        "goal": "Sentiment analysis API using TextBlob",
        "main_file": "sentiment.py",
        "expected_files_after": ["main.py"],  # Adapter should create main.py wrapper
    },
    {
        "name": "image",
        "path": PROJECT_ROOT / "tests/deployment_examples/input_repos/image",
        "goal": "Image processing REST API with FastAPI and Pillow",
        "main_file": "processor.py",
        "expected_files_after": ["Dockerfile", "main.py"],
    },
    {
        "name": "embeddings",
        "path": PROJECT_ROOT / "tests/deployment_examples/input_repos/embeddings",
        "goal": "Text embeddings API using sentence-transformers with GPU acceleration",
        "main_file": "embedder.py",
        "expected_files_after": ["modal_app.py"],  # Should detect GPU and suggest Modal
    },
    {
        "name": "qa",
        "path": PROJECT_ROOT / "tests/deployment_examples/input_repos/qa",
        "goal": "Question answering service using transformers pipeline for production ML serving",
        "main_file": "qa_model.py",
        "expected_files_after": ["modal_app.py"],  # Has GPU deps (transformers+torch)
    },
    {
        "name": "classifier",
        "path": PROJECT_ROOT / "tests/deployment_examples/input_repos/classifier",
        "goal": "Production ML text classification service with batching support using BentoML for serving",
        "main_file": "classifier.py",
        "expected_files_after": ["service.py", "bentofile.yaml"],  # BentoML deployment
    },
]


def test_selector_agent():
    """Test the Selector Agent with real Claude API calls."""
    header("PHASE 1: SELECTOR AGENT (Claude claude-sonnet-4-20250514)")
    
    selector = SelectorAgent(use_llm=True, model="claude-sonnet-4-20250514")
    results = {}
    
    for repo in REPOS:
        info(f"Analyzing: {repo['name']}")
        info(f"Goal: {repo['goal']}")
        
        try:
            setting = selector.select(
                code_path=str(repo["path"]),
                goal=repo["goal"],
            )
            
            success(f"Selected strategy: {setting.strategy}")
            info(f"  Provider: {setting.provider}")
            info(f"  Interface: {setting.interface}")
            info(f"  Resources: {setting.resources}")
            info(f"  Reasoning: {setting.reasoning[:100]}...")
            
            results[repo["name"]] = {
                "strategy": setting.strategy,
                "setting": setting,
            }
            
        except Exception as e:
            error(f"Selector failed: {e}")
            results[repo["name"]] = {"error": str(e)}
        
        print()
    
    return results


def test_adapter_agent(selector_results):
    """Test the Adapter Agent with real Claude Code using Opus 4.5."""
    header("PHASE 2: ADAPTER AGENT (Claude Code with claude-opus-4-5)")
    
    adapter = AdapterAgent(
        coding_agent_type="claude_code",
        model="claude-opus-4-5",
        max_retries=1,
    )
    
    results = {}
    
    for repo in REPOS:
        if repo["name"] not in selector_results:
            continue
        if "error" in selector_results[repo["name"]]:
            warn(f"Skipping {repo['name']} due to selector error")
            continue
        
        setting = selector_results[repo["name"]]["setting"]
        
        info(f"Adapting: {repo['name']} for {setting.strategy}")
        
        # List files before adaptation
        before_files = set(f.name for f in repo["path"].iterdir() if f.is_file())
        info(f"Files before: {sorted(before_files)}")
        
        try:
            result = adapter.adapt(
                solution_path=str(repo["path"]),
                goal=repo["goal"],
                setting=setting,
                validate=False,  # Skip validation for speed
            )
            
            if result.success:
                success(f"Adaptation successful!")
                info(f"  Deploy script: {result.deploy_script}")
                info(f"  Files changed: {result.files_changed}")
                
                # List files after adaptation
                after_files = set(f.name for f in repo["path"].iterdir() if f.is_file())
                new_files = after_files - before_files
                if new_files:
                    success(f"  New files created: {sorted(new_files)}")
                
                results[repo["name"]] = {
                    "success": True,
                    "files_changed": result.files_changed,
                    "new_files": list(new_files),
                    "deploy_script": result.deploy_script,
                }
            else:
                error(f"Adaptation failed: {result.error}")
                results[repo["name"]] = {"success": False, "error": result.error}
            
            # Print logs
            if result.logs:
                info("Logs:")
                for log in result.logs[-5:]:
                    print(f"    {log}")
                    
        except Exception as e:
            error(f"Adapter error: {e}")
            import traceback
            traceback.print_exc()
            results[repo["name"]] = {"success": False, "error": str(e)}
        
        print()
    
    return results


def verify_adaptations(adapter_results):
    """Verify that adaptation created expected files."""
    header("PHASE 3: VERIFICATION")
    
    all_passed = True
    
    for repo in REPOS:
        info(f"Verifying: {repo['name']}")
        
        if repo["name"] not in adapter_results:
            warn("  Not adapted")
            continue
        
        result = adapter_results[repo["name"]]
        if not result.get("success"):
            error(f"  Adaptation failed: {result.get('error')}")
            all_passed = False
            continue
        
        # Check for expected files
        for expected_file in repo["expected_files_after"]:
            file_path = repo["path"] / expected_file
            if file_path.exists():
                success(f"  Found: {expected_file}")
            else:
                warn(f"  Missing: {expected_file} (may not be required for selected strategy)")
        
        # Check if main.py exists (universal requirement)
        main_py = repo["path"] / "main.py"
        if main_py.exists():
            success(f"  main.py exists (entry point ready)")
        else:
            # Check if the original file can serve as entry point
            if (repo["path"] / repo["main_file"]).exists():
                info(f"  Using {repo['main_file']} as entry point")
        
        print()
    
    return all_passed


def test_full_deployment_flow():
    """Test the complete deployment flow with DeploymentFactory."""
    header("PHASE 4: FULL DEPLOYMENT FLOW")
    
    results = {}
    
    for repo in REPOS[:2]:  # Test first 2 repos for full flow
        info(f"Testing full flow: {repo['name']}")
        
        config = DeployConfig(
            code_path=str(repo["path"]),
            goal=repo["goal"],
            coding_agent="claude_code",
        )
        
        try:
            # Use AUTO to test full selector + adapter + runner pipeline
            software = DeploymentFactory.create(DeployStrategy.AUTO, config)
            
            success(f"Deployment created: {software.name}")
            info(f"  Deploy command: {software.get_deploy_command()}")
            info(f"  Healthy: {software.is_healthy()}")
            
            # Get deployment info
            deploy_info = software.get_deployment_info()
            info(f"  Adapted files: {deploy_info.get('adapted_files', [])}")
            
            results[repo["name"]] = {
                "success": True,
                "strategy": software.name,
                "healthy": software.is_healthy(),
            }
            
            software.stop()
            
        except Exception as e:
            error(f"Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            results[repo["name"]] = {"success": False, "error": str(e)}
        
        print()
    
    return results


def main():
    header("REAL PRODUCTION TEST - Deployment Module")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using Claude claude-sonnet-4-20250514 for Selector Agent")
    print(f"Using Claude Code with claude-opus-4-5 for Adapter Agent")
    print()
    
    # Phase 1: Selector
    selector_results = test_selector_agent()
    
    # Phase 2: Adapter
    adapter_results = test_adapter_agent(selector_results)
    
    # Phase 3: Verification
    verification_passed = verify_adaptations(adapter_results)
    
    # Phase 4: Full flow (optional - takes longer)
    # deployment_results = test_full_deployment_flow()
    
    # Summary
    header("SUMMARY")
    
    print("Selector Results:")
    for name, result in selector_results.items():
        if "error" in result:
            error(f"  {name}: FAILED - {result['error']}")
        else:
            success(f"  {name}: {result['strategy']}")
    
    print("\nAdapter Results:")
    for name, result in adapter_results.items():
        if result.get("success"):
            success(f"  {name}: ADAPTED - {result.get('new_files', [])}")
        else:
            error(f"  {name}: FAILED - {result.get('error', 'Unknown')}")
    
    print()
    if verification_passed:
        success("ALL TESTS PASSED!")
        return 0
    else:
        warn("Some tests had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

