#!/usr/bin/env python3
"""
Unified Deployment Test

Complete end-to-end test for all deployment strategies:
1. Takes repos from input_repos/ (original code only)
2. Copies to output_repos/ with strategy suffix
3. Runs Selector to choose deployment strategy
4. Runs Adapter to create deployment files
5. Deploys to cloud (BentoCloud, LangGraph Platform)
6. Runs sample calls to verify deployment

Supported strategies:
- local: Local Python process
- docker: Docker container
- modal: Modal.com serverless
- bentoml: BentoCloud
- langgraph: LangGraph Platform

Usage:
    python test_unified_deployment.py              # Run all repos
    python test_unified_deployment.py sentiment    # Run specific repo
    python test_unified_deployment.py --deploy     # Include actual deployment
    python test_unified_deployment.py --clean      # Clean output_repos first

Requirements:
    - ANTHROPIC_API_KEY in .env
    - BENTO_CLOUD_API_KEY in .env (for BentoCloud)
    - LANGSMITH_API_KEY in .env (for LangGraph Platform)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = Path(__file__).parent
INPUT_DIR = EXAMPLES_DIR / "input_repos"
OUTPUT_DIR = EXAMPLES_DIR / "output_repos"

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ANSI Colors
# =============================================================================
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def header(text: str):
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{C.END}\n")


def subheader(text: str):
    print(f"\n{C.BOLD}{C.CYAN}--- {text} ---{C.END}\n")


def success(text: str):
    print(f"{C.GREEN}✓ {text}{C.END}")


def error(text: str):
    print(f"{C.RED}✗ {text}{C.END}")


def info(text: str):
    print(f"{C.BLUE}ℹ {text}{C.END}")


def warn(text: str):
    print(f"{C.YELLOW}⚠ {text}{C.END}")


# =============================================================================
# Test Configuration
# =============================================================================
@dataclass
class RepoConfig:
    """Configuration for a test repository."""
    name: str
    goal: str
    main_file: str
    expected_strategy: str  # Expected auto-selected strategy
    sample_input: Dict[str, Any]  # Sample input for testing


# All test repositories
REPOS: List[RepoConfig] = [
    RepoConfig(
        name="sentiment",
        goal="Sentiment analysis API using TextBlob",
        main_file="sentiment.py",
        expected_strategy="local",  # Simple, no heavy deps
        sample_input={"text": "I love this product!"},
    ),
    RepoConfig(
        name="image",
        goal="Image processing REST API with Pillow",
        main_file="processor.py",
        expected_strategy="docker",  # HTTP interface
        sample_input={"operation": "resize", "width": 100, "height": 100},
    ),
    RepoConfig(
        name="embeddings",
        goal="Text embeddings API using sentence-transformers with GPU",
        main_file="embedder.py",
        expected_strategy="modal",  # GPU required
        sample_input={"text": "Hello world"},
    ),
    RepoConfig(
        name="qa",
        goal="Question answering using transformers pipeline",
        main_file="qa_model.py",
        expected_strategy="modal",  # Transformers = GPU
        sample_input={"question": "What is AI?", "context": "AI is artificial intelligence."},
    ),
    RepoConfig(
        name="classifier",
        goal="Production ML text classification with batching using BentoML",
        main_file="classifier.py",
        expected_strategy="bentoml",  # Production ML serving
        sample_input={"text": "This is a great product!"},
    ),
    RepoConfig(
        name="chatbot",
        goal="Conversational AI chatbot using LangGraph with memory",
        main_file="agent.py",
        expected_strategy="langgraph",  # LangGraph agent
        sample_input={"text": "Hello! Who are you?"},
    ),
]


# =============================================================================
# Test Result Tracking
# =============================================================================
@dataclass
class TestResult:
    """Result of testing a single repository."""
    name: str
    strategy: str
    selector_ok: bool
    adapter_ok: bool
    files_created: List[str]
    deploy_ok: Optional[bool]  # None if not attempted
    run_ok: Optional[bool]  # None if not attempted
    error: Optional[str]


# =============================================================================
# Main Test Functions
# =============================================================================
def check_credentials() -> Dict[str, bool]:
    """Check which credentials are available."""
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "bentocloud": bool(os.environ.get("BENTO_CLOUD_API_KEY")),
        "langsmith": bool(os.environ.get("LANGSMITH_API_KEY")),
        "modal": bool(os.environ.get("MODAL_TOKEN_ID")),
    }


def copy_repo(repo: RepoConfig) -> Path:
    """Copy input repo to output directory."""
    input_path = INPUT_DIR / repo.name
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input repo not found: {input_path}")
    
    # We'll determine output name after selection
    return input_path


def select_strategy(repo: RepoConfig, input_path: Path):
    """Run LLM-based selector to choose deployment strategy."""
    from src.deployment.selector.agent import SelectorAgent
    
    selector = SelectorAgent(model="claude-sonnet-4-20250514")
    return selector.select(str(input_path), repo.goal)


def adapt_repo(output_path: Path, goal: str, setting) -> Dict:
    """Run adapter to create deployment files."""
    from src.deployment.adapter.agent import AdapterAgent
    
    adapter = AdapterAgent(
        coding_agent_type="claude_code",
        model="claude-opus-4-5",
        max_retries=1,
    )
    
    result = adapter.adapt(
        solution_path=str(output_path),
        goal=goal,
        setting=setting,
        validate=False,
    )
    
    return {
        "success": result.success,
        "files_changed": result.files_changed,
        "deploy_script": result.deploy_script,
        "error": result.error,
    }


def deploy_repo(output_path: Path, strategy: str) -> Dict:
    """Deploy the adapted repo to cloud."""
    result = {"success": False, "endpoint": None, "error": None}
    
    if strategy == "bentoml":
        # Deploy to BentoCloud
        try:
            import subprocess
            
            # Get deployment name
            deployment_name = output_path.name.replace("_", "-")
            
            # Deploy using the generated deploy.py script
            deploy_script = output_path / "deploy.py"
            if deploy_script.exists():
                cmd = ["python", "deploy.py"]
            else:
                cmd = ["bentoml", "deploy", ".", "--name", deployment_name]
            
            proc = subprocess.run(
                cmd,
                cwd=str(output_path),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if proc.returncode == 0:
                result["success"] = True
                # Try to extract endpoint
                for line in proc.stdout.split("\n"):
                    if "endpoint" in line.lower() or "url" in line.lower():
                        result["endpoint"] = line.strip()
            else:
                result["error"] = proc.stderr[:200]
                
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "langgraph":
        # LangGraph uses langgraph up for local, LangSmith for cloud
        # For testing, we'll build the Docker image
        try:
            import subprocess
            
            # Build the Docker image (doesn't require cloud credentials)
            cmd = ["langgraph", "build", "-t", output_path.name]
            proc = subprocess.run(
                cmd,
                cwd=str(output_path),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if proc.returncode == 0:
                result["success"] = True
                result["endpoint"] = f"Docker image: {output_path.name}"
            else:
                # Try local dev mode
                result["error"] = f"Build failed: {proc.stderr[:150]}"
                
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "modal":
        # Deploy to Modal
        try:
            import subprocess
            
            modal_app = output_path / "modal_app.py"
            if modal_app.exists():
                cmd = ["modal", "deploy", "modal_app.py"]
                proc = subprocess.run(
                    cmd,
                    cwd=str(output_path),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                if proc.returncode == 0:
                    result["success"] = True
                    # Try to extract endpoint URL
                    for line in proc.stdout.split("\n"):
                        if "https://" in line:
                            result["endpoint"] = line.strip()
                            break
                else:
                    result["error"] = proc.stderr[:200]
            else:
                result["error"] = "modal_app.py not found"
                
        except Exception as e:
            result["error"] = str(e)
    
    else:
        result["error"] = f"Deployment not supported for strategy: {strategy}"
    
    return result


def run_sample(output_path: Path, strategy: str, sample_input: Dict) -> Dict:
    """Run a sample call to the deployed service."""
    result = {"success": False, "output": None, "error": None}
    
    if strategy == "bentoml":
        # Call BentoCloud
        try:
            from src.deployment.strategies.bentoml.runner import BentoMLRunner as BentoCloudRunner
            
            deployment_name = output_path.name.replace("_", "-")
            
            # Get endpoint from bentoml
            import subprocess
            proc = subprocess.run(
                ["bentoml", "deployment", "get", deployment_name, "-o", "json"],
                capture_output=True,
                text=True,
            )
            
            if proc.returncode == 0:
                import json
                data = json.loads(proc.stdout)
                endpoints = data.get("endpoint_urls", [])
                if endpoints:
                    runner = BentoCloudRunner(
                        deployment_name=deployment_name,
                        endpoint=endpoints[0],
                        code_path=str(output_path),
                    )
                    runner._deployed = True
                    runner._endpoint = endpoints[0]
                    
                    response = runner.run(sample_input)
                    result["success"] = response.get("status") == "success"
                    result["output"] = response
                else:
                    result["error"] = "No endpoint found"
            else:
                result["error"] = "Could not get deployment info"
                
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "modal":
        # Call Modal deployed function
        try:
            import subprocess
            import json
            
            # Use modal run to call the deployed function
            modal_app = output_path / "modal_app.py"
            if modal_app.exists():
                # Try to call via HTTP if we have an endpoint, or use modal run
                cmd = [
                    "modal", "run", "modal_app.py::predict",
                    "--input", json.dumps(sample_input),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(output_path),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                
                if proc.returncode == 0:
                    result["success"] = True
                    result["output"] = proc.stdout[:200]
                else:
                    # Modal run might not work this way, try local
                    result["error"] = f"Modal run failed: {proc.stderr[:100]}"
            else:
                result["error"] = "modal_app.py not found"
                
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "langgraph":
        # LangGraph requires the agent to be running
        try:
            # Try local execution via main.py
            sys.path.insert(0, str(output_path))
            
            try:
                # Import the main predict function
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", output_path / "main.py")
                if spec and spec.loader:
                    main_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(main_module)
                    
                    if hasattr(main_module, 'predict'):
                        response = main_module.predict(sample_input)
                        result["success"] = True
                        result["output"] = response
                    else:
                        result["error"] = "No predict function in main.py"
                else:
                    result["error"] = "Could not load main.py"
            finally:
                if str(output_path) in sys.path:
                    sys.path.remove(str(output_path))
                
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "local":
        # Run locally
        try:
            sys.path.insert(0, str(output_path))
            
            try:
                # Import and run predict
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", output_path / "main.py")
                if spec and spec.loader:
                    main_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(main_module)
                    
                    response = main_module.predict(sample_input)
                    result["success"] = True
                    result["output"] = response
                else:
                    result["error"] = "Could not load main.py"
            finally:
                if str(output_path) in sys.path:
                    sys.path.remove(str(output_path))
            
        except Exception as e:
            result["error"] = str(e)
    
    elif strategy == "docker":
        # Docker requires the container to be running
        result["error"] = "Docker requires container to be running. Run: docker run -p 8000:8000 <image>"
    
    else:
        result["error"] = f"Sample run not implemented for: {strategy}"
    
    return result


def test_repo(
    repo: RepoConfig,
    do_deploy: bool = False,
    do_run: bool = False,
    skip_adapt: bool = False,
) -> TestResult:
    """Test a single repository end-to-end."""
    subheader(f"Testing: {repo.name}")
    
    result = TestResult(
        name=repo.name,
        strategy="",
        selector_ok=False,
        adapter_ok=False,
        files_created=[],
        deploy_ok=None,
        run_ok=None,
        error=None,
    )
    
    try:
        # Phase 1: Select strategy (LLM-based)
        input_path = INPUT_DIR / repo.name
        info(f"Input: {input_path}")
        
        files_before = [f.name for f in input_path.glob("*") if f.is_file()]
        info(f"Files: {files_before}")
        
        setting = select_strategy(repo, input_path)
        result.strategy = setting.strategy
        result.selector_ok = True
        
        success(f"Strategy: {setting.strategy}")
        info(f"Reasoning: {setting.reasoning[:60]}...")
        
        # Phase 2: Check if output exists (for skip_adapt)
        output_name = f"{repo.name}_{setting.strategy}"
        output_path = OUTPUT_DIR / output_name
        
        # Check if we can skip adaptation (check for main.py or langgraph.json or modal_app.py)
        has_adapted_files = (
            (output_path / "main.py").exists() or
            (output_path / "langgraph.json").exists() or
            (output_path / "modal_app.py").exists() or
            (output_path / "service.py").exists()
        )
        if skip_adapt and output_path.exists() and has_adapted_files:
            success(f"Skipping adaptation - using existing: output_repos/{output_name}/")
            result.adapter_ok = True
            result.files_created = [f.name for f in output_path.glob("*") if f.is_file() and f.name not in files_before]
        else:
            # Copy fresh and adapt
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(input_path, output_path)
            success(f"Copied to: output_repos/{output_name}/")
            
            # Phase 3: Adapt
            import time
            adapt_start = time.time()
            info("Adapting... (this may take 30-60 seconds)")
            adapt_result = adapt_repo(output_path, repo.goal, setting)
            adapt_elapsed = time.time() - adapt_start
        
            if adapt_result["success"]:
                result.adapter_ok = True
                
                # Get new files
                files_after = [f.name for f in output_path.glob("*") if f.is_file()]
                new_files = [f for f in files_after if f not in files_before]
                result.files_created = new_files
                
                success(f"Adapted in {adapt_elapsed:.1f}s! New files: {new_files}")
                info(f"Deploy: {adapt_result['deploy_script']}")
            else:
                result.error = adapt_result["error"]
                error(f"Adaptation failed: {adapt_result['error']}")
                return result
        
        # Phase 4: Deploy (optional)
        if do_deploy and setting.strategy in ["bentoml", "langgraph", "modal"]:
            info("Deploying...")
            deploy_result = deploy_repo(output_path, setting.strategy)
            result.deploy_ok = deploy_result["success"]
            
            if deploy_result["success"]:
                success(f"Deployed! Endpoint: {deploy_result.get('endpoint', 'N/A')}")
            else:
                warn(f"Deploy failed: {deploy_result.get('error', 'Unknown')}")
        
        # Phase 5: Run sample (optional)
        # For local strategy, we can always run samples
        # For cloud strategies, we need successful deployment
        can_run = (
            setting.strategy == "local" or 
            (do_deploy and result.deploy_ok) or
            (setting.strategy == "langgraph")  # LangGraph can run locally via main.py
        )
        
        if do_run and can_run:
            info(f"Running sample: {repo.sample_input}")
            run_result = run_sample(output_path, setting.strategy, repo.sample_input)
            result.run_ok = run_result["success"]
            
            if run_result["success"]:
                success(f"Sample run OK!")
                info(f"Output: {str(run_result.get('output', {}))[:100]}...")
            else:
                warn(f"Sample run failed: {run_result.get('error', 'Unknown')}")
        
    except Exception as e:
        result.error = str(e)
        error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def print_summary(results: List[TestResult], credentials: Dict[str, bool]):
    """Print test summary."""
    header("SUMMARY")
    
    # Credentials status
    print("Credentials:")
    for name, available in credentials.items():
        status = f"{C.GREEN}✓{C.END}" if available else f"{C.RED}✗{C.END}"
        print(f"  {status} {name.upper()}")
    print()
    
    # Results table
    print(f"{'Repo':<12} {'Strategy':<10} {'Selector':<10} {'Adapter':<10} {'Deploy':<10} {'Run':<10}")
    print("-" * 72)
    
    for r in results:
        selector = f"{C.GREEN}✓{C.END}" if r.selector_ok else f"{C.RED}✗{C.END}"
        adapter = f"{C.GREEN}✓{C.END}" if r.adapter_ok else f"{C.RED}✗{C.END}"
        deploy = f"{C.GREEN}✓{C.END}" if r.deploy_ok else (f"{C.RED}✗{C.END}" if r.deploy_ok is False else "-")
        run = f"{C.GREEN}✓{C.END}" if r.run_ok else (f"{C.RED}✗{C.END}" if r.run_ok is False else "-")
        
        print(f"{r.name:<12} {r.strategy:<10} {selector:<19} {adapter:<19} {deploy:<19} {run:<19}")
    
    print()
    
    # Output repos created
    print("Output repos created:")
    for r in results:
        if r.adapter_ok:
            output_name = f"{r.name}_{r.strategy}"
            print(f"  output_repos/{output_name}/")
            if r.files_created:
                print(f"    New files: {r.files_created}")
    print()
    
    # Deployment commands
    print("To deploy manually:")
    strategies = set(r.strategy for r in results if r.adapter_ok)
    if "docker" in strategies:
        print("  DOCKER: cd output_repos/<name>_docker && docker build -t app . && docker run -p 8000:8000 app")
    if "modal" in strategies:
        print("  MODAL:  cd output_repos/<name>_modal && modal deploy modal_app.py")
    if "bentoml" in strategies:
        print("  BENTO:  cd output_repos/<name>_bentoml && bentoml deploy .")
    if "langgraph" in strategies:
        print("  LANGGRAPH: cd output_repos/<name>_langgraph && langgraph deploy")
    print()


def main():
    parser = argparse.ArgumentParser(description="Unified Deployment Test")
    parser.add_argument("repos", nargs="*", help="Specific repos to test (default: all)")
    parser.add_argument("--deploy", action="store_true", help="Actually deploy to cloud")
    parser.add_argument("--run", action="store_true", help="Run sample calls after deployment")
    parser.add_argument("--clean", action="store_true", help="Clean output_repos before testing")
    parser.add_argument("--skip-adapt", action="store_true", help="Skip adaptation if output repo exists")
    args = parser.parse_args()
    
    header("UNIFIED DEPLOYMENT TEST")
    
    # Check credentials
    credentials = check_credentials()
    if not credentials["anthropic"]:
        error("ANTHROPIC_API_KEY not set (required for LLM-based selector)")
        return 1
    
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selector: LLM-based (Claude)")
    print(f"Deploy: {args.deploy}")
    print(f"Run samples: {args.run}")
    print(f"Skip adapt: {args.skip_adapt}")
    print()
    
    # Clean output if requested
    if args.clean and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        info("Cleaned output_repos/")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter repos if specified
    repos_to_test = REPOS
    if args.repos:
        repos_to_test = [r for r in REPOS if r.name in args.repos]
        if not repos_to_test:
            error(f"No matching repos found. Available: {[r.name for r in REPOS]}")
            return 1
    
    # Run tests
    results: List[TestResult] = []
    
    import time
    total_start = time.time()
    
    for i, repo in enumerate(repos_to_test, 1):
        info(f"[{i}/{len(repos_to_test)}] Processing {repo.name}...")
        repo_start = time.time()
        
        result = test_repo(
            repo,
            do_deploy=args.deploy,
            do_run=args.run,
            skip_adapt=args.skip_adapt,
        )
        results.append(result)
        
        elapsed = time.time() - repo_start
        info(f"[{i}/{len(repos_to_test)}] {repo.name} completed in {elapsed:.1f}s")
    
    total_elapsed = time.time() - total_start
    info(f"Total time: {total_elapsed:.1f}s")
    
    # Print summary
    print_summary(results, credentials)
    
    # Return status
    all_ok = all(r.selector_ok and r.adapter_ok for r in results)
    if all_ok:
        success("ALL TESTS PASSED!")
        return 0
    else:
        error("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

