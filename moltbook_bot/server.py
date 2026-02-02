"""
KAPSO Brain Server

FastAPI server that exposes Kapso's optimization capabilities via HTTP.
Designed to be called by OpenClaw agents on Moltbook.

Supports two Claude Code backends:
1. AWS Bedrock (set AWS_BEARER_TOKEN_BEDROCK)
2. Anthropic API (set ANTHROPIC_API_KEY)
"""

import os
import uuid
import tempfile
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from kapso import Kapso, SolutionResult


# =============================================================================
# CONFIGURATION
# =============================================================================

app = FastAPI(
    title="KAPSO Brain Server",
    description="Knowledge-Grounded Optimization API for Moltbook Agents",
    version="1.0.0",
)

# Detect which Claude Code backend to use
# Priority: 1. Bedrock, 2. Anthropic API
def get_claude_backend() -> tuple[str, dict]:
    """
    Detect which Claude Code backend to use based on environment variables.
    
    Returns:
        tuple: (backend_name, env_dict) where backend_name is 'bedrock' or 'anthropic'
               and env_dict contains the environment variables to pass to Claude Code.
    """
    env = os.environ.copy()
    
    # Option 1: AWS Bedrock (production)
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_REGION"] = os.environ.get("AWS_REGION", "us-east-1")
        return ("bedrock", env)
    
    # Option 2: Anthropic API (development)
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Claude Code uses ANTHROPIC_API_KEY by default
        return ("anthropic", env)
    
    # Neither configured
    return ("none", env)

# Cache the backend detection at startup
CLAUDE_BACKEND, CLAUDE_ENV = get_claude_backend()

# Global Kapso instance (initialized on startup)
kapso_instance: Optional[Kapso] = None

# In-memory job store (use Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class OptimizeRequest(BaseModel):
    """Request to optimize code or solve a problem."""
    
    goal: str = Field(
        ...,
        description="The high-level objective/problem description",
        example="Optimize this Python function to run in O(n) time"
    )
    code: Optional[str] = Field(
        None,
        description="Optional code snippet to optimize"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context (e.g., 'Moltbook Post', constraints)"
    )


class OptimizeResponse(BaseModel):
    """Response from optimization."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    code: Optional[str] = Field(None, description="Optimized code (if completed)")
    cost: Optional[str] = Field(None, description="Total cost (e.g., '$0.123')")
    thought_process: Optional[str] = Field(None, description="Summary of optimization process")
    error: Optional[str] = Field(None, description="Error message if failed")


class ResearchRequest(BaseModel):
    """Request for web research."""
    
    objective: str = Field(
        ...,
        description="What to research on the public web"
    )


class ResearchResponse(BaseModel):
    """Response from research."""
    
    status: str = Field(..., description="Status: completed or failed")
    objective: str = Field(..., description="Original objective")
    report: Optional[str] = Field(None, description="Research report")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    kapso_initialized: bool
    kg_enabled: bool
    claude_backend: str  # 'bedrock', 'anthropic', or 'none'
    timestamp: str


class IntroduceRequest(BaseModel):
    """Request for introduction/questions about KAPSO."""
    
    question: str = Field(
        ...,
        description="Question about KAPSO or request for introduction",
        example="What is KAPSO and how does it work?"
    )


class IntroduceResponse(BaseModel):
    """Response from introduction endpoint."""
    
    status: str = Field(..., description="Status: completed or failed")
    question: str = Field(..., description="Original question")
    response: Optional[str] = Field(None, description="KAPSO's response")
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Kapso on server startup."""
    global kapso_instance
    
    # Optional: Load knowledge graph index
    kg_index = os.environ.get("KAPSO_KG_INDEX")
    config_path = os.environ.get("KAPSO_CONFIG_PATH")
    
    kapso_instance = Kapso(
        config_path=config_path,
        kg_index=kg_index,
    )
    
    print(f"KAPSO Brain Server initialized")
    print(f"  Knowledge Graph: {'enabled' if kg_index else 'disabled'}")
    print(f"  Claude Backend: {CLAUDE_BACKEND}")
    
    if CLAUDE_BACKEND == "none":
        print("  WARNING: No Claude Code backend configured!")
        print("  Set AWS_BEARER_TOKEN_BEDROCK or ANTHROPIC_API_KEY")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and Kapso status."""
    return HealthResponse(
        status="healthy",
        kapso_initialized=kapso_instance is not None,
        kg_enabled=kapso_instance.knowledge_search.is_enabled() if kapso_instance else False,
        claude_backend=CLAUDE_BACKEND,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_code(req: OptimizeRequest, background_tasks: BackgroundTasks):
    """
    Optimize code or solve a problem using KAPSO.
    
    This endpoint:
    1. Creates a temporary workspace with the provided code
    2. Runs Kapso.evolve() with 10 iterations to iteratively improve the solution
    3. Returns the optimized code and metrics
    """
    if kapso_instance is None:
        raise HTTPException(status_code=503, detail="KAPSO not initialized")
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "goal": req.goal,
        "created_at": datetime.now().isoformat(),
    }
    
    # Run optimization in background (10 iterations can take time)
    background_tasks.add_task(run_optimization_background, job_id, req)
    return OptimizeResponse(
        job_id=job_id,
        status="running",
        thought_process="Optimization started. Check /status/{job_id} for progress.",
    )


@app.get("/status/{job_id}", response_model=OptimizeResponse)
async def get_job_status(job_id: str):
    """Get the status of an optimization job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    return OptimizeResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        code=job.get("code"),
        cost=job.get("cost"),
        thought_process=job.get("thought_process"),
        error=job.get("error"),
    )


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """
    Perform web research using Kapso's research capability.
    
    Returns a study report with light depth.
    """
    if kapso_instance is None:
        raise HTTPException(status_code=503, detail="KAPSO not initialized")
    
    try:
        findings = kapso_instance.research(
            objective=req.objective,
            mode="study",
            depth="light",
        )
        
        return ResearchResponse(
            status="completed",
            objective=req.objective,
            report=str(findings.report) if hasattr(findings, 'report') else None,
        )
    except Exception as e:
        return ResearchResponse(
            status="failed",
            objective=req.objective,
            error=str(e),
        )


@app.post("/introduce", response_model=IntroduceResponse)
async def introduce(req: IntroduceRequest):
    """
    Answer questions about KAPSO using Claude Code.
    
    Uses the docs directory as knowledge base to introduce KAPSO
    or answer questions about how it works.
    
    Supports two backends:
    - AWS Bedrock (set AWS_BEARER_TOKEN_BEDROCK)
    - Anthropic API (set ANTHROPIC_API_KEY)
    """
    # Check if Claude backend is configured
    if CLAUDE_BACKEND == "none":
        return IntroduceResponse(
            status="failed",
            question=req.question,
            error="No Claude Code backend configured. Set AWS_BEARER_TOKEN_BEDROCK or ANTHROPIC_API_KEY.",
        )
    
    try:
        # Find docs path relative to this file or use absolute path
        script_dir = Path(__file__).parent.parent
        docs_path = str(script_dir / "docs")
        
        # Fallback to absolute path if relative doesn't exist
        if not os.path.exists(docs_path):
            docs_path = "/home/ubuntu/kapso/docs"
        
        prompt = f"""You are KAPSO, a Knowledge-grounded framework for Autonomous Program Synthesis and Optimization.

You have access to your documentation in the current directory. Use this knowledge to answer questions about yourself.

When introducing yourself:
- Explain that you are an AI-powered optimization framework
- Mention your key capabilities: evolve (iterative code optimization), research (web research), and knowledge graphs
- Be helpful and informative

Question: {req.question}

Please read the relevant documentation files and provide a clear, helpful response."""

        # Run Claude Code CLI with the docs directory as context
        # Use -p for print mode and pass prompt via stdin
        result = subprocess.run(
            [
                "claude",
                "-p",  # Print mode (non-interactive)
                "--allowedTools", "Read,Glob,Grep",
                "--add-dir", docs_path,
            ],
            input=prompt,
            cwd=docs_path,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env=CLAUDE_ENV,  # Use detected backend environment
        )
        
        if result.returncode != 0:
            # Check if there's useful output despite non-zero return
            if result.stdout.strip():
                return IntroduceResponse(
                    status="completed",
                    question=req.question,
                    response=result.stdout.strip(),
                )
            else:
                return IntroduceResponse(
                    status="failed",
                    question=req.question,
                    error=f"Claude Code failed: {result.stderr}",
                )
        
        return IntroduceResponse(
            status="completed",
            question=req.question,
            response=result.stdout.strip(),
        )
        
    except subprocess.TimeoutExpired:
        return IntroduceResponse(
            status="failed",
            question=req.question,
            error="Request timed out after 120 seconds",
        )
    except Exception as e:
        import traceback
        return IntroduceResponse(
            status="failed",
            question=req.question,
            error=f"{str(e)}\n{traceback.format_exc()}",
        )


# =============================================================================
# OPTIMIZATION LOGIC
# =============================================================================

async def run_optimization(job_id: str, req: OptimizeRequest) -> OptimizeResponse:
    """Run optimization synchronously and return result."""
    jobs[job_id]["status"] = "running"
    
    temp_dir = None
    output_dir = None
    try:
        # Create temporary workspace if code is provided
        if req.code:
            temp_dir = tempfile.mkdtemp(prefix="kapso_moltbook_")
            
            # Write code to file
            code_file = os.path.join(temp_dir, "solution.py")
            with open(code_file, "w") as f:
                f.write(req.code)
            
            # Create a simple evaluate.py
            eval_file = os.path.join(temp_dir, "evaluate.py")
            with open(eval_file, "w") as f:
                f.write('''
"""Auto-generated evaluation script."""
import sys
sys.path.insert(0, ".")
from solution import *

if __name__ == "__main__":
    # Basic syntax check - solution imports successfully
    print("SCORE: 1.0")
''')
            
            initial_repo = temp_dir
        else:
            initial_repo = None
        
        # Build the full goal with context
        full_goal = req.goal
        if req.context:
            full_goal = f"{req.goal}\n\n## Context\n{req.context}"
        
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="kapso_output_")
        
        # Run Kapso evolve (10 iterations, default coding agent)
        solution: SolutionResult = kapso_instance.evolve(
            goal=full_goal,
            initial_repo=initial_repo,
            output_path=output_dir,
            max_iterations=10,
        )
        
        # Read optimized code
        optimized_code = None
        if solution.code_path:
            solution_file = os.path.join(solution.code_path, "solution.py")
            if os.path.exists(solution_file):
                with open(solution_file, "r") as f:
                    optimized_code = f.read()
        
        # Build thought process summary
        thought_process = f"KAPSO Optimization Complete.\n"
        thought_process += f"Ran {solution.metadata.get('iterations', 0)} experiments.\n"
        thought_process += f"Stopped reason: {solution.metadata.get('stopped_reason', 'unknown')}"
        
        # Update job status
        jobs[job_id].update({
            "status": "completed",
            "code": optimized_code,
            "cost": solution.metadata.get("cost"),
            "thought_process": thought_process,
        })
        
        return OptimizeResponse(
            job_id=job_id,
            status="completed",
            code=optimized_code,
            cost=solution.metadata.get("cost"),
            thought_process=thought_process,
        )
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        jobs[job_id].update({
            "status": "failed",
            "error": error_msg,
        })
        return OptimizeResponse(
            job_id=job_id,
            status="failed",
            error=error_msg,
        )
    finally:
        # Cleanup temp directory (but keep output for debugging)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


async def run_optimization_background(job_id: str, req: OptimizeRequest):
    """Run optimization in background (for long-running jobs)."""
    await run_optimization(job_id, req)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("KAPSO_PORT", 8000))
    host = os.environ.get("KAPSO_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
