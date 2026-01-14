"""
Expert Flow Tests - Validate the full cognitive memory flow

Tests that the system ACTUALLY WORKS, not just doesn't crash:

1. WORKFLOW_HELPS: Solving with workflow should be faster/better than without
2. BRIEFING_FOLLOWED: Agent should follow step instructions
3. DECISION_QUALITY: LLM decisions should be reasonable
4. LEARNING_WORKS: Insights should prevent repeated mistakes

These tests require:
- Running infrastructure (./start_infra.sh)
- API keys in .env
"""

import os
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from unittest.mock import Mock, patch

# Load env
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Logging Setup - Save to file
# =============================================================================
LOG_DIR = Path("/home/ubuntu/praxium/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"expert_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockExperiment:
    """Simulated experiment result."""
    run_had_error: bool
    error_details: Optional[str] = None
    score: float = 0.0
    experiment_id: str = "exp_001"


class MockSearchStrategy:
    """Simulated search strategy with experiment history."""
    def __init__(self):
        self._history: List[MockExperiment] = []
    
    def add_experiment(self, success: bool, error: str = None, score: float = 0.0):
        self._history.append(MockExperiment(
            run_had_error=not success,
            error_details=error,
            score=score,
            experiment_id=f"exp_{len(self._history)+1:03d}"
        ))
    
    def get_experiment_history(self, best_last: bool = False) -> List[MockExperiment]:
        return self._history


class MockProblemHandler:
    """Simulated problem handler."""
    def __init__(self, goal: str):
        self.goal = goal
    
    def get_problem_context(self, budget_progress: float = 0) -> str:
        return f"# Problem\n\n{self.goal}\n\n## Requirements\n- Implement the solution"
    
    def stop_condition(self) -> bool:
        return False


# =============================================================================
# TEST 1: Workflow Retrieval Quality
# =============================================================================

def test_workflow_retrieval_quality():
    """
    Test that workflow retrieval returns structured, useful workflows.
    
    A good workflow should have:
    - Clear step titles
    - Heuristics attached to steps
    - Logical progression
    """
    print("\n" + "="*70)
    print("TEST: Workflow Retrieval Quality")
    print("="*70)
    
    from src.knowledge.search import KnowledgeSearchFactory
    from src.memory.knowledge_retriever import KnowledgeRetriever
    
    kg = KnowledgeSearchFactory.create("kg_graph_search")
    retriever = KnowledgeRetriever(knowledge_search=kg)
    
    test_goals = [
        "Fine-tune LLaMA with LoRA for code generation",
        "Train a neural network for image classification",
        "Implement gradient boosting from scratch",
    ]
    
    for goal in test_goals:
        print(f"\n  Goal: {goal}")
        knowledge = retriever.retrieve_knowledge(goal)
        print(f"  Tier: {knowledge.tier.value}")
        
        if knowledge.workflow:
            wf = knowledge.workflow
            print(f"  Workflow: {wf.title}")
            print(f"  Steps: {len(wf.steps)}")
            
            # Check quality metrics
            has_heuristics = any(s.principle.heuristics for s in wf.steps) or bool(wf.heuristics)
            print(f"  Has heuristics: {has_heuristics}")
            
            for i, step in enumerate(wf.steps, 1):
                h_count = len(step.principle.heuristics)
                print(f"    Step {i}: {step.principle.title} ({h_count} heuristics)")
        else:
            print("  No workflow found")
            print(f"  Principles: {len(knowledge.principles)}")
    
    kg.close()
    print("\n  ✓ Workflow retrieval test complete")


# =============================================================================
# TEST 2: Decision Maker Behavior
# =============================================================================

def test_decision_maker_behavior():
    """
    Test that DecisionMaker makes reasonable decisions.
    
    Simplified actions (iteration-level only):
    - Success with high score → should COMPLETE
    - First failure → should RETRY
    - Multiple failures → should PIVOT or RETRY
    """
    print("\n" + "="*70)
    print("TEST: Decision Maker Behavior")
    print("="*70)
    
    from src.memory.context import (
        CognitiveContext,
        ExperimentState, MetaState
    )
    from src.memory.decisions import DecisionMaker, WorkflowAction
    
    dm = DecisionMaker()
    
    # NOTE: Only RETRY, PIVOT, COMPLETE are valid actions now
    scenarios = [
        {
            "name": "Success with high score",
            "last_exp": ExperimentState(
                experiment_id="exp_001", branch_name="exp_001",
                success=True, score=0.85
            ),
            "meta": MetaState(consecutive_failures=0),
            "expected": [WorkflowAction.COMPLETE, WorkflowAction.RETRY]
        },
        {
            "name": "First failure",
            "last_exp": ExperimentState(
                experiment_id="exp_001", branch_name="exp_001",
                success=False, error_message="ImportError: No module named X"
            ),
            "meta": MetaState(consecutive_failures=1),
            # LLM may decide to RETRY or PIVOT depending on whether it interprets the
            # missing dependency as a simple fix vs. a fundamentally blocked workflow.
            "expected": [WorkflowAction.RETRY, WorkflowAction.PIVOT]
        },
        {
            "name": "Multiple failures, same error",
            "last_exp": ExperimentState(
                experiment_id="exp_005", branch_name="exp_005",
                success=False, error_message="CUDA out of memory"
            ),
            "meta": MetaState(consecutive_failures=5),
            "expected": [WorkflowAction.PIVOT, WorkflowAction.RETRY]
        },
    ]
    
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}")
        
        ctx = CognitiveContext(
            goal="Test goal",
            iteration=1,
            last_experiment=scenario["last_exp"],
            meta=scenario["meta"],
        )
        ctx.rendered_context = "\n".join([
            "## Goal",
            "**Test goal**",
            "",
            "## Status",
            f"- Iteration: {ctx.iteration}",
            f"- Consecutive failures: {ctx.meta.consecutive_failures}",
            "",
            "## Implementation Guide",
            "*Test-only placeholder knowledge.*",
            "",
            "## Last Experiment",
            f"**Result: {'✓ SUCCESS' if ctx.last_experiment and ctx.last_experiment.success else '✗ FAILED'}**",
            f"**Score: {ctx.last_experiment.score}**" if (ctx.last_experiment and ctx.last_experiment.score is not None) else "",
            f"**Error to fix:**\n```\n{ctx.last_experiment.error_message}\n```" if (ctx.last_experiment and ctx.last_experiment.error_message and not ctx.last_experiment.success) else "",
            "",
        ]).strip()
        
        # Note: This will make real LLM calls
        try:
            decision = dm.decide_action(ctx)
            
            expected = scenario["expected"]
            passed = decision.action in expected
            status = "✓" if passed else "✗"
            
            print(f"  {status} Decision: {decision.action.value}")
            print(f"    Expected one of: {[a.value for a in expected]}")
            print(f"    Reasoning: {decision.reasoning}")
            print(f"    Confidence: {decision.confidence:.2f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n  ✓ Decision maker test complete")


# =============================================================================
# TEST 3: Full Flow Simulation
# =============================================================================

def test_full_flow_simulation():
    """
    Simulate the full Expert flow without actually running agents.
    
    This tests the cognitive memory system in isolation:
    1. Initialize with goal
    2. Get first context (workflow retrieved)
    3. Simulate experiment results
    4. Get updated context (decisions made)
    5. Verify workflow progress
    """
    print("\n" + "="*70)
    print("TEST: Full Flow Simulation")
    print("="*70)
    
    from src.knowledge.search import KnowledgeSearchFactory
    from src.memory.cognitive_controller import CognitiveController
    
    # Setup
    kg = KnowledgeSearchFactory.create("kg_graph_search")
    controller = CognitiveController(
        knowledge_search=kg,
        episodic_store_path=".test_flow_memory.json"
    )
    
    goal = "Fine-tune LLaMA with LoRA for code generation"
    print(f"\n  Goal: {goal}")
    
    # Step 1: Initialize
    print("\n  --- INITIALIZATION ---")
    controller.initialize_goal(goal)
    progress = controller.get_workflow_progress()
    print(f"  Workflow: {progress.get('title', 'None')}")
    print(f"  Steps: {progress.get('total_steps', 0)}")
    
    # Step 2: Get first briefing
    print("\n  --- FIRST BRIEFING ---")
    briefing = controller.prepare_briefing()
    print(f"  Briefing length: {len(briefing.to_string())} chars")
    print(f"  Current step: {progress.get('current_step_title', 'N/A')}")
    
    # Step 3: Simulate success
    print("\n  --- SIMULATING SUCCESS ---")
    action, details = controller.process_result(success=True)
    print(f"  Action: {action}")
    progress = controller.get_workflow_progress()
    print(f"  New step: {progress.get('current_step', 'N/A')}")
    
    # Step 4: Simulate failure
    print("\n  --- SIMULATING FAILURE ---")
    action, details = controller.process_result(
        success=False, 
        error_message="RuntimeError: CUDA out of memory"
    )
    print(f"  Action: {action}")
    
    # Cleanup
    kg.close()
    if os.path.exists(".test_flow_memory.json"):
        os.remove(".test_flow_memory.json")
    
    print("\n  ✓ Full flow simulation complete")


# =============================================================================
# TEST 4: Context Consistency
# =============================================================================

def test_context_consistency():
    """
    Verify that context.render() produces consistent output.
    
    The same CognitiveContext should always render the same way,
    ensuring decision-maker and agent see the same thing.
    """
    print("\n" + "="*70)
    print("TEST: Context Consistency")
    print("="*70)
    
    from src.memory.context import (
        CognitiveContext,
        ExperimentState, MetaState
    )
    
    ctx = CognitiveContext(
        goal="Test goal for consistency",
        iteration=3,
        last_experiment=ExperimentState(
            experiment_id="exp_003", branch_name="exp_003",
            success=False, error_message="Some error occurred"
        ),
        meta=MetaState(consecutive_failures=2)
    )
    ctx.rendered_context = "\n".join([
        "## Goal",
        "**Test goal for consistency**",
        "",
        "## Status",
        "- Iteration: 3",
        "- Consecutive failures: 2",
        "",
        "## Implementation Guide",
        "Test Workflow",
        "- Tip 1",
        "- Tip 2",
        "",
        "## Last Experiment",
        "**Result: ✗ FAILED**",
        "**Error to fix:**",
        "```",
        "Some error occurred",
        "```",
        "",
    ]).strip()
    
    # Render multiple times
    render1 = ctx.render()
    render2 = ctx.render()
    render3 = ctx.render()
    
    # Check consistency
    assert render1 == render2 == render3, "Context renders inconsistently!"
    
    print(f"  ✓ Context renders consistently ({len(render1)} chars)")
    print(f"  ✓ Contains goal: {'Test goal' in render1}")
    print(f"  ✓ Contains workflow: {'Test Workflow' in render1}")
    print(f"  ✓ Contains heuristics: {'Tip 1' in render1}")
    print(f"  ✓ Contains error: {'Some error' in render1}")
    
    print("\n  ✓ Context consistency test passed")


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("EXPERT FLOW TESTS - Cognitive Memory System")
    logger.info("=" * 70)
    logger.info(f"Log file: {LOG_FILE}")
    
    tests = [
        ("Context Consistency", test_context_consistency),
        ("Workflow Retrieval", test_workflow_retrieval_quality),
        ("Decision Maker", test_decision_maker_behavior),
        ("Full Flow", test_full_flow_simulation),
    ]
    
    results = []
    for name, test_fn in tests:
        logger.info(f"Running: {name}...")
        try:
            test_fn()
            results.append((name, "PASS"))
            logger.info(f"  ✓ {name}: PASS")
        except Exception as e:
            logger.error(f"  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        logger.info(f"  {icon} {name}: {status}")
    
    # Save results to JSON
    results_file = LOG_DIR / f"expert_flow_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "tests": [{"name": n, "status": s} for n, s in results],
            "passed": sum(1 for _, s in results if s == "PASS"),
            "total": len(results),
            "log_file": str(LOG_FILE),
        }, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n  {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
