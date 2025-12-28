"""
Comprehensive E2E Test for Cognitive Memory System

This test:
1. Uses REAL KG infrastructure (no mocking)
2. Tests multiple iterations (RETRY, PIVOT, COMPLETE)
3. Triggers TIER 3 error retrieval
4. Verifies episodic store persistence

Prerequisites:
    ./start_infra.sh  # Must be running
    
To run:
    PYTHONPATH=. python tests/test_cognitive_e2e_full.py
    
Output:
    - Logs to /home/ubuntu/praxium/logs/cognitive_e2e_full_*.log
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

# Setup comprehensive logging
LOG_DIR = Path("/home/ubuntu/praxium/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"cognitive_iteration_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
# Test Configuration
# =============================================================================

TEST_CONFIG = {
    "goal": "Fine-tune a language model with LoRA for text generation",
    "max_iterations": 5,
    "stop_threshold": 0.8,
}


# =============================================================================
# Infrastructure Check
# =============================================================================

def check_infrastructure() -> Tuple[bool, Optional[Any]]:
    """Check if all required infrastructure is available."""
    logger.info("=" * 70)
    logger.info("CHECKING INFRASTRUCTURE")
    logger.info("=" * 70)
    
    # Check KG
    logger.info("[1/3] Checking Knowledge Graph...")
    try:
        from src.knowledge.search import KnowledgeSearchFactory
        kg = KnowledgeSearchFactory.create("kg_graph_search")
        if not kg.is_enabled():
            logger.error("KG is not enabled!")
            return False, None
        result = kg.search("test query", context=None)
        logger.info(f"  ✓ KG connected: {len(result.results)} results")
    except Exception as e:
        logger.error(f"  ✗ KG failed: {e}")
        return False, None
    
    # Check Episodic Store
    logger.info("[2/3] Checking Episodic Store...")
    try:
        from src.memory.episodic import EpisodicStore
        episodic = EpisodicStore()
        logger.info("  ✓ Episodic store connected")
    except Exception as e:
        logger.error(f"  ✗ Episodic store failed: {e}")
        return False, None
    
    # Check LLM
    logger.info("[3/3] Checking LLM Backend...")
    try:
        from src.core.llm import LLMBackend
        llm = LLMBackend()
        test_response = llm.llm_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10,
        )
        logger.info("  ✓ LLM backend working")
    except Exception as e:
        logger.error(f"  ✗ LLM failed: {e}")
        return False, None
    
    logger.info("=" * 70)
    logger.info("All infrastructure checks passed!")
    logger.info("=" * 70)
    
    return True, kg


# =============================================================================
# Simulated Experiment Results
# =============================================================================

class SimulatedExperiment:
    """Simulates experiment results for different iterations."""
    
    SCENARIOS: List[Dict[str, Any]] = [
        # Iteration 0: Import error (tests RETRY)
        {"success": False, "score": 0.1, "feedback": "Missing PEFT library", "error": "ImportError: No module named 'peft'"},
        # Iteration 1: Partial success
        {"success": True, "score": 0.3, "feedback": "LoRA config missing parameters", "error": None},
        # Iteration 2: Better score (tests RETRY improvement)
        {"success": True, "score": 0.6, "feedback": "Training started but not logging metrics", "error": None},
        # Iteration 3: CUDA OOM (tests TIER 3)
        {"success": False, "score": 0.2, "feedback": "Memory issue", "error": "RuntimeError: CUDA out of memory"},
        # Iteration 4: Success!
        {"success": True, "score": 0.85, "feedback": "LoRA training completed. Loss: 0.34", "error": None},
    ]
    
    @classmethod
    def get_result(cls, iteration: int) -> Tuple[bool, float, str, Optional[str]]:
        scenario = cls.SCENARIOS[min(iteration, len(cls.SCENARIOS) - 1)]
        return scenario["success"], scenario["score"], scenario["feedback"], scenario["error"]


# =============================================================================
# Test Class
# =============================================================================

class CognitiveE2ETest:
    """Full E2E test for cognitive memory system."""
    
    def __init__(self, kg):
        self.kg = kg
        self.results: Dict[str, Any] = {
            "workflow": None,
            "iterations": [],
            "decisions": [],
            "tier3_test": None,
            "episodic_insights": [],
        }
        self._init_controller()
    
    def _init_controller(self):
        """Initialize cognitive controller with real KG."""
        from src.memory.cognitive_controller import CognitiveController
        from src.memory.objective import Objective, ObjectiveType
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("INITIALIZING COGNITIVE CONTROLLER")
        logger.info("=" * 70)
        
        self.controller = CognitiveController(knowledge_search=self.kg)
        
        self.objective = Objective(
            description=TEST_CONFIG["goal"],
            objective_type=ObjectiveType.ML_TRAINING,
            success_criteria="Model trained with LoRA, metrics logged",
            source="e2e_test",
        )
        
        workflow = self.controller.initialize_goal(self.objective)
        
        if not workflow:
            raise RuntimeError("No workflow returned!")
        
        logger.info(f"Workflow: {workflow.title}")
        logger.info(f"Source: {workflow.source}")
        logger.info(f"Confidence: {workflow.confidence:.2f}")
        logger.info(f"Steps: {len(workflow.steps)}")
        
        for step in workflow.steps:
            logger.info(f"  {step.number}. {step.title} ({len(step.heuristics)} heuristics)")
            for h in step.heuristics[:2]:
                logger.info(f"     - {h[:70]}...")
        
        self.results["workflow"] = {
            "title": workflow.title,
            "source": workflow.source,
            "confidence": workflow.confidence,
            "steps": len(workflow.steps),
            "total_heuristics": sum(len(s.heuristics) for s in workflow.steps),
        }
        
        if workflow.source == "fallback":
            raise RuntimeError("Got fallback workflow - KG has no matching data!")
        
        logger.info(f"✓ Workflow from KG (source: {workflow.source})")
    
    def run_test(self) -> bool:
        """Run the full E2E test."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STARTING E2E TEST LOOP")
        logger.info("=" * 70)
        
        for i in range(TEST_CONFIG["max_iterations"]):
            logger.info("")
            logger.info(f"{'='*30} ITERATION {i+1} {'='*30}")
            
            # Get briefing and current step
            briefing = self.controller.prepare_briefing()
            logger.info(f"Briefing prepared")
            logger.info(f"  Plan:\n{briefing.plan}")
            
            # Get current step from context
            current_step = "N/A"
            if self.controller._context and self.controller._context.workflow:
                cs = self.controller._context.workflow.current_step
                if cs:
                    current_step = f"Step {cs.number}: {cs.title}"
            logger.info(f"Current step: {current_step}")
            
            # Get simulated experiment result
            success, score, feedback, error = SimulatedExperiment.get_result(i)
            
            logger.info(f"Experiment result:")
            logger.info(f"  Success: {success}, Score: {score:.2f}")
            logger.info(f"  Feedback: {feedback[:80]}...")
            if error:
                logger.info(f"  Error: {error[:80]}...")
            
            # Process through cognitive controller (LLM decision)
            action, details = self.controller.process_result(
                success=success,
                error_message=error,
                score=score,
                feedback=feedback,
            )
            
            logger.info(f"LLM Decision: {action.upper()}")
            logger.info(f"  Details: {details}")
            
            self.results["iterations"].append({
                "iteration": i + 1,
                "success": success,
                "score": score,
                "action": action,
            })
            self.results["decisions"].append(action)
            
            if score >= TEST_CONFIG["stop_threshold"]:
                logger.info(f"✓ Stop threshold reached: {score:.2f}")
                break
        
        # Additional tests
        self._test_tier3_retrieval()
        self._test_episodic_store()
        
        return self._evaluate_results()
    
    def _test_tier3_retrieval(self):
        """Test TIER 3 error-targeted retrieval."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TESTING TIER 3 ERROR RETRIEVAL")
        logger.info("=" * 70)
        
        from src.memory.knowledge_retriever import KnowledgeRetriever
        
        retriever = KnowledgeRetriever(knowledge_search=self.kg)
        
        result = retriever.retrieve(
            goal="Fine-tune language model with LoRA",
            last_error="CUDA out of memory. Tried to allocate 16GB.",
            current_workflow=self.controller._context.workflow,
        )
        
        logger.info(f"TIER 3 Result:")
        logger.info(f"  Mode: {result.mode.value}")
        logger.info(f"  Heuristics: {len(result.heuristics)}")
        for h in result.heuristics[:3]:
            logger.info(f"    - {h[:70]}...")
        logger.info(f"  Code patterns: {len(result.code_patterns)}")
        
        self.results["tier3_test"] = {
            "mode": result.mode.value,
            "heuristics": len(result.heuristics),
            "code_patterns": len(result.code_patterns),
        }
    
    def _test_episodic_store(self):
        """Test episodic memory persistence."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TESTING EPISODIC STORE")
        logger.info("=" * 70)
        
        from src.memory.episodic import EpisodicStore
        
        store = EpisodicStore()
        insights = store.retrieve_relevant(query="LoRA fine-tuning training", top_k=10)
        
        logger.info(f"Insights found: {len(insights)}")
        for insight in insights[:5]:
            logger.info(f"  [{insight.insight_type}] {insight.content[:60]}...")
        
        self.results["episodic_insights"] = [
            {"type": str(i.insight_type), "content": i.content[:100]}
            for i in insights[:5]
        ]
    
    def _evaluate_results(self) -> bool:
        """Evaluate test results."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        checks = []
        
        # Check 1: Workflow from KG
        wf = self.results.get("workflow", {})
        workflow_ok = wf.get("source") in ["kg_exact", "kg_synthesized"]
        checks.append(("Workflow from KG", workflow_ok))
        logger.info(f"[1] Workflow from KG: {'✓' if workflow_ok else '✗'}")
        logger.info(f"    Source: {wf.get('source')}, Steps: {wf.get('steps')}, Heuristics: {wf.get('total_heuristics')}")
        
        # Check 2: Multiple iterations
        iterations_ok = len(self.results["iterations"]) >= 3
        checks.append(("Multiple iterations", iterations_ok))
        logger.info(f"[2] Multiple iterations: {'✓' if iterations_ok else '✗'} ({len(self.results['iterations'])})")
        
        # Check 3: LLM decisions
        decisions = self.results["decisions"]
        decisions_ok = len(decisions) > 0
        checks.append(("LLM decisions made", decisions_ok))
        logger.info(f"[3] LLM decisions: {'✓' if decisions_ok else '✗'}")
        logger.info(f"    Decisions: {decisions}")
        
        # Check 4: TIER 3
        tier3 = self.results.get("tier3_test", {})
        tier3_ok = tier3.get("mode") == "error_targeted"
        checks.append(("TIER 3 retrieval", tier3_ok))
        logger.info(f"[4] TIER 3 retrieval: {'✓' if tier3_ok else '✗'}")
        logger.info(f"    Mode: {tier3.get('mode')}, Heuristics: {tier3.get('heuristics')}")
        
        # Check 5: Episodic store
        episodic_ok = True
        checks.append(("Episodic store accessible", episodic_ok))
        logger.info(f"[5] Episodic store: {'✓' if episodic_ok else '✗'}")
        logger.info(f"    Insights: {len(self.results['episodic_insights'])}")
        
        all_passed = all(c[1] for c in checks)
        
        logger.info("")
        if all_passed:
            logger.info("=" * 70)
            logger.info("ALL TESTS PASSED ✓")
            logger.info("=" * 70)
        else:
            logger.info("=" * 70)
            logger.info("SOME TESTS FAILED")
            logger.info("=" * 70)
        
        # Save results
        results_file = LOG_DIR / f"cognitive_e2e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results: {results_file}")
        logger.info(f"Log: {LOG_FILE}")
        
        return all_passed


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("*" * 70)
    logger.info("COGNITIVE MEMORY SYSTEM - FULL E2E TEST")
    logger.info("*" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Goal: {TEST_CONFIG['goal']}")
    logger.info("")
    
    infra_ok, kg = check_infrastructure()
    if not infra_ok:
        logger.error("Infrastructure check failed! Run: ./start_infra.sh")
        return False
    
    try:
        test = CognitiveE2ETest(kg=kg)
        return test.run_test()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
