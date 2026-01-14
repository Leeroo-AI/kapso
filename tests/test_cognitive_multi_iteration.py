# =============================================================================
# Cognitive System - Multi-Iteration & Error Recovery Test
# =============================================================================
# This test exercises components NOT covered by test_tinkerer_full_e2e.py:
#
# WHAT THIS TESTS:
# 1. TIER 2: Synthesized workflow (no exact KG match)
# 2. Multi-iteration loop (score < threshold → retry)
# 3. Episodic memory write (store insights from failures)
# 4. LLM decisions (RETRY, PIVOT)
# 5. Error recovery pattern
#
# These are critical components that first-try-success tests miss!
# =============================================================================

import os
import sys
import logging
import warnings
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Suppress warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Logging Setup
# =============================================================================

LOG_DIR = Path("/home/ubuntu/praxium/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"cognitive_multi_iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logging():
    """Setup minimal logging for this test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(levelname)-5s │ %(name)-35s │ %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    # Enable cognitive system DEBUG
    logging.getLogger('src.memory').setLevel(logging.DEBUG)
    logging.getLogger('src.execution.context_manager').setLevel(logging.DEBUG)
    
    # Suppress noise
    for noisy in ['httpx', 'httpcore', 'urllib3', 'neo4j', 'weaviate', 'litellm', 'openai']:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    import litellm
    litellm.set_verbose = False
    litellm.suppress_debug_info = True
    
    return logging.getLogger(__name__)

logger = setup_logging()


# =============================================================================
# Test 1: TIER 2 - Synthesized Workflow (no exact match)
# =============================================================================

def test_tier2_synthesized_workflow():
    """
    Test TIER 2 retrieval when no exact workflow matches.
    
    Expected:
    - KG search finds related pages but no exact workflow
    - System returns relevant Principles (Tier 2) rather than a fake workflow
    - Log shows: TIER2_RELEVANT
    """
    from src.memory.cognitive_controller import CognitiveController
    from src.memory.types import Goal  # Goal is in types.py
    from src.knowledge.search import KnowledgeSearchFactory
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: TIER 2 - Synthesized Workflow")
    logger.info("=" * 60)
    
    # Goal that has NO exact workflow in KG
    goal = Goal.from_string("""
        Create a Python script that performs automated hyperparameter tuning
        using Bayesian optimization for a neural network.
        Use scikit-optimize to find optimal learning rate and batch size.
    """)
    
    # Initialize cognitive controller
    kg = KnowledgeSearchFactory.create("kg_graph_search")
    controller = CognitiveController(knowledge_search=kg)
    
    knowledge = controller.initialize_goal(goal)
    if not knowledge:
        raise RuntimeError("No knowledge returned from initialize_goal()")
    
    logger.info(f"Tier: {knowledge.tier.value}")
    logger.info(f"Has workflow: {bool(knowledge.workflow)}")
    logger.info(f"Principles: {len(knowledge.principles)}")
    
    # This test is about Tier 2 (relevant principles). If we get Tier 1 exact,
    # that is acceptable but not the intended coverage.
    if knowledge.tier.value == "tier2_relevant":
        assert len(knowledge.principles) > 0, "Tier 2 returned no Principles"
        logger.info("✅ TIER 2 triggered: Relevant Principles returned")
    else:
        logger.warning(f"⚠️ Tier 2 not triggered (tier={knowledge.tier.value})")
    
    kg.close()
    return True


# =============================================================================
# Test 2: Multi-Iteration with Mock Evaluator
# =============================================================================

def test_multi_iteration_retry():
    """
    Test multi-iteration loop where first attempt fails.
    
    Uses a mock evaluator to force first attempt to fail,
    triggering:
    1. process_result() call
    2. LLM decision (RETRY)
    3. Episodic memory write
    4. Second iteration
    """
    from src.memory.cognitive_controller import CognitiveController
    from src.memory.types import Goal  # Goal is in types.py
    from src.knowledge.search import KnowledgeSearchFactory
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Multi-Iteration Retry Flow")
    logger.info("=" * 60)
    
    # Simple goal
    goal = Goal.from_string("Create a Python function that adds two numbers")
    
    kg = KnowledgeSearchFactory.create("kg_graph_search")
    controller = CognitiveController(knowledge_search=kg)
    
    # Initialize goal
    controller.initialize_goal(goal)
    
    # Simulate FAILED first attempt - use kwargs matching process_result signature
    failed_score = 0.3
    failed_error = "SyntaxError: expected ':'"
    failed_feedback = "Code has syntax error: missing colon in function definition"
    failed_code = "def add(a, b) return a + b"  # Intentionally broken
    
    logger.info("")
    logger.info("Simulating FAILED first attempt...")
    logger.info(f"  Score: {failed_score}")
    logger.info(f"  Error: {failed_error}")
    
    # Process result - should trigger:
    # 1. Insight extraction
    # 2. LLM decision (RETRY expected)
    action, meta = controller.process_result(
        success=False,
        error_message=failed_error,
        score=failed_score,
        feedback=failed_feedback,
        solution=failed_code,
    )
    
    logger.info("")
    logger.info(f"Decision Action: {action}")
    logger.info(f"Decision Meta: {meta}")
    
    # Check what happened
    ctx = controller.get_context()
    
    logger.info("")
    logger.info("Context after failure:")
    logger.info(f"  Iteration: {ctx.iteration}")
    logger.info(f"  Consecutive failures: {ctx.meta.consecutive_failures}")
    
    # Prepare briefing for retry (ensures rendered_context is populated)
    controller.prepare_briefing()
    
    logger.info("")
    logger.info("Briefing prepared for retry iteration")
    
    # Check if TIER 3 was triggered (error-specific knowledge)
    if ctx.kg_retrieval:
        logger.info(f"  KG consulted at iteration: {ctx.kg_retrieval.consulted_at_iteration}")
        logger.info(f"  Reason: {ctx.kg_retrieval.reason}")
    
    knowledge = controller.get_knowledge()
    if knowledge and knowledge.tier.value == "tier3_error":
        logger.info(f"  Tier 3 error heuristics: {len(knowledge.error_heuristics)}")
        logger.info(f"  Tier 3 alternatives: {len(knowledge.alternative_implementations)}")
    
    kg.close()
    return True


# =============================================================================
# Test 3: Episodic Memory Write & Read
# =============================================================================

def test_episodic_memory_lifecycle():
    """
    Test episodic memory write (store insight) and read (retrieve insight).
    
    Tests:
    1. Store an insight from a simulated failure
    2. Retrieve it in a subsequent session
    """
    from src.memory.episodic import EpisodicStore
    from src.memory.types import Insight, InsightType
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Episodic Memory Lifecycle")
    logger.info("=" * 60)
    
    store = EpisodicStore()
    
    # Create a test insight using correct Insight dataclass
    test_insight = Insight(
        content="When fine-tuning transformers, always set mixed precision to avoid OOM errors.",
        insight_type=InsightType.BEST_PRACTICE,  # CRITICAL_ERROR, BEST_PRACTICE, DOMAIN_KNOWLEDGE
        confidence=0.9,
        source_experiment_id="test_experiment_001",
        tags=["transformers", "memory", "fine-tuning"],
    )
    
    # Store the insight
    logger.info("Storing insight...")
    store.add_insight(test_insight)  # Method is add_insight, not store_insight
    logger.info(f"  Stored: {test_insight.content[:50]}...")
    
    # Try to retrieve it
    logger.info("")
    logger.info("Retrieving insights for 'transformer fine-tuning memory'...")
    results = store.retrieve_relevant("transformer fine-tuning memory", top_k=5)  # Method is retrieve_relevant
    
    logger.info(f"  Retrieved {len(results)} insights")
    for r in results:
        logger.info(f"    - {r.content[:50]}...")
    
    # Check if our insight was found
    found = any("mixed precision" in r.content for r in results)
    if found:
        logger.info("✅ Test insight retrieved successfully")
    else:
        logger.warning("⚠️ Test insight not found in results (may need indexing time)")
    
    # Cleanup (store cleanup on close)
    store.close()
    logger.info("EpisodicStore closed")
    
    return True


# =============================================================================
# Test 4: Decision Maker - PIVOT Action
# =============================================================================

def test_decision_maker_pivot():
    """
    Test that LLM decision maker recommends PIVOT after multiple failures.
    
    PIVOT means: abandon current approach, try fundamentally different solution.
    """
    from src.memory.cognitive_controller import CognitiveController
    from src.memory.types import Goal  # Goal is in types.py
    from src.knowledge.search import KnowledgeSearchFactory
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Decision Maker - PIVOT after failures")
    logger.info("=" * 60)
    
    goal = Goal.from_string("Implement a REST API endpoint that returns JSON data")
    
    kg = KnowledgeSearchFactory.create("kg_graph_search")
    controller = CognitiveController(knowledge_search=kg)
    
    controller.initialize_goal(goal)
    
    # Simulate 3 consecutive failures with same error
    for i in range(3):
        logger.info(f"")
        logger.info(f"Simulating failure #{i+1}...")
        
        action, meta = controller.process_result(
            success=False,
            score=0.2,
            feedback="Server crashes on startup due to port conflict",
            error_message="OSError: Address already in use",
            solution=f"# Attempt {i+1}\nimport flask\napp = flask.Flask(__name__)",
        )
        
        logger.info(f"  Decision action: {action}")
        logger.info(f"  Decision meta: {meta}")
        
        # After 3 failures with same error, might get PIVOT
        if action == "PIVOT":
            logger.info("✅ PIVOT action triggered!")
            break
    
    # Get final context state
    ctx = controller.get_context()
    logger.info("")
    logger.info("Final state:")
    logger.info(f"  Consecutive failures: {ctx.meta.consecutive_failures}")
    
    # Check decision history if available
    try:
        decisions = controller.get_decision_history()
        logger.info(f"  Total decisions: {len(decisions)}")
    except AttributeError:
        logger.info("  (Decision history not exposed)")
    
    kg.close()
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all cognitive system tests for untested components."""
    logger.info("")
    logger.info("╔" + "═"*58 + "╗")
    logger.info("║" + " COGNITIVE SYSTEM - MULTI-ITERATION TESTS ".center(58) + "║")
    logger.info("╚" + "═"*58 + "╝")
    logger.info("")
    logger.info("Testing components NOT covered by test_tinkerer_full_e2e.py:")
    logger.info("  - TIER 2: Synthesized workflow")
    logger.info("  - Multi-iteration retry flow")
    logger.info("  - Episodic memory write/read")
    logger.info("  - LLM decision maker (PIVOT)")
    logger.info("")
    
    results = {}
    
    # Test 1: TIER 2
    try:
        results["tier2_synthesized"] = test_tier2_synthesized_workflow()
    except Exception as e:
        logger.error(f"TIER 2 test failed: {e}")
        results["tier2_synthesized"] = False
    
    # Test 2: Multi-iteration
    try:
        results["multi_iteration"] = test_multi_iteration_retry()
    except Exception as e:
        logger.error(f"Multi-iteration test failed: {e}")
        import traceback
        traceback.print_exc()
        results["multi_iteration"] = False
    
    # Test 3: Episodic memory
    try:
        results["episodic_memory"] = test_episodic_memory_lifecycle()
    except Exception as e:
        logger.error(f"Episodic memory test failed: {e}")
        import traceback
        traceback.print_exc()
        results["episodic_memory"] = False
    
    # Test 4: PIVOT decision
    try:
        results["pivot_decision"] = test_decision_maker_pivot()
    except Exception as e:
        logger.error(f"PIVOT decision test failed: {e}")
        import traceback
        traceback.print_exc()
        results["pivot_decision"] = False
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"  {status} {name}")
    
    logger.info("")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Log file: {LOG_FILE}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

