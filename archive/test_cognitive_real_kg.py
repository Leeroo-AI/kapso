"""
Cognitive Memory Test - REQUIRES REAL KG INFRASTRUCTURE

This test FAILS if KG infrastructure is not running.
No mocking. No fallbacks. Real data only.

Prerequisites:
    ./start_infra.sh  # Must be running

To run:
    PYTHONPATH=. python tests/test_cognitive_real_kg.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Logging Setup - Save to file
# =============================================================================
LOG_DIR = Path("/home/ubuntu/kapso/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"cognitive_real_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {LOG_FILE}")


def check_kg_available():
    """Check if KG infrastructure is running. FAIL if not."""
    try:
        from kapso.knowledge.search import KnowledgeSearchFactory
        kg = KnowledgeSearchFactory.create("kg_graph_search")
        
        if not kg.is_enabled():
            logger.error("KG is not enabled!")
            logger.error("Run: ./start_infra.sh")
            return None
        
        # Try a test query to verify connection
        result = kg.search("test query", context=None)
        logger.info(f"KG connected. Test query returned {len(result.results)} results.")
        return kg
        
    except Exception as e:
        logger.error(f"Failed to connect to KG: {e}")
        logger.error("Make sure infrastructure is running: ./start_infra.sh")
        return None


def test_real_kg_workflow():
    """
    Test cognitive system with REAL KG.
    
    NO FALLBACKS. If KG doesn't have workflows, the test fails.
    """
    from kapso.memory.cognitive_controller import CognitiveController
    from kapso.memory.objective import Objective, ObjectiveType, DataFile
    
    logger.info("=" * 60)
    logger.info("COGNITIVE TEST - REAL KG (NO MOCKING)")
    logger.info("=" * 60)
    
    # STEP 1: Connect to real KG
    logger.info("\n[1] Connecting to KG...")
    kg = check_kg_available()
    
    if kg is None:
        logger.error("FAILED: KG not available. Cannot run test without real KG.")
        return False
    
    # STEP 2: Create controller with REAL KG
    logger.info("\n[2] Creating controller with real KG...")
    controller = CognitiveController(
        knowledge_search=kg,  # REAL KG, not None!
    )
    
    try:
        # STEP 3: Initialize with a goal that should match a real workflow
        logger.info("\n[3] Initializing goal...")
        
        objective = Objective(
            description="Fine-tune a language model with LoRA",
            objective_type=ObjectiveType.ML_TRAINING,
            success_criteria="Model trains without error",
            source="test",
        )
        
        knowledge = controller.initialize_goal(objective)
        
        # STEP 4: Verify workflow came from KG, not fallback
        logger.info("\n[4] Verifying workflow source...")
        
        if knowledge is None or knowledge.workflow is None:
            logger.error("FAILED: No workflow returned")
            return False
        
        workflow = knowledge.workflow
        
        # Check it's not a fallback
        if workflow.source == "fallback":
            logger.error(f"FAILED: Got fallback workflow, not real KG data")
            logger.error(f"Workflow: {workflow.title}")
            logger.error("The KG may not have matching workflows loaded.")
            return False
        
        if workflow.source not in ["kg_exact", "kg_synthesized"]:
            logger.warning(f"Workflow source: {workflow.source} (expected kg_exact or kg_synthesized)")
        
        logger.info(f"✓ Workflow from KG: {workflow.title}")
        logger.info(f"  Source: {workflow.source}")
        logger.info(f"  Confidence: {workflow.confidence}")
        logger.info(f"  Steps: {len(workflow.steps)}")
        
        for step in workflow.steps:
            logger.info(f"    {step.number}. {step.principle.title}")
            heuristics = step.principle.heuristics
            logger.info(f"       Heuristics: {len(heuristics)}")
            for h in heuristics[:2]:
                logger.info(f"         - {h.title}")
        
        # STEP 5: Process a result
        logger.info("\n[5] Processing experiment result...")
        
        action, details = controller.process_result(
            success=True,
            score=0.85,
            feedback="Model trained successfully with LoRA",
        )
        
        logger.info(f"Action: {action}")
        logger.info(f"Details: {details}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST PASSED - Used real KG data")
        logger.info("=" * 60)
        
        return True
    finally:
        # Prevent leaked sockets in local test runs (Weaviate/Neo4j clients).
        try:
            controller.close()
        except Exception:
            pass
        try:
            kg.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        result = test_real_kg_workflow()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
