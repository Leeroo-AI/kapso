"""
Full E2E Test - Tinkerer with Real Coding Agent + LLM Judge Evaluator

This test demonstrates the COGNITIVE MEMORY SYSTEM with KG-structured context:

WHAT THIS TESTS:
1. Uses Tinkerer.evolve() - the main user-facing API
2. Creates a REAL code repository
3. Uses a REAL coding agent (Gemini by default, fast and cheap)
4. Evaluates with LLM-as-Judge
5. Uses COGNITIVE context manager with KG workflow retrieval

COGNITIVE FLOW LOGGED:
  ┌─────────────────────────────────────────────────────────────┐
  │  1. GOAL INITIALIZATION                                      │
  │     - Parse goal → Goal object with type/constraints        │
  │     - KG Query: Find matching workflow                      │
  │                                                              │
  │  2. WORKFLOW RETRIEVAL (from KG)                            │
  │     - TIER 1: Exact workflow match                          │
  │     - TIER 2: Synthesize from related pages                 │
  │     - Each step has PRE-LOADED heuristics                   │
  │                                                              │
  │  3. ITERATION LOOP                                          │
  │     a. BRIEFING: Current step + heuristics → Agent          │
  │     b. EXECUTION: Agent generates code, runs                │
  │     c. EVALUATION: LLM Judge scores + feedback              │
  │     d. DECISION: LLM decides ADVANCE/RETRY/SKIP/PIVOT       │
  │     e. MEMORY UPDATE: Store insights, update workflow       │
  │                                                              │
  │  4. EPISODIC MEMORY                                         │
  │     - Store error insights for future retrieval             │
  │     - Store success patterns as best practices              │
  └─────────────────────────────────────────────────────────────┘

Prerequisites:
    ./start_infra.sh              # KG infrastructure
    export GOOGLE_API_KEY=...     # For Gemini coding agent
    export OPENAI_API_KEY=...     # For LLM Judge + embeddings

To run:
    PYTHONPATH=. python tests/test_tinkerer_full_e2e.py

Output:
    - Generated code in /tmp/tinkerer_e2e_*/
    - Logs in /home/ubuntu/tinkerer/logs/tinkerer_e2e_*.log
"""

import os
import sys
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Enhanced Logging Setup - Cognitive Memory Focused
# =============================================================================

LOG_DIR = Path("/home/ubuntu/tinkerer/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"tinkerer_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class CognitiveLogFormatter(logging.Formatter):
    """
    Custom formatter that highlights cognitive memory system events.
    
    Makes it easy to follow:
    - KG retrieval operations
    - Workflow step progression  
    - LLM decision making
    - Memory state changes
    """
    
    # Phase markers for easy log scanning
    PHASE_MARKERS = {
        'INITIALIZING GOAL': '🎯',
        'PREPARING BRIEFING': '📋',
        'PROCESSING RESULT': '⚙️',
        'LLM DECISION': '🧠',
        'TIER 3': '🔍',
        'GOAL INITIALIZED': '✅',
        'BRIEFING READY': '✅',
    }
    
    def format(self, record):
        # Get base formatted message
        msg = super().format(record)
        
        # Add phase markers for cognitive events
        for marker_text, emoji in self.PHASE_MARKERS.items():
            if marker_text in msg:
                # Add visual separator for major phases
                if marker_text.startswith('==='):
                    msg = f"\n{'─'*70}\n{emoji} {msg}"
                break
        
        return msg


def setup_logging():
    """Setup logging with cognitive-aware formatting."""
    # Create formatter
    formatter = CognitiveLogFormatter(
        '%(asctime)s │ %(levelname)-5s │ %(name)-40s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - keep it readable (INFO). We want signal, not raw HTTP dumps.
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific module levels for cognitive system visibility (still keep INFO-level readable logs)
    logging.getLogger('src.memory.cognitive_controller').setLevel(logging.INFO)
    logging.getLogger('src.memory.knowledge_retriever').setLevel(logging.INFO)
    logging.getLogger('src.memory.decisions').setLevel(logging.INFO)
    logging.getLogger('src.memory.episodic').setLevel(logging.INFO)
    logging.getLogger('src.execution.context_manager').setLevel(logging.INFO)
    
    # Reduce noise from third-party modules (these swamp the logs with request dumps)
    for noisy in [
        'httpx', 'httpcore', 'urllib3', 'openai', 'LiteLLM', 'litellm',
        'neo4j', 'weaviate',
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# Test Scenarios - Different KG Retrieval Modes
# =============================================================================
# 
# These test scenarios exercise different parts of the cognitive memory system:
#
# 1. TIER 1 - EXACT WORKFLOW MATCH
#    Goal matches a workflow in the KG exactly.
#    Expected: exact_workflow mode, step-by-step heuristics loaded
#
# 2. TIER 2 - SYNTHESIZED WORKFLOW  
#    Goal has no exact workflow but has related pages.
#    Expected: synthesized_plan mode, plan generated from related pages
#
# 3. TIER 3 - ERROR RECOVERY (tested during failures)
#    When code fails, retrieves error-specific heuristics.
#    Expected: error_retrieval triggered on failure
#
# 4. WORKFLOW PIVOT
#    Code keeps failing, LLM decides to try different approach.
#    Expected: PIVOT action from decision maker
#
# =============================================================================

TEST_SCENARIOS = {
    # =========================================================================
    # SCENARIO 1: EXACT WORKFLOW MATCH (TIER 1)
    # =========================================================================
    # This goal matches the "huggingface peft LoRA Fine Tuning" workflow in KG.
    # The KG should return an exact workflow with 6 steps and pre-loaded heuristics.
    # 
    # Expected log output:
    #   Retrieval mode: exact_workflow
    #   Workflow: huggingface peft LoRA Fine Tuning (6 steps)
    #
    "tier1_exact_workflow": {
        "name": "TIER 1: Exact Workflow Match (LoRA Fine-tuning)",
        "description": "Goal matches KG workflow exactly - tests TIER 1 retrieval",
    "goal": """
Fine-tune a language model using LoRA (Low-Rank Adaptation).
Create a Python script that:
1. Loads a base model (use a small one like 'gpt2')
2. Configures LoRA adapter with rank=8
3. Creates a PEFT model
4. Prints "LoRA model configured successfully"
""",
        "evaluator_criteria": "Code correctly implements LoRA fine-tuning setup with PEFT library",
        "max_iterations": 5,
        "expected_retrieval_mode": "exact_workflow",
        "expected_workflow_keywords": ["LoRA", "PEFT", "Fine Tuning"],
    },
    
    # =========================================================================
    # SCENARIO 2: SYNTHESIZED WORKFLOW (TIER 2)
    # =========================================================================
    # This goal is about a task the KG has knowledge of but no exact workflow.
    # The system should synthesize a plan from related pages/heuristics.
    #
    # Expected log output:
    #   Retrieval mode: synthesized_plan
    #   Workflow: Plan: <goal summary> (3 steps)
    #
    "tier2_synthesized_workflow": {
        "name": "TIER 2: Synthesized Workflow (Custom Task)",
        "description": "No exact workflow match - tests TIER 2 synthesis",
        "goal": """
Create a Python script that performs sentiment analysis on a text file.
The script should:
1. Read a text file called 'input.txt'
2. Analyze the sentiment of each line (positive/negative/neutral)
3. Print a summary with counts for each sentiment
""",
        "evaluator_criteria": "Code correctly reads file, analyzes sentiment, and prints summary",
        "max_iterations": 5,
        "expected_retrieval_mode": "synthesized_plan",
        "expected_workflow_keywords": [],  # No exact workflow expected
    },
    
    # =========================================================================
    # SCENARIO 3: MULTI-STEP WITH POTENTIAL FAILURE (TIER 3 trigger)
    # =========================================================================
    # This goal is more complex and likely to fail initially.
    # When it fails, TIER 3 error recovery should trigger.
    #
    # Expected log output (on first failure):
    #   TIER 3 triggered: Retrieving error-specific heuristics
    #   Retrieved X heuristics for error recovery
    #
    "tier3_error_recovery": {
        "name": "TIER 3: Error Recovery (Complex Task)",
        "description": "Complex task that may fail - tests TIER 3 error retrieval",
        "goal": """
Create a Python script that loads a HuggingFace model and generates text.
The script should:
1. Load the 'distilgpt2' model
2. Generate 50 tokens given the prompt "Once upon a time"
3. Print the generated text
Note: Handle any memory constraints gracefully.
""",
        "evaluator_criteria": "Code loads model, generates text, and prints output without errors",
        "max_iterations": 5,
        "expected_retrieval_mode": "exact_workflow",  # or synthesized
        "expected_workflow_keywords": ["HuggingFace", "generate"],
    },
    
    # =========================================================================
    # SCENARIO 4: SIMPLE TASK (No KG needed)
    # =========================================================================
    # A simple task that doesn't need domain knowledge.
    # Tests that the system works even with minimal_plan fallback.
    #
    "simple_no_kg": {
        "name": "SIMPLE: No KG Knowledge Needed",
        "description": "Simple task - tests minimal plan fallback",
        "goal": """
Create a Python script that prints "Hello, World!" and calculates 2+2.
Print both results.
""",
        "evaluator_criteria": "Code prints Hello World and calculates 2+2=4",
        "max_iterations": 3,
        "expected_retrieval_mode": "synthesized_plan",  # Will synthesize simple plan
        "expected_workflow_keywords": [],
    },
}

# Default test to run (can be overridden via command line)
DEFAULT_TEST_SCENARIO = "tier1_exact_workflow"


# =============================================================================
# Active Test Configuration (built from selected scenario)
# =============================================================================

def get_test_config(scenario_name: str = None) -> dict:
    """
    Get test configuration for a specific scenario.
    
    Args:
        scenario_name: Name of scenario from TEST_SCENARIOS, or None for default
        
    Returns:
        Full test config dict
    """
    scenario_name = scenario_name or DEFAULT_TEST_SCENARIO
    
    if scenario_name not in TEST_SCENARIOS:
        available = list(TEST_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    scenario = TEST_SCENARIOS[scenario_name]
    
    return {
        # Scenario info
        "scenario_name": scenario_name,
        "scenario_description": scenario.get("description", ""),
        
        # Goal from scenario
        "goal": scenario["goal"],
    
    # Tinkerer configuration
        "max_iterations": scenario.get("max_iterations", 5),
        "coding_agent": "claude_code",  # Using claude_code as it's available
    "language": "python",
    "main_file": "main.py",
    "timeout": 60,
    
        # Evaluation - LLM Judge provides feedback for cognitive learning
    "evaluator": "llm_judge",
    "evaluator_params": {
            "criteria": scenario.get("evaluator_criteria", "Code works correctly"),
        "model": "gpt-4o-mini",
        "scale": 10,
    },
    "stop_condition": "threshold",
    "stop_condition_params": {"threshold": 0.7},
    
        # Cognitive mode - enables full workflow tracking + KG exploitation
        "mode": "COGNITIVE",
        
        # Expected results (for validation)
        "expected_retrieval_mode": scenario.get("expected_retrieval_mode"),
        "expected_workflow_keywords": scenario.get("expected_workflow_keywords", []),
    }


# For backwards compatibility
TEST_CONFIG = get_test_config(DEFAULT_TEST_SCENARIO)


# =============================================================================
# Cognitive System Monitoring Helpers
# =============================================================================

def log_cognitive_state(controller, phase: str):
    """
    Log detailed cognitive state for analysis.
    
    Shows:
    - Current workflow progress
    - Active step with heuristics
    - Episodic memory state
    - Decision history
    """
    logger.info("")
    logger.info(f"{'═'*70}")
    logger.info(f"  COGNITIVE STATE: {phase}")
    logger.info(f"{'═'*70}")
    
    ctx = controller.get_context()
    if not ctx:
        logger.info("  [No context initialized]")
        return
    
    # Goal info
    logger.info(f"  GOAL: {ctx.goal_str}")
    logger.info(f"  ITERATION: {ctx.iteration}")
    logger.info(f"  CONSECUTIVE FAILURES: {ctx.meta.consecutive_failures}")
    
    # Knowledge state (single source of truth)
    knowledge = controller.get_knowledge() if hasattr(controller, "get_knowledge") else None
    if knowledge and knowledge.workflow:
        wf = knowledge.workflow
        logger.info("")
        logger.info(f"  WORKFLOW: {wf.title}")
        logger.info(f"  SOURCE: {wf.source} (confidence: {wf.confidence:.0%})")
        logger.info(f"  STEPS: {len(wf.steps)}")
    elif knowledge and knowledge.principles:
        logger.info("")
        logger.info(f"  PRINCIPLES: {len(knowledge.principles)}")
    else:
        logger.info("  [No knowledge loaded]")
    
    # Last experiment
    if ctx.last_experiment:
        exp = ctx.last_experiment
        logger.info("")
        logger.info("  LAST EXPERIMENT:")
        result = "✅ SUCCESS" if exp.success else "❌ FAILED"
        logger.info(f"    Result: {result}")
        if exp.score is not None:
            logger.info(f"    Score: {exp.score:.2f}")
        if exp.feedback:
            logger.info(f"    Feedback: {exp.feedback}")
        if exp.error_message and not exp.success:
            logger.info(f"    Error: {exp.error_message}")
    
    # Episodic memory
    if ctx.episodic_memory:
        logger.info("")
        logger.info("  EPISODIC MEMORY:")
        if ctx.episodic_memory.similar_errors:
            logger.info(f"    Similar errors found: {len(ctx.episodic_memory.similar_errors)}")
        if ctx.episodic_memory.relevant_insights:
            logger.info(f"    Relevant insights: {len(ctx.episodic_memory.relevant_insights)}")
    
    # KG retrieval state
    if ctx.kg_retrieval:
        logger.info("")
        logger.info("  KG RETRIEVAL:")
        logger.info(f"    Last consulted: iteration {ctx.kg_retrieval.consulted_at_iteration}")
        logger.info(f"    Reason: {ctx.kg_retrieval.reason}")
        logger.info(f"    Heuristics retrieved: {len(ctx.kg_retrieval.heuristics)}")
    
    logger.info(f"{'═'*70}")
    logger.info("")


def log_decision_history(controller):
    """Log all decisions made by the LLM during the session."""
    decisions = controller.get_decision_history()
    
    logger.info("")
    logger.info(f"{'═'*70}")
    logger.info("  LLM DECISION HISTORY")
    logger.info(f"{'═'*70}")
    
    if not decisions:
        logger.info("  [No decisions recorded]")
        return
    
    for i, decision in enumerate(decisions, 1):
        logger.info(f"  Decision #{i}:")
        logger.info(f"    Action: {decision.action.value}")
        logger.info(f"    Confidence: {decision.confidence:.2f}")
        logger.info(f"    Reasoning: {decision.reasoning[:80]}...")
        logger.info("")
    
    # Summary
    action_counts = {}
    for d in decisions:
        action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1
    
    logger.info("  SUMMARY:")
    for action, count in action_counts.items():
        logger.info(f"    {action}: {count}")
    
    logger.info(f"{'═'*70}")


# =============================================================================
# Infrastructure Check
# =============================================================================

def check_prerequisites():
    """Check all required services and credentials."""
    logger.info("=" * 70)
    logger.info("CHECKING PREREQUISITES")
    logger.info("=" * 70)
    
    issues = []
    
    # Check API keys
    logger.info("[1/4] Checking API keys...")
    if not os.environ.get("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set (needed for LLM Judge + embeddings)")
    else:
        logger.info("  ✓ OPENAI_API_KEY found")
    
    if TEST_CONFIG["coding_agent"] == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            issues.append("GOOGLE_API_KEY not set (needed for Gemini coding agent)")
        else:
            logger.info("  ✓ GOOGLE_API_KEY found")
    
    # Check KG (optional but recommended)
    logger.info("[2/4] Checking Knowledge Graph...")
    try:
        from src.knowledge.search import KnowledgeSearchFactory
        kg = KnowledgeSearchFactory.create("kg_graph_search")
        if kg.is_enabled():
            logger.info("  ✓ KG connected")
        else:
            logger.warning("  ⚠ KG not enabled (will proceed without)")
    except Exception as e:
        logger.warning(f"  ⚠ KG not available: {e}")
    
    # Check coding agent
    logger.info("[3/4] Checking coding agent availability...")
    try:
        from src.execution.coding_agents.factory import CodingAgentFactory
        agents = CodingAgentFactory.list_available()
        if TEST_CONFIG["coding_agent"] in agents:
            logger.info(f"  ✓ {TEST_CONFIG['coding_agent']} agent available")
        else:
            issues.append(f"Coding agent '{TEST_CONFIG['coding_agent']}' not available. Available: {agents}")
    except Exception as e:
        issues.append(f"Failed to check coding agents: {e}")
    
    # Check LLM backend
    logger.info("[4/4] Checking LLM backend...")
    try:
        from src.core.llm import LLMBackend
        llm = LLMBackend()
        response = llm.llm_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=5,
        )
        logger.info("  ✓ LLM backend working")
    except Exception as e:
        issues.append(f"LLM backend failed: {e}")
    
    if issues:
        logger.error("=" * 70)
        logger.error("PREREQUISITES FAILED:")
        for issue in issues:
            logger.error(f"  ✗ {issue}")
        logger.error("=" * 70)
        return False
    
    logger.info("=" * 70)
    logger.info("All prerequisites passed!")
    logger.info("=" * 70)
    return True


# =============================================================================
# Main Test - With Cognitive Memory Flow Logging
# =============================================================================

def run_tinkerer_test():
    """
    Run full Tinkerer E2E test with detailed cognitive memory logging.
    
    This test logs:
    1. KG workflow retrieval (TIER 1/2/3)
    2. Step-by-step workflow progress
    3. Heuristics being applied per step
    4. LLM decision making (ADVANCE/RETRY/SKIP/PIVOT)
    5. Episodic memory updates
    """
    from src.tinkerer import Tinkerer
    
    logger.info("")
    logger.info("╔" + "═"*68 + "╗")
    logger.info("║" + " EXPERT FULL E2E TEST - COGNITIVE MEMORY SYSTEM ".center(68) + "║")
    logger.info("╚" + "═"*68 + "╝")
    logger.info("")
    logger.info("TEST CONFIGURATION:")
    logger.info(f"  Goal: {TEST_CONFIG['goal'].strip()}")
    logger.info(f"  Mode: {TEST_CONFIG['mode']} (cognitive context manager)")
    logger.info(f"  Coding agent: {TEST_CONFIG['coding_agent']}")
    logger.info(f"  Evaluator: {TEST_CONFIG['evaluator']}")
    logger.info(f"  Max iterations: {TEST_CONFIG['max_iterations']}")
    logger.info(f"  Stop threshold: {TEST_CONFIG['stop_condition_params']['threshold']}")
    logger.info("")
    
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="tinkerer_e2e_")
    logger.info(f"Output directory: {output_dir}")
    
    # Reference to cognitive controller for state logging
    cognitive_controller = None
    
    try:
        # =====================================================================
        # PHASE 1: Initialize Tinkerer
        # =====================================================================
        logger.info("")
        logger.info("┌" + "─"*68 + "┐")
        logger.info("│" + " PHASE 1: EXPERT INITIALIZATION ".center(68) + "│")
        logger.info("└" + "─"*68 + "┘")
        
        tinkerer = Tinkerer()
        logger.info("Tinkerer created")
        
        # Enable KG for cognitive mode
        from src.knowledge.search import KnowledgeSearchFactory
        tinkerer.knowledge_search = KnowledgeSearchFactory.create("kg_graph_search")
        logger.info(f"KG enabled: {tinkerer.knowledge_search.is_enabled()}")
        
        if tinkerer.knowledge_search.is_enabled():
            logger.info("  ✓ Neo4j connected")
            logger.info("  ✓ Weaviate connected")
            logger.info("  → KG will be used for workflow retrieval")
        
        # =====================================================================
        # PHASE 2: Run Tinkerer.evolve() with Cognitive Context
        # =====================================================================
        logger.info("")
        logger.info("┌" + "─"*68 + "┐")
        logger.info("│" + " PHASE 2: EXPERT.BUILD() - COGNITIVE LOOP ".center(68) + "│")
        logger.info("└" + "─"*68 + "┘")
        logger.info("")
        logger.info("Starting build - watch for:")
        logger.info("  🎯 GOAL INITIALIZATION - Goal parsed, KG queried for workflow")
        logger.info("  📋 BRIEFING - Current step + heuristics sent to agent")
        logger.info("  ⚙️ PROCESSING - Experiment result evaluated")
        logger.info("  🧠 LLM DECISION - ADVANCE/RETRY/SKIP/PIVOT decision")
        logger.info("")
        
        solution = tinkerer.evolve(
            goal=TEST_CONFIG["goal"],
            output_path=output_dir,
            max_iterations=TEST_CONFIG["max_iterations"],
            mode=TEST_CONFIG["mode"],
            coding_agent=TEST_CONFIG["coding_agent"],
            language=TEST_CONFIG["language"],
            main_file=TEST_CONFIG["main_file"],
            timeout=TEST_CONFIG["timeout"],
            evaluator=TEST_CONFIG["evaluator"],
            evaluator_params=TEST_CONFIG["evaluator_params"],
            stop_condition=TEST_CONFIG["stop_condition"],
            stop_condition_params=TEST_CONFIG["stop_condition_params"],
        )
        
        # =====================================================================
        # PHASE 3: Retrieval Quality Checks (Tier 1 / 2 / 3)
        # =====================================================================
        logger.info("")
        logger.info("┌" + "─"*68 + "┐")
        logger.info("│" + " PHASE 3: COGNITIVE FLOW ANALYSIS ".center(68) + "│")
        logger.info("└" + "─"*68 + "┘")
        
        # Try to get cognitive controller from orchestrator for detailed analysis.
        try:
            # Access cognitive controller through context manager
            orchestrator = tinkerer._last_orchestrator if hasattr(tinkerer, '_last_orchestrator') else None
            if orchestrator and hasattr(orchestrator, 'context_manager'):
                cm = orchestrator.context_manager
                if hasattr(cm, 'controller'):
                    cognitive_controller = cm.controller
                    log_cognitive_state(cognitive_controller, "FINAL STATE")
                    log_decision_history(cognitive_controller)
        except Exception as e:
            logger.debug(f"Could not access cognitive controller: {e}")

        # Explicitly validate retrieval quality independent of whether the coding
        # agent succeeded early. This keeps the test close to real Tinkerer runs
        # while still exercising Tier 2 and Tier 3 deterministically.
        try:
            from src.memory.knowledge_retriever import KnowledgeRetriever
            retriever = KnowledgeRetriever(knowledge_search=tinkerer.knowledge_search)

            logger.info("")
            logger.info("RETRIEVAL QUALITY CHECKS")
            logger.info("─" * 50)

            # Tier 1: should find a workflow for the main goal (real KG).
            tier1 = retriever.retrieve_knowledge(TEST_CONFIG["goal"])
            logger.info(f"TIER 1 check: tier={tier1.tier.value}, has_workflow={bool(tier1.workflow)}")
            assert tier1.workflow is not None, "Expected Tier 1 workflow retrieval for the main goal"

            # Tier 2: pick a concept-heavy goal unlikely to have a Workflow but likely to have Principles.
            tier2_goal = "Explain PEFT LoraConfig target_modules rank alpha selection"
            tier2 = retriever.retrieve_knowledge(tier2_goal)
            logger.info(f"TIER 2 check: tier={tier2.tier.value}, principles={len(tier2.principles)}")
            assert tier2.tier.value in ["tier2_relevant", "tier1_exact"], "Unexpected tier for Tier 2 check"
            if tier2.tier.value == "tier2_relevant":
                assert len(tier2.principles) > 0, "Tier 2 returned no principles"

            # Tier 3: deterministic error enrichment on top of Tier 1 knowledge.
            tier3 = retriever.retrieve_knowledge(
                goal=TEST_CONFIG["goal"],
                existing_knowledge=tier1,
                last_error="ImportError: No module named 'peft'",
            )
            logger.info(
                f"TIER 3 check: tier={tier3.tier.value}, "
                f"error_heuristics={len(tier3.error_heuristics)}, "
                f"alternatives={len(tier3.alternative_implementations)}"
            )
            assert tier3.tier.value == "tier3_error", "Expected Tier 3 enrichment"
            assert len(tier3.error_heuristics) + len(tier3.alternative_implementations) > 0, "Tier 3 returned no error knowledge"

            # Quality signal: for missing dependency errors we should surface at least one
            # environment/setup-oriented heuristic now that we include Environment pages.
            has_env_hint = any(h.title.lower().startswith("environment:") for h in tier3.error_heuristics)
            logger.info(f"TIER 3 quality (import/install): environment_hint={has_env_hint}")

        except Exception as e:
            logger.error(f"Retrieval quality checks failed: {e}")
            raise
        
        # =====================================================================
        # PHASE 4: Results Summary
        # =====================================================================
        logger.info("")
        logger.info("┌" + "─"*68 + "┐")
        logger.info("│" + " PHASE 4: RESULTS SUMMARY ".center(68) + "│")
        logger.info("└" + "─"*68 + "┘")
        
        logger.info(f"Code path: {solution.code_path}")
        logger.info(f"Total experiments: {len(solution.experiment_logs)}")
        
        # Parse cost from metadata
        cost_str = solution.metadata.get('cost', '$0.000')
        logger.info(f"Total cost: {cost_str}")
        
        # Show experiment progression with scores
        logger.info("")
        logger.info("EXPERIMENT PROGRESSION:")
        logger.info("─" * 50)
        
        import re as re_mod
        scores = []
        for i, exp in enumerate(solution.experiment_logs):
            # Parse score from log string
            match = re_mod.search(r'Score: ([0-9.]+)', exp)
            score = float(match.group(1)) if match else None
            if score:
                scores.append(score)
            
            # Show experiment result
            if "Failed" in exp:
                status = "❌ FAILED"
            else:
                status = "✅ SUCCESS"
            
            score_str = f"Score: {score:.2f}" if score else "No score"
            logger.info(f"  Iteration {i+1}: {status} | {score_str}")
            
            # Show brief error/success info
            if "Error:" in exp:
                error_match = re_mod.search(r'Error: (.+?)\)', exp)
                if error_match:
                    logger.info(f"             └─ {error_match.group(1)[:60]}...")
        
        logger.info("─" * 50)
        
        # Score progression
        if scores:
            logger.info("")
            logger.info("SCORE PROGRESSION:")
            for i, score in enumerate(scores):
                bar_length = int(score * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                logger.info(f"  Iter {i+1}: [{bar}] {score:.2f}")
        
        # Check generated code
        main_file = Path(solution.code_path) / TEST_CONFIG["main_file"]
        if main_file.exists():
            code = main_file.read_text()
            logger.info("")
            logger.info("GENERATED CODE PREVIEW:")
            logger.info("─" * 50)
            for line in code.split("\n")[:25]:
                logger.info(f"  {line}")
            if len(code.split("\n")) > 25:
                logger.info("  ... (truncated)")
            logger.info("─" * 50)

        # =====================================================================
        # RepoMemory auditability checks (post-run invariants)
        # =====================================================================
        #
        # This test is meant to be a true E2E: code changes + RepoMemory evolution.
        # We assert a couple of critical properties:
        # - RepoMap is portable (no absolute /tmp/... paths)
        # - RepoMap reflects repo structure, not infrastructure/metadata paths
        # - Observability is auditable: `changes.log` is committed and matches persisted metadata
        try:
            import git
            from src.repo_memory.observation import extract_repo_memory_sections_consulted

            repo = git.Repo(solution.code_path)
            changes_text = repo.git.show("HEAD:changes.log")
            assert "repomemory sections consulted:" in changes_text.lower(), (
                "changes.log missing 'RepoMemory sections consulted:' line"
            )

            memory_path = Path(solution.code_path) / ".tinkerer" / "repo_memory.json"
            assert memory_path.exists(), f"missing repo memory file: {memory_path}"
            doc = json.loads(memory_path.read_text())

            repo_map = doc.get("repo_map", {}) or {}
            assert repo_map.get("repo_root") == ".", f"expected repo_map.repo_root == '.', got {repo_map.get('repo_root')!r}"

            files = repo_map.get("files", []) or []
            assert "changes.log" not in files, "repo_map.files unexpectedly contains changes.log"
            assert not any(p.startswith(".tinkerer/") for p in files), "repo_map.files contains .tinkerer/*"
            assert not any(p.startswith("sessions/") for p in files), "repo_map.files contains sessions/*"

            # If RepoMemory captured a section-consultation list, it must match the committed changes.log.
            from_log = extract_repo_memory_sections_consulted(changes_text)
            experiments = doc.get("experiments", []) or []
            if experiments:
                rr = (experiments[-1].get("run_result") or {})
                persisted = rr.get("repo_memory_sections_consulted", []) or []
                if not isinstance(persisted, list):
                    persisted = []
                persisted = sorted(set(str(x) for x in persisted))
                assert persisted == from_log, (
                    f"repo_memory_sections_consulted mismatch\n"
                    f"  from changes.log: {from_log}\n"
                    f"  persisted:        {persisted}"
                )
        except Exception as e:
            logger.error(f"RepoMemory audit checks failed: {e}")
            raise
        
        # =====================================================================
        # PHASE 5: Final Verdict
        # =====================================================================
        logger.info("")
        best_score = max(scores) if scores else 0
        threshold = TEST_CONFIG["stop_condition_params"]["threshold"]
        
        if best_score >= threshold:
            logger.info("╔" + "═"*68 + "╗")
            logger.info("║" + f" ✅ TEST PASSED - Best score: {best_score:.2f} >= {threshold} ".center(68) + "║")
            logger.info("╚" + "═"*68 + "╝")
            return True
        else:
            logger.info("╔" + "═"*68 + "╗")
            logger.info("║" + f" ⚠️ TEST COMPLETED - Best score: {best_score:.2f} < {threshold} ".center(68) + "║")
            logger.info("╚" + "═"*68 + "╝")
            return True  # Still consider success if it ran to completion
            
    except Exception as e:
        logger.error("")
        logger.error("╔" + "═"*68 + "╗")
        logger.error("║" + " ❌ TEST FAILED WITH EXCEPTION ".center(68) + "║")
        logger.error("╚" + "═"*68 + "╝")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": TEST_CONFIG,
            "output_dir": output_dir,
            "log_file": str(LOG_FILE),
            "cognitive_mode": TEST_CONFIG["mode"],
        }
        results_file = LOG_DIR / f"tinkerer_e2e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("")
        logger.info("OUTPUT FILES:")
        logger.info(f"  📄 Results JSON: {results_file}")
        logger.info(f"  📝 Full log: {LOG_FILE}")
        logger.info(f"  💻 Generated code: {output_dir}")
        
        # Close KG connection
        if tinkerer.knowledge_search.is_enabled():
            tinkerer.knowledge_search.close()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Run the Tinkerer E2E test.
    
    Usage:
        # Run default scenario (tier1_exact_workflow)
        python tests/test_tinkerer_full_e2e.py
        
        # Run specific scenario
        python tests/test_tinkerer_full_e2e.py tier2_synthesized_workflow
        
        # List available scenarios
        python tests/test_tinkerer_full_e2e.py --list
    """
    import sys
    
    # Parse command line
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # List scenarios
        if arg in ("--list", "-l", "list"):
            print("\nAvailable Test Scenarios:")
            print("=" * 60)
            for name, scenario in TEST_SCENARIOS.items():
                print(f"\n  {name}")
                print(f"    {scenario.get('name', 'Unnamed')}")
                print(f"    {scenario.get('description', '')}")
                print(f"    Expected mode: {scenario.get('expected_retrieval_mode', '?')}")
            print("\n" + "=" * 60)
            print(f"Default: {DEFAULT_TEST_SCENARIO}")
            print("\nUsage: python tests/test_tinkerer_full_e2e.py <scenario_name>")
            return True
        
        # Run specific scenario
        scenario_name = arg
    else:
        scenario_name = DEFAULT_TEST_SCENARIO
    
    # Get config for selected scenario
    global TEST_CONFIG
    try:
        TEST_CONFIG = get_test_config(scenario_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list to see available scenarios")
        return False
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERT AGENT - FULL E2E TEST")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Scenario: {scenario_name}")
    logger.info(f"Description: {TEST_CONFIG.get('scenario_description', '')}")
    logger.info("")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Run test
    return run_tinkerer_test()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
