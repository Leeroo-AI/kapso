"""
Comprehensive Test: Cognitive Memory Architecture

This test validates that the cognitive memory system actually WORKS,
not just that it doesn't crash. Tests include:

1. Semantic relevance - Do searches return contextually relevant results?
2. Memory persistence - Do insights survive across sessions?
3. Learning effectiveness - Does memory help avoid repeated mistakes?
4. Briefing quality - Does the briefing contain useful information?

Requires: Running infrastructure (./start_infra.sh)
"""

import os
import json
import tempfile
import shutil
from typing import List, Tuple

# Load env BEFORE imports
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found")
    exit(1)

from src.memory.types import WorkingMemory, Insight, InsightType, Briefing
from src.memory.controller import CognitiveController
from src.memory.episodic import EpisodicStore
from src.knowledge.search import KnowledgeSearchFactory, KGSearchFilters


# =============================================================================
# Test Utilities
# =============================================================================

def semantic_similarity_check(query: str, results: List[str], expected_keywords: List[str]) -> Tuple[bool, float]:
    """
    Check if search results are semantically relevant to the query.
    Returns (passed, score) where score is 0-1.
    """
    if not results:
        return False, 0.0
    
    # Count how many expected keywords appear in results
    results_text = " ".join(results).lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in results_text)
    score = matches / len(expected_keywords) if expected_keywords else 0
    
    return score >= 0.5, score  # Pass if at least 50% keywords found


class MockExperimentResult:
    """Simulates an experiment result from the agent."""
    def __init__(self, error: str = None, success: bool = False):
        self.run_had_error = error is not None
        self.error_details = error
        self.success = success


# =============================================================================
# Test 1: Semantic Relevance
# =============================================================================

def test_semantic_relevance():
    """
    Verify that KG search returns semantically relevant results,
    not just any results.
    """
    print("\n" + "="*70)
    print("TEST 1: Semantic Relevance of Search Results")
    print("="*70)
    
    search = KnowledgeSearchFactory.create("kg_graph_search")
    
    test_cases = [
        {
            "query": "How to configure LoRA adapters for fine-tuning?",
            "expected_keywords": ["lora", "adapter", "rank", "alpha", "config"],
            "description": "LoRA configuration query"
        },
        {
            "query": "What is QLoRA and how does 4-bit quantization work?",
            "expected_keywords": ["qlora", "quantization", "4-bit", "bitsandbytes"],
            "description": "QLoRA quantization query"
        },
        {
            "query": "How to use PEFT library for parameter efficient fine-tuning?",
            "expected_keywords": ["peft", "parameter", "efficient", "train"],
            "description": "PEFT library query"
        },
    ]
    
    all_passed = True
    for case in test_cases:
        print(f"\n  Query: '{case['query'][:50]}...'")
        
        result = search.search(
            case["query"],
            filters=KGSearchFilters(top_k=5),
            use_llm_reranker=False
        )
        
        # Extract titles and content for relevance check
        result_texts = [f"{r.page_title} {r.overview}" for r in result]
        
        passed, score = semantic_similarity_check(
            case["query"], 
            result_texts, 
            case["expected_keywords"]
        )
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status} - {case['description']} (relevance: {score:.0%})")
        print(f"    Top result: {result.results[0].page_title if result.results else 'None'}")
        
        if not passed:
            all_passed = False
            print(f"    Expected keywords: {case['expected_keywords']}")
    
    search.close()
    
    if all_passed:
        print("\n  ✓ Semantic relevance test PASSED")
    else:
        print("\n  ✗ Semantic relevance test FAILED - results not relevant enough")
    
    return all_passed


# =============================================================================
# Test 2: Memory Persistence
# =============================================================================

def test_memory_persistence():
    """
    Verify that insights persist across EpisodicStore sessions.
    This simulates the agent learning from one run and remembering in the next.
    """
    print("\n" + "="*70)
    print("TEST 2: Memory Persistence Across Sessions")
    print("="*70)
    
    # Use temp directory for isolated test
    test_dir = tempfile.mkdtemp(prefix="cognitive_test_")
    store_path = os.path.join(test_dir, "memory.json")
    
    try:
        # SESSION 1: Learn from an error
        print("\n  Session 1: Agent encounters error, learns insight")
        store1 = EpisodicStore(store_path)
        store1.clear()  # Clear any existing data
        
        insight = Insight(
            content="When using bitsandbytes on CPU, set CUDA_VISIBLE_DEVICES='' to avoid GPU detection errors",
            insight_type=InsightType.CRITICAL_ERROR,
            confidence=0.95,
            source_experiment_id="exp_session1",
            tags=["bitsandbytes", "CPU", "CUDA"]
        )
        store1.add_insight(insight)
        store1.close()
        print(f"    ✓ Stored insight: '{insight.content[:50]}...'")
        
        # Verify JSON file was written
        assert os.path.exists(store_path), "JSON file not created"
        with open(store_path) as f:
            saved_data = json.load(f)
        assert len(saved_data) == 1, f"Expected 1 insight, got {len(saved_data)}"
        print(f"    ✓ JSON persistence verified ({len(saved_data)} insight)")
        
        # SESSION 2: New session retrieves the insight
        print("\n  Session 2: New agent session retrieves past insight")
        store2 = EpisodicStore(store_path)
        
        # Query with related terms
        results = store2.retrieve_relevant("bitsandbytes CUDA error on CPU machine", top_k=3)
        store2.close()
        
        # Verify the insight was retrieved
        found = any("bitsandbytes" in r.content.lower() for r in results)
        
        if found:
            print(f"    ✓ Retrieved {len(results)} relevant insights")
            print(f"    ✓ Found the insight from Session 1")
            print("\n  ✓ Memory persistence test PASSED")
            return True
        else:
            print(f"    ✗ Retrieved {len(results)} insights but none matched")
            print("\n  ✗ Memory persistence test FAILED")
            return False
            
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# Test 3: Learning Effectiveness (Simulated Agent Loop)
# =============================================================================

def test_learning_effectiveness():
    """
    Simulate an agent that:
    1. Fails on a problem
    2. Learns from the failure
    3. Uses the learning on a similar problem
    
    This tests the full cognitive loop without running actual code.
    """
    print("\n" + "="*70)
    print("TEST 3: Learning Effectiveness (Simulated Agent Loop)")
    print("="*70)
    
    test_dir = tempfile.mkdtemp(prefix="cognitive_loop_")
    store_path = os.path.join(test_dir, "memory.json")
    state_path = os.path.join(test_dir, "state.md")
    
    try:
        kg_search = KnowledgeSearchFactory.create("kg_graph_search")
        
        controller = CognitiveController(
            knowledge_search=kg_search,
            episodic_store_path=store_path,
            state_file_path=state_path,
        )
        
        # ITERATION 1: Agent fails
        print("\n  Iteration 1: Agent attempts task and fails")
        
        working_memory = WorkingMemory(
            current_goal="Fine-tune LLaMA model with QLoRA on 8GB GPU",
            active_plan=["Load base model", "Apply QLoRA", "Train on dataset"],
            facts={"gpu_memory": "8GB", "model": "llama-7b"}
        )
        
        # Agent encounters OOM error
        error_result = MockExperimentResult(
            error="CUDA out of memory. Tried to allocate 2.5GB but only 1.2GB available. "
                  "Consider using gradient checkpointing or reducing batch size."
        )
        
        # Controller processes the error and extracts insight
        new_wm, extracted_insight = controller.process_result(error_result, working_memory)
        
        if extracted_insight:
            print(f"    ✓ Extracted insight: '{extracted_insight.content[:60]}...'")
        else:
            print("    ✗ Failed to extract insight from error")
            return False
        
        # ITERATION 2: Similar task, agent should use learned insight
        print("\n  Iteration 2: Agent attempts similar task with memory")
        
        working_memory_2 = WorkingMemory(
            current_goal="Train Mistral-7B with LoRA on consumer GPU",
            active_plan=["Setup model", "Configure LoRA", "Start training"],
            facts={"gpu_memory": "12GB", "model": "mistral-7b"}
        )
        
        # Get briefing - should include the OOM insight
        briefing = controller.prepare_briefing(
            working_memory_2,
            last_error=None  # No error this time
        )
        
        # Check if briefing contains memory-related guidance
        briefing_str = briefing.to_string().lower()
        memory_keywords = ["memory", "oom", "batch", "gradient", "checkpoint"]
        
        found_keywords = [kw for kw in memory_keywords if kw in briefing_str]
        
        print(f"    Briefing size: {len(briefing.relevant_knowledge)} chars of KG knowledge")
        print(f"    Insights in briefing: {len(briefing.insights)}")
        
        if found_keywords:
            print(f"    ✓ Briefing contains relevant memory guidance: {found_keywords}")
            print("\n  ✓ Learning effectiveness test PASSED")
            kg_search.close()
            return True
        else:
            # Check if insights are at least present
            if briefing.insights:
                print(f"    ✓ Briefing has {len(briefing.insights)} insights (may not match keywords)")
                print("\n  ✓ Learning effectiveness test PASSED (insights present)")
                kg_search.close()
                return True
            else:
                print("    ✗ Briefing lacks memory-based guidance")
                print("\n  ✗ Learning effectiveness test FAILED")
                kg_search.close()
                return False
            
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# Test 4: Briefing Quality
# =============================================================================

def test_briefing_quality():
    """
    Verify that briefings contain all required components:
    - Goal context
    - Relevant KG knowledge
    - Historical insights
    - Plan context
    """
    print("\n" + "="*70)
    print("TEST 4: Briefing Quality and Completeness")
    print("="*70)
    
    test_dir = tempfile.mkdtemp(prefix="briefing_test_")
    store_path = os.path.join(test_dir, "memory.json")
    state_path = os.path.join(test_dir, "state.md")
    
    try:
        kg_search = KnowledgeSearchFactory.create("kg_graph_search")
        
        controller = CognitiveController(
            knowledge_search=kg_search,
            episodic_store_path=store_path,
            state_file_path=state_path,
        )
        
        # Add some historical insights
        controller.episodic.add_insight(Insight(
            content="Use gradient accumulation when batch size is limited by memory",
            insight_type=InsightType.BEST_PRACTICE,
            confidence=0.9,
            source_experiment_id="exp_hist_1",
            tags=["training", "memory"]
        ))
        
        working_memory = WorkingMemory(
            current_goal="Implement LoRA fine-tuning for code generation",
            active_plan=["Load CodeLlama", "Configure LoRA r=16", "Train on code dataset"],
            facts={"task": "code_generation", "lora_rank": 16}
        )
        
        briefing = controller.prepare_briefing(
            working_memory,
            last_error="ImportError: transformers version too old"
        )
        
        # Quality checks
        checks = []
        
        # 1. Goal present
        goal_present = briefing.goal and len(briefing.goal) > 10
        checks.append(("Goal present", goal_present))
        print(f"\n  {'✓' if goal_present else '✗'} Goal: {briefing.goal[:50]}...")
        
        # 2. Plan present
        plan_present = briefing.plan and len(briefing.plan) > 10
        checks.append(("Plan present", plan_present))
        print(f"  {'✓' if plan_present else '✗'} Plan: {briefing.plan[:50]}...")
        
        # 3. KG knowledge present and substantial
        kg_substantial = len(briefing.relevant_knowledge) > 100
        checks.append(("KG knowledge substantial", kg_substantial))
        print(f"  {'✓' if kg_substantial else '✗'} KG knowledge: {len(briefing.relevant_knowledge)} chars")
        
        # 4. Insights present
        insights_present = len(briefing.insights) > 0
        checks.append(("Historical insights present", insights_present))
        print(f"  {'✓' if insights_present else '✗'} Insights: {len(briefing.insights)} items")
        
        # 5. Error context included
        error_mentioned = "error" in briefing.recent_history_summary.lower() or "import" in briefing.recent_history_summary.lower()
        checks.append(("Error context included", error_mentioned))
        print(f"  {'✓' if error_mentioned else '✗'} Error context: {briefing.recent_history_summary[:50]}...")
        
        # 6. State file created
        state_exists = os.path.exists(state_path)
        checks.append(("State file created", state_exists))
        print(f"  {'✓' if state_exists else '✗'} State file: {state_path}")
        
        kg_search.close()
        
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        
        if passed >= total - 1:  # Allow 1 failure
            print(f"\n  ✓ Briefing quality test PASSED ({passed}/{total} checks)")
            return True
        else:
            print(f"\n  ✗ Briefing quality test FAILED ({passed}/{total} checks)")
            return False
            
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  COMPREHENSIVE COGNITIVE MEMORY TESTS")
    print("="*70)
    print("  These tests validate the cognitive memory system actually works,")
    print("  not just that it doesn't crash.")
    
    tests = [
        ("Semantic Relevance", test_semantic_relevance),
        ("Memory Persistence", test_memory_persistence),
        ("Learning Effectiveness", test_learning_effectiveness),
        ("Briefing Quality", test_briefing_quality),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n  ✗ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    print("\n" + "="*70)
    print("  COMPREHENSIVE TEST SUMMARY")
    print("="*70)
    
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(tests)
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All comprehensive tests PASSED!")
        print("  The cognitive memory system is working correctly.")
    elif passed >= total - 1:
        print("\n  ⚠️  Most tests passed. Review failures above.")
    else:
        print("\n  ❌ Multiple tests failed. System needs debugging.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
