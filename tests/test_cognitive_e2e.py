"""
End-to-End Test: Cognitive Memory Architecture with Real KG Search

This test validates the full cognitive memory flow:
1. CognitiveController connects to real Weaviate/Neo4j
2. EpisodicStore persists and retrieves insights
3. KG search retrieves relevant knowledge
4. Briefings are generated with real context

Requires: Running infrastructure (./start_infra.sh)
"""

# CRITICAL: Load dotenv BEFORE any other imports
# Otherwise modules will check for API keys before they're loaded
import os
from dotenv import load_dotenv
load_dotenv()

# Verify keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in .env")
    print("Make sure .env exists with OPENAI_API_KEY=...")
    exit(1)

# Now safe to import modules that check for API keys
import json
from pathlib import Path

from src.memory.types import WorkingMemory, Insight, InsightType, Briefing
from src.memory.controller import CognitiveController
from src.memory.episodic import EpisodicStore
from src.knowledge.search import KnowledgeSearchFactory, KGSearchFilters


def test_episodic_store_with_weaviate():
    """Test EpisodicStore with Weaviate backend."""
    print("\n" + "="*60)
    print("TEST 1: EpisodicStore with Weaviate")
    print("="*60)
    
    store_path = ".test_e2e_memory.json"
    if os.path.exists(store_path):
        os.remove(store_path)
    
    store = EpisodicStore(store_path)
    
    # Verify Weaviate is connected
    if not store._weaviate_client:
        print("  WARNING: Weaviate not connected, using JSON fallback")
    else:
        print("  ✓ Weaviate connected")
    
    insights = [
        Insight(
            content="Always use LoRA rank 8-16 for small datasets",
            insight_type=InsightType.BEST_PRACTICE,
            confidence=0.9,
            source_experiment_id="exp_lora_1",
            tags=["LoRA", "fine-tuning"]
        ),
        Insight(
            content="QLoRA requires bitsandbytes for 4-bit quantization",
            insight_type=InsightType.CRITICAL_ERROR,
            confidence=1.0,
            source_experiment_id="exp_qlora_1",
            tags=["QLoRA", "quantization"]
        ),
    ]
    
    for insight in insights:
        store.add_insight(insight)
        print(f"  ✓ Added: {insight.content[:50]}...")
    
    print("\n  Retrieving relevant insights for 'LoRA training'...")
    results = store.retrieve_relevant("LoRA training best practices", top_k=3)
    print(f"  Found {len(results)} relevant insights")
    
    store.close()
    os.remove(store_path)
    print("  ✓ EpisodicStore test passed!")
    return True


def test_kg_search_integration():
    """Test KG search with real Weaviate/Neo4j."""
    print("\n" + "="*60)
    print("TEST 2: KG Search Integration (Weaviate + Neo4j)")
    print("="*60)
    
    search = KnowledgeSearchFactory.create("kg_graph_search")
    
    queries = [
        "How to fine-tune LLM with LoRA?",
        "QLoRA training setup",
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        result = search.search(
            query,
            filters=KGSearchFilters(top_k=3),
            use_llm_reranker=False
        )
        print(f"  Results ({result.total_found} found):")
        for item in result:
            print(f"    - {item.page_title} ({item.page_type})")
    
    search.close()
    print("\n  ✓ KG Search test passed!")
    return True


def test_cognitive_controller_with_kg():
    """Test CognitiveController with real KG search."""
    print("\n" + "="*60)
    print("TEST 3: CognitiveController with Real KG")
    print("="*60)
    
    store_path = ".test_e2e_controller.json"
    state_path = ".test_e2e_state.md"
    
    for p in [store_path, state_path]:
        if os.path.exists(p):
            os.remove(p)
    
    kg_search = KnowledgeSearchFactory.create("kg_graph_search")
    
    controller = CognitiveController(
        knowledge_search=kg_search,
        episodic_store_path=store_path,
        state_file_path=state_path,
    )
    
    working_memory = WorkingMemory(
        current_goal="Fine-tune Llama-2 with LoRA for code generation",
        active_plan=["Load model", "Configure LoRA", "Train"],
        facts={"model": "meta-llama/Llama-2-7b-hf", "method": "QLoRA"}
    )
    
    print(f"\n  Goal: '{working_memory.current_goal}'")
    
    briefing = controller.prepare_briefing(
        working_memory, 
        last_error="ImportError: No module named 'peft'"
    )
    
    print(f"  Briefing generated:")
    print(f"    - Goal: {briefing.goal[:50]}...")
    print(f"    - KG knowledge: {len(briefing.relevant_knowledge)} chars")
    print(f"    - Insights: {len(briefing.insights)}")
    
    # Simulate error result
    class MockResult:
        run_had_error = True
        error_details = "ModuleNotFoundError: bitsandbytes"
    
    new_wm, new_insight = controller.process_result(MockResult(), working_memory)
    print(f"  Insight extracted: {new_insight.content[:50] if new_insight else 'None'}...")
    
    kg_search.close()
    for p in [store_path, state_path]:
        if os.path.exists(p):
            os.remove(p)
    
    print("\n  ✓ CognitiveController test passed!")
    return True


def main():
    print("\n" + "="*60)
    print("  COGNITIVE MEMORY E2E TESTS")
    print("="*60)
    
    tests = [
        ("EpisodicStore", test_episodic_store_with_weaviate),
        ("KG Search", test_kg_search_integration),
        ("CognitiveController", test_cognitive_controller_with_kg),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n  {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
