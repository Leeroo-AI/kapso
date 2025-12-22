import logging
import json
import os
from typing import Optional, List, Dict, Any, Tuple
from copy import deepcopy

from src.core.llm import LLMBackend
from src.knowledge.search.base import KnowledgeSearch
from src.memory.types import WorkingMemory, Briefing, Insight, InsightType
from src.memory.episodic import EpisodicStore

logger = logging.getLogger(__name__)

class CognitiveController:
    """
    Intelligent mediator between the Orchestrator, Knowledge, and History.
    Implements the 'Meta-Cognition' loop: Reflect -> Retrieve -> Synthesize.
    """
    
    def __init__(
        self, 
        knowledge_search: Optional[KnowledgeSearch] = None,
        episodic_store_path: str = ".memory_store.json",
        state_file_path: str = ".praxium_state.md",
        llm_model: str = "gpt-4o-mini"
    ):
        self.kg = knowledge_search
        self.episodic = EpisodicStore(episodic_store_path)
        self.state_file_path = state_file_path
        self.llm = LLMBackend()
        self.model = llm_model
        
    def prepare_briefing(
        self, 
        working_memory: WorkingMemory, 
        last_error: Optional[str] = None
    ) -> Briefing:
        """
        Constructs a focused briefing for the agent based on current state.
        """
        # Sync state to file
        self._save_state_to_file(working_memory)

        # 1. Meta-Cognition: Generate Search Queries
        query_context = f"Goal: {working_memory.current_goal}\nPlan: {working_memory.active_plan}"
        if last_error:
            query_context += f"\nLastError: {last_error}"
            
        search_query = self._generate_search_query(query_context)
        
        # 2. Retrieval
        kg_text = ""
        if self.kg and self.kg.is_enabled():
            kg_result = self.kg.search(search_query, context=last_error)
            kg_text = kg_result.to_context_string()

        # B. Episodic Knowledge (Weaviate/JSON)
        episodes = self.episodic.retrieve_relevant(search_query)
        insights = [e.content for e in episodes]
        
        return Briefing(
            goal=working_memory.current_goal,
            plan="\n".join(working_memory.active_plan),
            insights=insights,
            relevant_knowledge=kg_text,
            recent_history_summary=f"Last Error: {last_error}" if last_error else "No recent errors."
        )

    def process_result(
        self, 
        experiment_result: Any, 
        working_memory: WorkingMemory
    ) -> Tuple[WorkingMemory, Optional[Insight]]:
        """
        Reflects on the result, updates working memory, and extracts new insights.
        """
        # 1. Extract Insight
        new_insight = None
        if experiment_result.run_had_error:
            new_insight = self._extract_insight(experiment_result)
            if new_insight:
                self.episodic.add_insight(new_insight)
                
        # 2. Update Working Memory
        new_memory = deepcopy(working_memory)
        if experiment_result.run_had_error:
             new_memory.facts["last_failure"] = str(experiment_result.error_details)[:200]
        
        # Save updated state to file
        self._save_state_to_file(new_memory)
             
        return new_memory, new_insight

    def _save_state_to_file(self, memory: WorkingMemory):
        """Persist the working memory to a markdown file for visibility."""
        content = f"""# 🧠 Agent Cognitive State

## Current Goal
{memory.current_goal}

## Active Plan
{chr(10).join(f"- [ ] {step}" for step in memory.active_plan) if memory.active_plan else "- (No plan yet)"}

## Knowledge / Facts
```json
{json.dumps(memory.facts, indent=2)}
```
"""
        try:
            with open(self.state_file_path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to save state file: {e}")

    def _generate_search_query(self, context: str) -> str:
        prompt = f"""
        Given the following context (Goal, Plan, and potentially an Error),
        generate a single, specific search query to find relevant documentation or fix instructions.
        
        Context:
        {context}
        
        Return ONLY the search query string.
        """
        try:
            return self.llm.llm_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            ).strip()
        except Exception:
            return context.split("\n")[0]

    def _extract_insight(self, result: Any) -> Optional[Insight]:
        if not result.run_had_error:
            return None
            
        error_msg = str(result.error_details)
        if len(error_msg) > 1000:
            error_msg = error_msg[:1000] + "..."
            
        prompt = f"""
        Analyze this error from a software engineering experiment.
        Extract a SINGLE, concise, generalized rule or insight that would prevent this error in the future.
        
        Error:
        {error_msg}
        
        Format as JSON:
        {{
            "rule": "Do not use X, use Y instead",
            "type": "critical_error",
            "confidence": 0.9
        }}
        """
        
        try:
            response = self.llm.llm_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response)
            
            return Insight(
                content=data["rule"],
                insight_type=InsightType(data["type"]),
                confidence=data["confidence"],
                source_experiment_id="exp_unknown",
                tags=["auto-generated"]
            )
        except Exception as e:
            logger.warning(f"Failed to extract insight: {e}")
            return None
