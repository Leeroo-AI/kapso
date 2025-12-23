# =============================================================================
# Cognitive Controller
# =============================================================================
#
# Orchestrates memory retrieval and briefing generation for the agent.
# Implements the Meta-Cognition loop: Reflect -> Retrieve -> Synthesize
#
# Configuration: See cognitive_memory.yaml for all tunable parameters.
# =============================================================================

import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from copy import deepcopy

from src.core.llm import LLMBackend
from src.knowledge.search.base import KnowledgeSearch
from src.memory.types import WorkingMemory, Briefing, Insight, InsightType, ExperimentResultProtocol
from src.memory.episodic import EpisodicStore

if TYPE_CHECKING:
    from src.memory.config import CognitiveMemoryConfig

logger = logging.getLogger(__name__)


class CognitiveController:
    """
    Intelligent mediator between the Orchestrator, Knowledge, and History.
    Implements the 'Meta-Cognition' loop: Reflect -> Retrieve -> Synthesize.
    
    Configuration can be passed directly or loaded from cognitive_memory.yaml.
    
    Example:
        # Using defaults
        controller = CognitiveController(knowledge_search=kg)
        
        # With preset
        from src.memory.config import CognitiveMemoryConfig
        config = CognitiveMemoryConfig.load(preset="high_quality")
        controller = CognitiveController(knowledge_search=kg, config=config)
    """
    
    def __init__(
        self, 
        knowledge_search: Optional[KnowledgeSearch] = None,
        episodic_store_path: Optional[str] = None,
        state_file_path: Optional[str] = None,
        llm_model: Optional[str] = None,
        config: Optional["CognitiveMemoryConfig"] = None
    ):
        """
        Initialize CognitiveController.
        
        Args:
            knowledge_search: KG search backend (optional)
            episodic_store_path: Override memory file path
            state_file_path: Override state file path  
            llm_model: Override LLM model
            config: Full CognitiveMemoryConfig (or loads from YAML)
        """
        # Load config if not provided
        if config is None:
            from src.memory.config import CognitiveMemoryConfig
            config = CognitiveMemoryConfig.load()
        
        self._config = config
        
        # Controller settings (allow overrides)
        self.model = llm_model or config.controller.llm_model
        self.fallback_models = config.controller.fallback_models
        self.state_file_path = state_file_path or config.controller.state_file_path
        self.max_error_length = config.controller.max_error_length
        self.max_fact_length = config.controller.max_fact_length
        
        # Insight extraction settings
        self.insight_extraction_enabled = config.insight_extraction.enabled
        self.min_error_length = config.insight_extraction.min_error_length
        self.max_insight_length = config.insight_extraction.max_insight_length
        self.default_confidence = config.insight_extraction.default_confidence
        self.auto_tags = config.insight_extraction.auto_tags
        
        # Briefing settings
        self.max_kg_context = config.briefing.max_kg_context
        self.max_insights = config.briefing.max_insights
        self.include_plan = config.briefing.include_plan
        self.include_error_history = config.briefing.include_error_history
        
        # Initialize components
        self.kg = knowledge_search
        self.episodic = EpisodicStore(
            persist_path=episodic_store_path,
            config=config.episodic
        )
        self.llm = LLMBackend()
        
        logger.info(f"CognitiveController initialized (model: {self.model})")
        
    def prepare_briefing(
        self, 
        working_memory: WorkingMemory, 
        last_error: Optional[str] = None
    ) -> Briefing:
        """
        Constructs a focused briefing for the agent based on current state.
        
        Args:
            working_memory: Current working memory state
            last_error: Most recent error message (if any)
            
        Returns:
            Briefing with goal, plan, insights, and KG knowledge
        """
        # Sync state to file
        self._save_state_to_file(working_memory)

        # 1. Meta-Cognition: Generate Search Query
        plan_str = ", ".join(working_memory.active_plan) if working_memory.active_plan else "None"
        query_context = f"Goal: {working_memory.current_goal}\nPlan: {plan_str}"
        if last_error:
            query_context += f"\nLastError: {last_error[:self.max_error_length]}"
            
        search_query = self._generate_search_query(query_context)
        
        # 2. Retrieval from KG
        kg_text = ""
        if self.kg and self.kg.is_enabled():
            kg_result = self.kg.search(search_query, context=last_error)
            kg_text = kg_result.to_context_string()
            # Truncate if needed
            if len(kg_text) > self.max_kg_context:
                kg_text = kg_text[:self.max_kg_context] + "\n... (truncated)"

        # 3. Retrieval from Episodic Memory
        episodes = self.episodic.retrieve_relevant(search_query, top_k=self.max_insights)
        insights = [e.content for e in episodes]
        
        # 4. Build briefing
        briefing_plan_str = ""
        if self.include_plan:
            briefing_plan_str = "\n".join(working_memory.active_plan)
        
        history_str = ""
        if self.include_error_history and last_error:
            history_str = f"Last Error: {last_error[:self.max_error_length]}"
        elif not last_error:
            history_str = "No recent errors."
        
        return Briefing(
            goal=working_memory.current_goal,
            plan=briefing_plan_str,
            insights=insights,
            relevant_knowledge=kg_text,
            recent_history_summary=history_str
        )

    def process_result(
        self, 
        experiment_result: ExperimentResultProtocol, 
        working_memory: WorkingMemory,
        experiment_id: Optional[str] = None
    ) -> Tuple[WorkingMemory, Optional[Insight]]:
        """
        Reflects on the result, updates working memory, and extracts new insights.
        
        Args:
            experiment_result: Result from experiment execution
            working_memory: Current working memory
            experiment_id: Identifier for the experiment (for insight tracking)
            
        Returns:
            Tuple of (updated_working_memory, extracted_insight or None)
        """
        new_insight = None
        exp_id = experiment_id or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # 1. Extract Insight from error (if enabled)
        if self.insight_extraction_enabled and experiment_result.run_had_error:
            error_str = str(experiment_result.error_details) if experiment_result.error_details else ""
            if len(error_str) >= self.min_error_length:
                new_insight = self._extract_insight(experiment_result, exp_id)
                if new_insight:
                    self.episodic.add_insight(new_insight)
                
        # 2. Update Working Memory
        new_memory = deepcopy(working_memory)
        if experiment_result.run_had_error:
            error_details = str(experiment_result.error_details) if experiment_result.error_details else "Unknown error"
            new_memory.facts["last_failure"] = error_details[:self.max_fact_length]
        
        # Save updated state to file
        self._save_state_to_file(new_memory)
             
        return new_memory, new_insight

    def _save_state_to_file(self, memory: WorkingMemory):
        """Persist the working memory to a markdown file for visibility (atomic write)."""
        content = f"""# 🧠 Agent Cognitive State

## Current Goal
{memory.current_goal}

## Active Plan
{chr(10).join(f"- [ ] {step}" for step in memory.active_plan) if memory.active_plan else "- (No plan yet)"}

## Knowledge / Facts
```json
{json.dumps(memory.facts, indent=2)}
```

_Last updated: {datetime.now(timezone.utc).isoformat()}_
"""
        try:
            # Atomic write: write to temp file then rename
            temp_path = self.state_file_path + ".tmp"
            with open(temp_path, "w") as f:
                f.write(content)
            os.replace(temp_path, self.state_file_path)
        except Exception as e:
            logger.warning(f"Failed to save state file: {e}")

    def _generate_search_query(self, context: str) -> str:
        """Generate a search query using LLM with fallback support."""
        prompt = f"""Given the following context (Goal, Plan, and potentially an Error),
generate a single, specific search query to find relevant documentation or fix instructions.

Context:
{context}

Return ONLY the search query string."""

        # Try primary model, then fallbacks
        models_to_try = [self.model] + self.fallback_models
        
        for model in models_to_try:
            try:
                result = self.llm.llm_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                ).strip()
                return result
            except Exception as e:
                logger.warning(f"Search query generation with {model} failed ({e})")
                continue
        
        # Final fallback: use first line of context
        logger.warning("All LLM models failed, using context fallback")
        return context.split("\n")[0].replace("Goal: ", "")

    def _extract_insight(self, result: ExperimentResultProtocol, experiment_id: str) -> Optional[Insight]:
        """Extract a generalized insight from an error with fallback support."""
        if not result.run_had_error:
            return None
            
        error_msg = str(result.error_details) if result.error_details else ""
        if len(error_msg) > self.max_error_length:
            error_msg = error_msg[:self.max_error_length] + "..."
            
        prompt = f"""Analyze this error from a software engineering experiment.
Extract a SINGLE, concise, generalized rule or insight that would prevent this error in the future.
Keep the rule under {self.max_insight_length} characters.

Error:
{error_msg}

Format as JSON:
{{
    "rule": "Do not use X, use Y instead",
    "type": "critical_error",
    "confidence": 0.9
}}"""

        # Try primary model, then fallbacks
        models_to_try = [self.model] + self.fallback_models
        
        for model in models_to_try:
            try:
                response = self.llm.llm_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                data = json.loads(response)
                
                # Truncate if needed
                rule = data["rule"][:self.max_insight_length]
                
                return Insight(
                    content=rule,
                    insight_type=InsightType(data.get("type", "critical_error")),
                    confidence=data.get("confidence", self.default_confidence),
                    source_experiment_id=experiment_id,
                    tags=self.auto_tags.copy()
                )
            except Exception as e:
                logger.warning(f"Insight extraction with {model} failed ({e})")
                continue
        
        logger.warning("All LLM models failed for insight extraction")
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return self._config.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cognitive controller."""
        return {
            "model": self.model,
            "fallback_models": self.fallback_models,
            "kg_enabled": self.kg is not None and self.kg.is_enabled(),
            "episodic_stats": self.episodic.get_stats(),
            "insight_extraction_enabled": self.insight_extraction_enabled,
        }

    def close(self):
        """Clean up resources (Weaviate connections)."""
        if self.episodic:
            self.episodic.close()
