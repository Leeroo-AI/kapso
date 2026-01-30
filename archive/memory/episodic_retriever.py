# =============================================================================
# Episodic Retriever - LLM-governed retrieval from episodic memory
# =============================================================================
#
# Instead of simple semantic search on error text, we use LLM to:
# 1. Understand the current context (goal, step, error type)
# 2. Formulate a smart query for episodic memory
# 3. Filter/rank retrieved insights for relevance
#
# This makes retrieval CONTEXT-AWARE and GOAL-ORIENTED.
#
# Prompts:
# - This module currently uses inline prompts.
# - (Some older prompt templates may still exist on disk, but are not loaded here.)
#
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from kapso.memory.episodic import EpisodicStore
    from kapso.memory.types import Insight
    from kapso.core.llm import LLMBackend

logger = logging.getLogger(__name__)

# Models known to support OpenAI-style JSON mode. This mirrors the DecisionMaker
# logic so we don't trigger avoidable errors (LLMBackend retries can be slow).
JSON_MODE_MODELS: Set[str] = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "gpt-4.1-mini", "gpt-4.1",
    "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
}


@dataclass
class RetrievalQuery:
    """
    A smart query formulated by the LLM for episodic retrieval.
    
    Attributes:
        primary_query: Main semantic search query
        fallback_queries: Alternative queries if primary fails
        filter_tags: Tags to filter results
        min_confidence: Minimum confidence threshold
        reasoning: Why the LLM chose this query
    """
    primary_query: str
    fallback_queries: List[str]
    filter_tags: List[str]
    min_confidence: float
    reasoning: str


@dataclass
class RankedInsight:
    """
    An insight ranked by the LLM for relevance.
    
    Attributes:
        insight: The original insight
        relevance_score: How relevant to current context (0-1)
        applicability: How the insight applies to current situation
        should_use: Whether this should be included in context
    """
    content: str
    insight_type: str
    original_confidence: float
    relevance_score: float
    applicability: str
    should_use: bool


class EpisodicRetriever:
    """
    LLM-governed retrieval from episodic memory.
    
    The LLM acts as a "librarian" that:
    1. Understands what information would help
    2. Formulates effective search queries
    3. Evaluates if retrieved insights are actually relevant
    
    This ensures we don't just dump similar errors into context,
    but provide ACTIONABLE, RELEVANT insights.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        episodic_store: "EpisodicStore",
        llm: Optional["LLMBackend"] = None,
        model: Optional[str] = None,
    ):
        self.store = episodic_store
        self._llm = llm
        self.model = model or self.DEFAULT_MODEL
    
    def _get_llm(self) -> "LLMBackend":
        if self._llm is None:
            from kapso.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    def retrieve_relevant_insights(
        self,
        goal: str,
        current_step: Optional[str] = None,
        last_error: Optional[str] = None,
        last_feedback: Optional[str] = None,
        max_insights: int = 5,
    ) -> List[RankedInsight]:
        """
        Retrieve and rank relevant insights from episodic memory.
        
        Uses LLM to:
        1. Formulate smart query based on context
        2. Retrieve candidate insights
        3. Rank/filter for actual relevance
        
        Args:
            goal: Current goal
            current_step: Current workflow step
            last_error: Most recent error (if any)
            last_feedback: Most recent evaluator feedback (if any)
            max_insights: Maximum insights to return
            
        Returns:
            List of RankedInsight objects, sorted by relevance
        """
        # Step 1: Formulate query
        query = self._formulate_query(goal, current_step, last_error, last_feedback)
        logger.debug(f"Episodic query: {query.primary_query}")
        logger.debug(f"Query reasoning: {query.reasoning}")
        
        # Step 2: Retrieve candidates
        candidates = self._retrieve_candidates(query, max_insights * 2)
        if not candidates:
            logger.debug("No episodic insights found")
            return []
        
        logger.debug(f"Retrieved {len(candidates)} candidate insights")
        
        # Step 3: Rank for relevance
        ranked = self._rank_insights(
            candidates,
            goal,
            current_step,
            last_error,
            max_insights,
        )
        
        # Filter to only relevant ones
        relevant = [r for r in ranked if r.should_use]
        logger.debug(f"Returning {len(relevant)} relevant insights")
        
        return relevant
    
    def _formulate_query(
        self,
        goal: str,
        current_step: Optional[str],
        last_error: Optional[str],
        last_feedback: Optional[str],
    ) -> RetrievalQuery:
        """Use LLM to formulate a smart query."""
        context_parts = [f"Goal: {goal}"]
        if current_step:
            context_parts.append(f"Current step: {current_step}")
        if last_error:
            context_parts.append(f"Recent error: {last_error}")
        if last_feedback:
            context_parts.append(f"Recent feedback: {last_feedback}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are helping search an episodic memory database for relevant past learnings.

## Current Context
{context}

## Task
Formulate a search query to find RELEVANT past insights.
Think about:
- What type of problem is this?
- What lessons from the past would help?
- What keywords would match useful insights?

Respond in JSON:
{{
  "primary_query": "Main search query (semantic, 10-30 words)",
  "fallback_queries": ["Alternative query 1", "Alternative query 2"],
  "filter_tags": ["tag1", "tag2"],
  "min_confidence": 0.0-1.0,
  "reasoning": "Why these queries would find helpful insights"
}}

Focus on finding ACTIONABLE, TRANSFERABLE lessons.
Respond ONLY with JSON."""

        try:
            response = self._call_llm(prompt)
            return self._parse_query_response(response)
        except Exception as e:
            logger.warning(f"Query formulation failed: {e}")
            # Fallback: simple query from context
            return self._fallback_query(goal, last_error)
    
    def _retrieve_candidates(
        self,
        query: RetrievalQuery,
        max_results: int,
    ) -> List["Insight"]:
        """Retrieve candidates from episodic store."""
        all_results = []
        
        # Try primary query
        results = self.store.retrieve_relevant(
            query.primary_query,
            top_k=max_results,
        )
        all_results.extend(results)
        
        # Try fallback queries if needed
        if len(all_results) < max_results // 2:
            for fallback in query.fallback_queries:
                more = self.store.retrieve_relevant(fallback, top_k=max_results // 2)
                for r in more:
                    if r not in all_results:
                        all_results.append(r)
        
        # Optionally filter by tags (LLM-proposed). This is a soft filter: if it
        # would eliminate everything, we keep the unfiltered set to avoid false
        # negatives due to tag mismatch.
        if query.filter_tags:
            wanted = {t.strip().lower() for t in query.filter_tags if t and t.strip()}
            if wanted:
                tag_filtered = []
                for r in all_results:
                    tags = {t.lower() for t in (getattr(r, "tags", None) or []) if t}
                    if tags & wanted:
                        tag_filtered.append(r)
                if tag_filtered:
                    all_results = tag_filtered
        
        # Filter by confidence
        filtered = [
            r for r in all_results
            if r.confidence >= query.min_confidence
        ]
        
        return filtered[:max_results]
    
    def _rank_insights(
        self,
        candidates: List["Insight"],
        goal: str,
        current_step: Optional[str],
        last_error: Optional[str],
        max_insights: int,
    ) -> List[RankedInsight]:
        """Use LLM to rank insights for relevance."""
        if not candidates:
            return []
        
        # Format candidates for LLM
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"\n[{i+1}] {c.content}"
            candidates_text += f"\n    Type: {c.insight_type.value}, Confidence: {c.confidence:.2f}"
        
        context = f"Goal: {goal}"
        if current_step:
            context += f"\nStep: {current_step}"
        if last_error:
            context += f"\nError: {last_error}"
        
        prompt = f"""You are filtering episodic memory insights for relevance.

## Current Context
{context}

## Candidate Insights
{candidates_text}

## Task
Rank each insight for relevance to the current situation.

Respond in JSON:
{{
  "rankings": [
    {{
      "index": 1,
      "relevance_score": 0.0-1.0,
      "applicability": "How this insight applies to current situation",
      "should_use": true/false
    }},
    ...
  ]
}}

Be SELECTIVE - only mark should_use=true for insights that are DIRECTLY APPLICABLE.
Respond ONLY with JSON."""

        try:
            response = self._call_llm(prompt)
            return self._parse_ranking_response(response, candidates, max_insights)
        except Exception as e:
            logger.warning(f"Insight ranking failed: {e}")
            # Fallback: use original confidence as relevance
            return [
                RankedInsight(
                    content=c.content,
                    insight_type=c.insight_type.value,
                    original_confidence=c.confidence,
                    relevance_score=c.confidence,
                    applicability="Auto-retrieved (ranking failed)",
                    should_use=c.confidence >= 0.5,
                )
                for c in candidates[:max_insights]
            ]
    
    def _supports_json_mode(self, model: str) -> bool:
        """Check if model supports JSON mode."""
        if model in JSON_MODE_MODELS:
            return True
        for known in JSON_MODE_MODELS:
            if model.startswith(known):
                return True
        return False
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM for JSON output.
        
        IMPORTANT:
        - We only request JSON mode when the model is known to support it.
          Otherwise we'd trigger LLMBackend retries/sleeps on expected errors.
        - Parsing still handles "JSON in text" defensively.
        """
        llm = self._get_llm()
        kwargs = {}
        if self._supports_json_mode(self.model):
            kwargs["response_format"] = {"type": "json_object"}
        
        return llm.llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            **kwargs,
        )
    
    def _parse_query_response(self, response: str) -> RetrievalQuery:
        import json
        import re
        
        # Handle markdown code blocks
        if "```" in response:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if match:
                response = match.group(1)
        
        data = self._safe_json_load(response)
        
        return RetrievalQuery(
            primary_query=data.get("primary_query", ""),
            fallback_queries=data.get("fallback_queries", []),
            filter_tags=data.get("filter_tags", []),
            min_confidence=float(data.get("min_confidence", 0.3)),
            reasoning=data.get("reasoning", ""),
        )
    
    def _parse_ranking_response(
        self,
        response: str,
        candidates: List["Insight"],
        max_insights: int,
    ) -> List[RankedInsight]:
        import json
        import re
        
        # Handle markdown code blocks
        if "```" in response:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if match:
                response = match.group(1)
        
        data = self._safe_json_load(response)
        rankings = data.get("rankings", [])
        
        results = []
        for r in rankings:
            idx = r.get("index", 0) - 1  # Convert to 0-indexed
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                results.append(RankedInsight(
                    content=c.content,
                    insight_type=c.insight_type.value,
                    original_confidence=c.confidence,
                    relevance_score=float(r.get("relevance_score", 0.5)),
                    applicability=r.get("applicability", "Unknown"),
                    should_use=r.get("should_use", False),
                ))
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:max_insights]
    
    def _safe_json_load(self, response: str) -> dict:
        """
        Parse a JSON object from an LLM response.
        
        Handles:
        - raw JSON
        - JSON wrapped in markdown code fences
        - JSON with surrounding text (best-effort extraction)
        """
        import json
        
        raw = (response or "").strip()
        if not raw:
            return {}
        
        # First attempt: direct parse.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Second attempt: extract the first JSON object substring.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        raise json.JSONDecodeError("Could not parse JSON from response", raw, 0)
    
    def _fallback_query(
        self,
        goal: str,
        last_error: Optional[str],
    ) -> RetrievalQuery:
        """Create fallback query when LLM fails."""
        if last_error:
            # Extract key terms from error
            query = f"error {last_error}"
        else:
            query = f"how to {goal}"
        
        return RetrievalQuery(
            primary_query=query,
            fallback_queries=[goal],
            filter_tags=[],
            min_confidence=0.3,
            reasoning="Fallback query (LLM formulation failed)",
        )

