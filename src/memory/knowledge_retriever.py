# =============================================================================
# Knowledge Retriever - Unified retrieval exploiting wiki structure
# =============================================================================
#
# Wiki Structure (from developer_docs/wiki_page_structure.md):
#   Workflow
#     ├── uses_heuristic → Heuristic (workflow-level)
#     └── step → Principle
#                  ├── implemented_by → Implementation (mandatory)
#                  │                      ├── requires_env → Environment
#                  │                      └── uses_heuristic → Heuristic
#                  └── uses_heuristic → Heuristic (principle-level)
#
# RETRIEVAL STRATEGY:
#   TIER 1: Exact workflow match → graph traversal for full nested knowledge
#   TIER 2: No workflow → return relevant Principles (no fake workflow)
#   TIER 3: On error → add error-specific heuristics to existing knowledge
#
# KEY DESIGN:
#   - Semantic search finds entry point (Workflow or Principles)
#   - Graph traversal gets the FULL linked knowledge (no truncation)
#   - KGKnowledge is the unified output for all tiers
# =============================================================================

import logging
from src.knowledge.search.base import KGSearchFilters, KGResultItem
import re
from pathlib import Path
from typing import Optional, List, Dict, TYPE_CHECKING

# Import new KG types
from src.memory.kg_types import (
    KGKnowledge, KGTier, Workflow, WorkflowStep,
    Principle, Implementation, Heuristic, Environment,
)
from src.memory.config import get_config

if TYPE_CHECKING:
    from src.knowledge.search.base import KnowledgeSearch
    from src.core.llm import LLMBackend

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Tiered knowledge retrieval that exploits wiki structure.
    
    The wiki has workflows with steps, each step linking to principles/heuristics.
    This retriever:
    1. Finds matching workflows
    2. Parses step structure
    3. Extracts linked heuristics PER STEP (not as afterthought)
    4. Falls back to synthesis if no exact match
    
    KEY: Heuristics are part of the workflow from the start, not added on errors.
    OPTIMIZATION: Uses SINGLE batch query for all linked heuristics instead of N queries.
    """
    
    WORKFLOW_MATCH_THRESHOLD = 0.7
    SYNTHESIZE_CONFIDENCE = 0.6
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(self, knowledge_search: Optional["KnowledgeSearch"] = None, 
                 llm: Optional["LLMBackend"] = None):
        self.kg = knowledge_search
        self._llm = llm
        # Cache for heuristics to avoid repeated lookups
        self._heuristic_cache: Dict[str, str] = {}

    # =========================================================================
    # Small shared utilities (used by Tier 1/2/3)
    # =========================================================================
    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        Extract code blocks from wiki/markdown content.
        
        Why this exists:
        - Implementation/Heuristic pages embed code samples.
        - We pass these through as-is (no truncation) so the agent can use them.
        
        Supported formats:
        - MediaWiki: <syntaxhighlight lang="python"> ... </syntaxhighlight>
        - Markdown: ```python ... ```
        """
        if not content:
            return []
        
        blocks: List[str] = []
        
        # MediaWiki syntaxhighlight blocks.
        blocks.extend(
            re.findall(
                r"<syntaxhighlight[^>]*>(.*?)</syntaxhighlight>",
                content,
                re.DOTALL | re.IGNORECASE,
            )
        )
        
        # Markdown fenced code blocks.
        blocks.extend(re.findall(r"```[^\n]*\n(.*?)```", content, re.DOTALL))
        
        # Keep only non-empty blocks.
        return [b.strip("\n") for b in blocks if b and b.strip()]
    
    def _get_llm(self) -> "LLMBackend":
        if self._llm is None:
            from src.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    # =========================================================================
    # KGKnowledge-based retrieval (single source of truth)
    # =========================================================================
    
    def retrieve_knowledge(
        self,
        goal: str,
        existing_knowledge: Optional[KGKnowledge] = None,
        last_error: Optional[str] = None,
        exclude_workflow: Optional[str] = None,
    ) -> KGKnowledge:
        """
        Retrieve knowledge using tiered strategy, returning KGKnowledge.
        
        This is the NEW interface that returns properly nested wiki structure.
        
        Args:
            goal: The goal to achieve
            existing_knowledge: Existing knowledge (for TIER 3 to add to)
            last_error: Last error message (triggers TIER 3)
            exclude_workflow: Workflow title to exclude (for pivot)
            
        Returns:
            KGKnowledge with full nested structure (no truncation)
        """
        if not self.kg or not self.kg.is_enabled():
            logger.info("KG not available, returning empty knowledge")
            return KGKnowledge(tier=KGTier.TIER2_RELEVANT)
        
        # TIER 3: Error-targeted retrieval (adds to existing knowledge)
        if last_error and existing_knowledge:
            return self._tier3_add_error_knowledge(goal, last_error, existing_knowledge)
        
        # TIER 1: Exact workflow match
        knowledge = self._tier1_build_knowledge(goal, exclude_workflow)
        if knowledge.workflow:
            return knowledge
        
        # TIER 2: Relevant knowledge (no fake workflow)
        return self._tier2_build_knowledge(goal)
    
    def _tier1_build_knowledge(
        self, 
        goal: str, 
        exclude_workflow: Optional[str] = None
    ) -> KGKnowledge:
        """
        TIER 1: Find exact workflow match and build full nested KGKnowledge.
        
        Uses graph traversal to get the complete structure:
        Workflow → Steps → Principles → Implementations → Heuristics
        """
        # Generate varied queries to find the best workflow match
        queries = self._llm_generate_search_queries(goal, query_type="workflow")
        if not queries:
            raise RuntimeError("TIER 1 requires LLM-generated workflow queries, but none were produced.")
            
        logger.info(f"TIER 1: Searching with {len(queries)} queries: {queries}")
        
        best_match = None
        best_score = 0.0
        query_used = ""
        
        for query in queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=5,
                    page_types=["Workflow"],
                    min_score=0.5,
                ),
            )
            if kg_result.results:
                top = kg_result.results[0]
                logger.info(
                    f"TIER 1: Top hit for query='{query}': "
                    f"{top.page_title} ({top.page_type}) score={top.score:.2f}"
                )
            
            for item in kg_result.results:
                # Skip excluded workflow (for pivot)
                if exclude_workflow and item.page_title == exclude_workflow:
                    continue
                
                # Check for workflow match
                if item.page_type == "Workflow" and item.score >= self.WORKFLOW_MATCH_THRESHOLD:
                    # If we found a significantly better match, take it
                    if item.score > best_score:
                        best_match = item
                        best_score = item.score
                        query_used = query
            
            # If we found a very strong match, stop searching
            if best_match and best_score > 0.85:
                break
        
        if best_match:
            workflow = self._build_workflow_from_graph(best_match)
            if workflow and workflow.steps:
                logger.info(
                    f"TIER 1: Found workflow '{best_match.page_title}' "
                    f"with {len(workflow.steps)} steps (score: {best_score:.2f}, query_used='{query_used}')"
                )
                return KGKnowledge(
                    tier=KGTier.TIER1_EXACT,
                    confidence=best_score,
                    query_used=query_used,
                    source_pages=[best_match.id],
                    workflow=workflow,
                )
        
        # No workflow found
        return KGKnowledge(tier=KGTier.TIER2_RELEVANT, query_used=" | ".join(queries))
    
    def _build_workflow_from_graph(self, workflow_item) -> Optional[Workflow]:
        """
        Build complete Workflow structure from graph traversal.
        
        Traverses: Workflow → STEP → Principle → IMPLEMENTED_BY → Implementation
        Also gets heuristics at each level.
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            return None
        
        workflow_id = workflow_item.id
        steps = []
        workflow_heuristics = []
        
        try:
            with self.kg._neo4j_driver.session() as session:
                # Get all linked data in one query
                query = """
                    MATCH (w:WikiPage {id: $workflow_id})-[:STEP]->(p:WikiPage)
                    OPTIONAL MATCH (p)-[:IMPLEMENTED_BY]->(impl:WikiPage)
                    OPTIONAL MATCH (impl)-[:REQUIRES_ENV]->(env:WikiPage)
                    OPTIONAL MATCH (p)-[:USES_HEURISTIC]->(hp:WikiPage)
                    OPTIONAL MATCH (impl)-[:USES_HEURISTIC]->(hi:WikiPage)
                    OPTIONAL MATCH (w)-[:USES_HEURISTIC]->(hw:WikiPage)
                    RETURN 
                        p.id AS principle_id,
                        p.page_title AS principle_title,
                        impl.id AS impl_id,
                        impl.page_title AS impl_title,
                        env.id AS env_id,
                        env.page_title AS env_title,
                        collect(DISTINCT {id: hp.id, title: hp.page_title}) AS principle_heuristics,
                        collect(DISTINCT {id: hi.id, title: hi.page_title}) AS impl_heuristics,
                        collect(DISTINCT {id: hw.id, title: hw.page_title}) AS workflow_heuristics
                """
                result = session.run(query, workflow_id=workflow_id)
                
                seen_principles = set()
                step_number = 0
                
                for record in result:
                    principle_id = record["principle_id"]
                    if not principle_id or principle_id in seen_principles:
                        continue
                    seen_principles.add(principle_id)
                    step_number += 1
                    
                    # Build Implementation (if exists)
                    implementations = []
                    if record["impl_id"]:
                        impl_content = self._fetch_page_content(record["impl_id"])
                        impl_heuristics = self._build_heuristics(record["impl_heuristics"])
                        
                        # Build Environment
                        environment = None
                        if record["env_id"]:
                            env_content = self._fetch_page_content(record["env_id"])
                            environment = Environment(
                                id=record["env_id"],
                                title=record["env_title"] or "",
                                overview=env_content.get("overview", ""),
                                content=env_content.get("content", ""),
                                requirements=env_content.get("overview", ""),
                            )
                        
                        implementations.append(Implementation(
                            id=record["impl_id"],
                            title=record["impl_title"] or "",
                            overview=impl_content.get("overview", ""),
                            content=impl_content.get("content", ""),
                            code_snippets=impl_content.get("code_snippets", []),
                            environment=environment,
                            heuristics=impl_heuristics,
                        ))
                    
                    # Build Principle heuristics
                    principle_heuristics = self._build_heuristics(record["principle_heuristics"])
                    
                    # Fetch Principle content
                    principle_content = self._fetch_page_content(principle_id)
                    
                    # Build Principle
                    principle = Principle(
                        id=principle_id,
                        title=record["principle_title"] or "",
                        overview=principle_content.get("overview", ""),
                        content=principle_content.get("content", ""),
                        implementations=implementations,
                        heuristics=principle_heuristics,
                    )
                    
                    # Build WorkflowStep
                    steps.append(WorkflowStep(
                        number=step_number,
                        principle=principle,
                    ))
                    
                    # Collect workflow-level heuristics (only once)
                    if not workflow_heuristics:
                        workflow_heuristics = self._build_heuristics(record["workflow_heuristics"])
                
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
            return None
        
        if not steps:
            return None
        
        return Workflow(
            id=workflow_id,
            title=workflow_item.page_title,
            overview=workflow_item.overview or "",
            content=workflow_item.content or "",
            source="kg_exact",
            confidence=workflow_item.score,
            steps=steps,
            heuristics=workflow_heuristics,
        )
    
    def _fetch_page_content(self, page_id: str) -> Dict[str, any]:
        """
        Fetch FULL page content from Weaviate (no truncation).
        """
        if not self.kg:
            return {}
        
        # Extract title from ID
        page_title = page_id.split("/")[-1].replace("_", " ") if "/" in page_id else page_id
        
        page = self.kg.get_page(page_title)
        if page:
            return {
                "overview": page.overview or "",
                "content": page.content or "",  # FULL content
                "code_snippets": self._extract_code_blocks(page.content) if page.content else [],
            }
        
        # Try alternate title format
        alt_title = page_id.replace("_", " ").split("/")[-1] if "/" in page_id else page_id
        page = self.kg.get_page(alt_title)
        if page:
            return {
                "overview": page.overview or "",
                "content": page.content or "",
                "code_snippets": self._extract_code_blocks(page.content) if page.content else [],
            }
        
        return {}
    
    def _build_heuristics(self, heuristic_records: List[Dict]) -> List[Heuristic]:
        """
        Build list of Heuristic objects from Neo4j records.
        Fetches FULL content for each heuristic.
        """
        heuristics = []
        for h in heuristic_records:
            if not h.get("id"):
                continue
            content = self._fetch_page_content(h["id"])
            heuristics.append(Heuristic(
                id=h.get("id", ""),
                title=h.get("title", ""),
                overview=content.get("overview", ""),
                content=content.get("content", ""),
                code_snippets=content.get("code_snippets", []),
            ))
        return heuristics
    
    def _tier2_build_knowledge(self, goal: str) -> KGKnowledge:
        """
        TIER 2: No workflow found - return relevant Principles.
        
        NO fake workflow. Just return the relevant knowledge directly.
        """
        logger.info("TIER 2: No workflow found, searching for relevant Principles")

        # Generate varied queries to improve recall when goal phrasing doesn't match
        # the KG terminology for Principles.
        queries = self._llm_generate_search_queries(goal, query_type="principle")
        if not queries:
            raise RuntimeError("TIER 2 requires LLM-generated principle queries, but none were produced.")

        logger.info(f"TIER 2: Searching with {len(queries)} queries: {queries}")

        # Search for Principles across queries and dedupe by id, keeping best score.
        best_by_id: Dict[str, KGResultItem] = {}
        for query in queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=10,
                    page_types=["Principle"],
                    min_score=0.4,
                ),
            )
            if kg_result.results:
                top = kg_result.results[0]
                logger.info(
                    f"TIER 2: Top hit for query='{query}': "
                    f"{top.page_title} ({top.page_type}) score={top.score:.2f}"
                )

            for item in kg_result.results:
                if item.page_type != "Principle":
                    continue
                prev = best_by_id.get(item.id)
                if prev is None or item.score > prev.score:
                    best_by_id[item.id] = item

        items = sorted(best_by_id.values(), key=lambda x: x.score, reverse=True)

        principles: List[Principle] = []
        for item in items:
            # Graph traverse to get linked Implementation and Heuristics
            linked = self._traverse_principle_for_knowledge(item.id)

            principle_content = self._fetch_page_content(item.id)
            principles.append(
                Principle(
                    id=item.id,
                    title=item.page_title,
                    overview=principle_content.get("overview", "") or item.overview or "",
                    content=principle_content.get("content", ""),
                    implementations=linked.get("implementations", []),
                    heuristics=linked.get("heuristics", []),
                )
            )

        if not principles:
            logger.info("TIER 2: No Principles found")
        else:
            logger.info(f"TIER 2: Found {len(principles)} relevant Principles")

        confidence = items[0].score if items else 0.0
        return KGKnowledge(
            tier=KGTier.TIER2_RELEVANT,
            confidence=confidence,
            query_used=" | ".join(queries),
            source_pages=[item.id for item in items],
            principles=principles,
        )
    
    def _traverse_principle_for_knowledge(self, principle_id: str) -> Dict[str, List]:
        """
        Traverse graph from Principle to get linked Implementations and Heuristics.
        Returns full objects, not just IDs.
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            return {"implementations": [], "heuristics": []}
        
        implementations = []
        heuristics = []
        
        try:
            with self.kg._neo4j_driver.session() as session:
                query = """
                    MATCH (p:WikiPage {id: $principle_id})
                    OPTIONAL MATCH (p)-[:IMPLEMENTED_BY]->(impl:WikiPage)
                    OPTIONAL MATCH (impl)-[:REQUIRES_ENV]->(env:WikiPage)
                    OPTIONAL MATCH (p)-[:USES_HEURISTIC]->(h:WikiPage)
                    OPTIONAL MATCH (impl)-[:USES_HEURISTIC]->(hi:WikiPage)
                    RETURN 
                        impl.id AS impl_id,
                        impl.page_title AS impl_title,
                        env.id AS env_id,
                        env.page_title AS env_title,
                        collect(DISTINCT {id: h.id, title: h.page_title}) AS principle_heuristics,
                        collect(DISTINCT {id: hi.id, title: hi.page_title}) AS impl_heuristics
                """
                records = session.run(query, principle_id=principle_id)
                
                for record in records:
                    # Build Implementation
                    if record["impl_id"]:
                        impl_content = self._fetch_page_content(record["impl_id"])
                        impl_heuristics = self._build_heuristics(record["impl_heuristics"])
                        
                        environment = None
                        if record["env_id"]:
                            env_content = self._fetch_page_content(record["env_id"])
                            environment = Environment(
                                id=record["env_id"],
                                title=record["env_title"] or "",
                                overview=env_content.get("overview", ""),
                                content=env_content.get("content", ""),
                                requirements=env_content.get("overview", ""),
                            )
                        
                        implementations.append(Implementation(
                            id=record["impl_id"],
                            title=record["impl_title"] or "",
                            overview=impl_content.get("overview", ""),
                            content=impl_content.get("content", ""),
                            code_snippets=impl_content.get("code_snippets", []),
                            environment=environment,
                            heuristics=impl_heuristics,
                        ))
                    
                    # Build Principle-level heuristics
                    heuristics = self._build_heuristics(record["principle_heuristics"])
                    break  # Only one record expected
                    
        except Exception as e:
            logger.warning(f"Principle graph traversal failed: {e}")
        
        return {"implementations": implementations, "heuristics": heuristics}
    
    def _tier3_add_error_knowledge(
        self,
        goal: str,
        error: str,
        existing_knowledge: KGKnowledge,
    ) -> KGKnowledge:
        """
        TIER 3: Add error-specific knowledge to existing.
        
        1. LLM generates search queries based on error + goal
        2. Search KG with those queries
        3. Add results to existing knowledge
        """
        logger.info("TIER 3: Adding error-specific knowledge")
        
        # Step 1: LLM generates search queries for this error
        search_queries = self._llm_generate_search_queries(goal, error, query_type="error")

        logger.info(f"TIER 3: LLM generated {len(search_queries)} search queries")
        logger.info(f"TIER 3: Queries: {search_queries}")
        
        # Step 2: Search KG with each query
        error_heuristics = []
        alternative_implementations = []
        seen_ids = set()
        
        for query in search_queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=5,
                    # Include Environment pages so dependency / setup errors can retrieve
                    # install/requirements guidance (critical for ImportError cases).
                    page_types=["Heuristic", "Implementation", "Environment"],
                ),
                use_llm_reranker=False,
            )
            if kg_result.results:
                top = kg_result.results[0]
                logger.info(
                    f"TIER 3: Top hit for query='{query}': "
                    f"{top.page_title} ({top.page_type}) score={top.score:.2f}"
                )
            
            for item in kg_result.results:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                
                content = self._fetch_page_content(item.id)
                
                if item.page_type == "Heuristic":
                    error_heuristics.append(Heuristic(
                        id=item.id,
                        title=item.page_title,
                        overview=content.get("overview", "") or item.overview or "",
                        content=content.get("content", ""),
                        code_snippets=content.get("code_snippets", []),
                    ))
                elif item.page_type == "Implementation":
                    alternative_implementations.append(Implementation(
                        id=item.id,
                        title=item.page_title,
                        overview=content.get("overview", "") or item.overview or "",
                        content=content.get("content", ""),
                        code_snippets=content.get("code_snippets", []),
                    ))
                elif item.page_type == "Environment":
                    # KGKnowledge doesn't have a dedicated Tier3 Environment bucket.
                    # We treat environment/setup pages as heuristics so they show up
                    # prominently in the error-recovery section.
                    error_heuristics.append(Heuristic(
                        id=item.id,
                        title=f"Environment: {item.page_title}",
                        overview=content.get("overview", "") or item.overview or "",
                        content=content.get("content", ""),
                        code_snippets=content.get("code_snippets", []),
                    ))
        
        logger.info(f"TIER 3: Found {len(error_heuristics)} heuristics, "
                   f"{len(alternative_implementations)} implementations")
        
        # If semantic search returned nothing, fail fast.
        #
        # Why:
        # - Tier 3 is intentionally LLM-query → semantic search only.
        # - We do NOT keep legacy / keyword-based fallbacks (they hide regressions).
        if not error_heuristics and not alternative_implementations:
            raise RuntimeError("TIER 3 semantic search returned no results; refusing to silently fallback.")
        
        # Add to existing knowledge - no truncation, pages are already size-limited
        return existing_knowledge.add_error_knowledge(
            error_heuristics=error_heuristics,
            alternative_implementations=alternative_implementations,
        )
    
    def _llm_generate_search_queries(
        self, 
        goal: str, 
        error: Optional[str] = None, 
        query_type: str = "error"
    ) -> List[str]:
        """
        LLM generates search queries based on goal/error.
        
        Args:
            goal: User's goal
            error: Error message (optional)
            query_type: "workflow", "principle", or "error"
        """
        # Use _get_llm() to lazily initialize LLM if needed
        try:
            llm = self._get_llm()
        except Exception as e:
            raise RuntimeError(f"LLM unavailable for query generation (type={query_type}): {e}")
        
        system_prompt = ""
        user_prompt = ""
        
        if query_type == "workflow":
            system_prompt = """You generate search queries to find a specific Workflow page in a knowledge base.

Given a goal, generate EXACTLY 3 candidate queries to find a Workflow (recipe / implementation guide).

Hard rules:
- Each query MUST include at least one distinctive keyword from the goal (a proper noun / acronym / library name).
- Avoid generic queries like "fine-tuning language models" or "machine learning" (too broad).
- Prefer title-shaped queries that resemble wiki page titles.
- Each query should be 4-10 words.
- Return queries inside <queries> tags, one per line.

Guidance:
- One query SHOULD start with: "Workflow for ..."
- One query SHOULD start with: "Implementation Guide ..."
- One query SHOULD include the likely library/framework keyword(s) from the goal.
"""
            user_prompt = f"Goal: {goal}\n\nGenerate search queries:"
            
        elif query_type == "principle":
            system_prompt = """You generate search queries to find Principle pages (core theory/concepts) for a goal.

Given a goal, generate EXACTLY 3 candidate queries that target DISTINCT underlying concepts.

Hard rules:
- Each query MUST be specific to the goal. Do NOT output generic phrases like "fine-tuning language models".
- Each query should be 3-8 words.
- At least ONE query must include a concrete technical term / API / parameter name implied by the goal.
- At least ONE query must be about a mechanism/constraint (e.g., rank/alpha/target modules/memory).
- Return queries inside <queries> tags, one per line.
"""
            user_prompt = f"Goal: {goal}\n\nGenerate search queries:"
            
        else: # "error"
            # Limit error text only via config (never hardcode truncation).
            # We use the *tail* since tracebacks typically end with the actionable exception.
            cfg = get_config()
            max_error_chars = getattr(cfg.controller, "max_error_length", None)
            if max_error_chars is None or max_error_chars <= 0:
                error_tail = error or ""
            else:
                error_tail = (error or "")[-max_error_chars:]
            
            system_prompt = """You generate search queries to find solutions for code errors.

Given an error and goal, generate 2-3 search queries that would find helpful tips or implementations.

Rules:
- Generate queries that describe SOLUTIONS, not the error itself
- Each query should be 3-6 words
- Return queries inside <queries> tags, one per line
- Focus on actionable techniques
 - If the error indicates a missing dependency/module, at least ONE query must target installation/requirements/environment setup
"""
            user_prompt = f"""## Error
```
{error_tail}
```

## Goal
{goal}

Generate search queries to find solutions:"""

        try:
            response = llm.llm_completion_with_system_prompt(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_message=user_prompt,
            )
            queries = self._parse_queries(response)
            
            if not queries:
                raise RuntimeError(f"LLM query generation returned no queries (type={query_type}).")

            return queries
        except Exception as e:
            # Fail fast: do not silently fall back to goal-only queries.
            raise RuntimeError(f"{query_type} query generation failed: {e}")
    
    def _parse_queries(self, response: str) -> List[str]:
        """Parse LLM response to extract search queries."""
        import re
        
        match = re.search(r'<queries>(.*?)</queries>', response, re.DOTALL)
        if not match:
            return []
        
        content = match.group(1).strip()
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        return queries  # Return all queries, LLM prompt already limits to 2-3
    
    # NOTE: Legacy RetrievalResult/WorkflowState based retrieval removed.
    # The single supported interface is retrieve_knowledge() returning KGKnowledge.