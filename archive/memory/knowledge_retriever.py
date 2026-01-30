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
from kapso.knowledge.search.base import KGSearchFilters, KGResultItem
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

# Import new KG types
from kapso.memory.kg_types import (
    KGKnowledge, KGTier, Workflow, WorkflowStep,
    Principle, Implementation, Heuristic, Environment,
)
from kapso.memory.config import get_config

if TYPE_CHECKING:
    from kapso.knowledge.search.base import KnowledgeSearch
    from kapso.core.llm import LLMBackend
    from kapso.memory.config import CognitiveMemoryConfig

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
                 llm: Optional["LLMBackend"] = None,
                 config: Optional["CognitiveMemoryConfig"] = None):
        self.kg = knowledge_search
        self._llm = llm
        # Prefer a config object passed from the controller so the entire
        # cognitive stack uses the same preset/overrides.
        self._config = config
        # Small in-process cache for page lookups.
        #
        # Why:
        # - Graph traversal touches many pages (principles, implementations,
        #   heuristics, environments).
        # - `KnowledgeSearch.get_page()` can be a network call (Weaviate).
        # - Caching keeps Tier 1/2 rendering fast without changing semantics.
        self._page_cache: Dict[str, Dict[str, any]] = {}
    
    def _get_config(self):
        """
        Get the CognitiveMemoryConfig.
        
        If the controller injected a config, use it. Otherwise fall back to the
        default config loader.
        """
        return self._config or get_config()

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
            from kapso.core.llm import LLMBackend
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
        
        cfg = self._get_config()
        
        for query in queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=cfg.controller.tier1_top_k,
                    page_types=["Workflow"],
                    min_score=cfg.controller.tier1_min_score,
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
                if item.page_type == "Workflow" and item.score >= cfg.controller.workflow_match_threshold:
                    # If we found a significantly better match, take it
                    if item.score > best_score:
                        best_match = item
                        best_score = item.score
                        query_used = query
            
            # If we found a very strong match, stop searching
            if best_match and best_score > cfg.controller.workflow_strong_match_threshold:
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
    
    def _extract_ordered_principle_ids_from_workflow_content(self, workflow_content: str) -> List[str]:
        """
        Extract ordered Principle page IDs from a Workflow page's content.
        
        Why this exists:
        - Neo4j `STEP` edges currently do not encode order.
        - The workflow markdown *does* encode order via sequential step sections.
        - We prefer the workflow-authored order for determinism and correctness.
        
        Expected link format in workflow pages:
          [[step::Principle:huggingface_peft_Base_Model_Loading]]
        
        Returns:
          A list of Principle IDs matching the Neo4j/Weaviate ID convention used
          by the typed wiki structure: "Principle/<page name>" with spaces.
        """
        if not workflow_content:
            return []
        
        ordered: List[str] = []
        for m in re.finditer(r"\[\[step::Principle:([^\]]+)\]\]", workflow_content):
            raw = (m.group(1) or "").strip()
            if not raw:
                continue
            normalized = raw.replace("_", " ")
            pid = f"Principle/{normalized}"
            if pid not in ordered:
                ordered.append(pid)
        return ordered
    
    def _build_workflow_from_graph(self, workflow_item) -> Optional[Workflow]:
        """
        Build complete Workflow structure from graph traversal.
        
        Traverses: Workflow → STEP → Principle → IMPLEMENTED_BY → Implementation
        Also gets heuristics at each level.
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            return None
        
        workflow_id = workflow_item.id
        
        try:
            with self.kg._neo4j_driver.session() as session:
                # 1) Workflow-level heuristics
                heur_query = """
                    MATCH (w:WikiPage {id: $workflow_id})-[:USES_HEURISTIC]->(hw:WikiPage)
                    RETURN collect(DISTINCT {id: hw.id, title: hw.page_title}) AS workflow_heuristics
                """
                heur_record = session.run(heur_query, workflow_id=workflow_id).single()
                workflow_heuristics = self._build_heuristics((heur_record or {}).get("workflow_heuristics", []) or [])
                
                # 2) The set of principles linked from this workflow (unordered in graph)
                steps_query = """
                    MATCH (w:WikiPage {id: $workflow_id})-[:STEP]->(p:WikiPage)
                    RETURN p.id AS principle_id, p.page_title AS principle_title
                """
                step_rows = list(session.run(steps_query, workflow_id=workflow_id))
                if not step_rows:
                    return None
                
                principle_title_by_id: Dict[str, str] = {}
                for r in step_rows:
                    pid = r.get("principle_id")
                    if pid:
                        principle_title_by_id[pid] = r.get("principle_title") or ""
                
                # 3) Determine step order from the workflow page content when possible.
                #    This is more correct than relying on Neo4j return order.
                ordered_ids = self._extract_ordered_principle_ids_from_workflow_content(workflow_item.content or "")
                if ordered_ids:
                    # Keep only principles that exist in the graph for this workflow.
                    ordered_ids = [pid for pid in ordered_ids if pid in principle_title_by_id]
                else:
                    # Fallback: stable deterministic order (not necessarily the author-intended order).
                    ordered_ids = sorted(principle_title_by_id.keys(), key=lambda pid: (principle_title_by_id.get(pid, ""), pid))
                
                steps: List[WorkflowStep] = []
                for idx, principle_id in enumerate(ordered_ids, start=1):
                    linked = self._traverse_principle_for_knowledge(principle_id)
                    principle_content = self._fetch_page_content(principle_id)
                    
                    principle = Principle(
                        id=principle_id,
                        title=principle_title_by_id.get(principle_id, ""),
                        overview=principle_content.get("overview", ""),
                        content=principle_content.get("content", ""),
                        implementations=linked.get("implementations", []),
                        heuristics=linked.get("heuristics", []),
                    )
                    steps.append(WorkflowStep(number=idx, principle=principle))
                
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
        
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
            return None
    
    def _fetch_page_content(self, page_id: str) -> Dict[str, any]:
        """
        Fetch FULL page content from Weaviate (no truncation).
        """
        if not self.kg:
            return {}
        if not page_id:
            return {}
        
        # Cache hit.
        cached = self._page_cache.get(page_id)
        if cached is not None:
            return cached
        
        # First try: direct lookup by page_id (preferred, stable).
        #
        # Note: some KnowledgeSearch backends historically only supported title
        # lookup. Our `kg_graph_search` backend supports both id and title.
        page = self.kg.get_page(page_id)
        if page and getattr(page, "id", None) == page_id:
            data = {
                "overview": page.overview or "",
                "content": page.content or "",  # FULL content
                "code_snippets": self._extract_code_blocks(page.content) if page.content else [],
            }
            self._page_cache[page_id] = data
            return data
        
        # Fallback: derive the display title from the id (works for typed wiki structure).
        page_title = page_id.split("/")[-1].replace("_", " ") if "/" in page_id else page_id
        page = self.kg.get_page(page_title)
        if page:
            data = {
                "overview": page.overview or "",
                "content": page.content or "",
                "code_snippets": self._extract_code_blocks(page.content) if page.content else [],
            }
            self._page_cache[page_id] = data
            return data
        
        self._page_cache[page_id] = {}
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
        cfg = self._get_config()
        
        best_by_id: Dict[str, KGResultItem] = {}
        for query in queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=cfg.controller.tier2_top_k,
                    page_types=["Principle"],
                    min_score=cfg.controller.tier2_min_score,
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
        
        implementations: List[Implementation] = []
        heuristics: List[Heuristic] = []
        
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
                records = list(session.run(query, principle_id=principle_id))
                if not records:
                    return {"implementations": [], "heuristics": []}
                
                # Collect principle-level heuristics across all rows (they should
                # be identical, but we dedupe defensively).
                principle_heuristic_records: List[Dict] = []
                
                # Collect implementations across rows (a Principle may have many).
                impl_records_by_id: Dict[str, Dict[str, Any]] = {}
                for record in records:
                    principle_heuristic_records.extend(record.get("principle_heuristics") or [])
                    
                    impl_id = record.get("impl_id")
                    if not impl_id:
                        continue
                    
                    entry = impl_records_by_id.get(impl_id)
                    if entry is None:
                        entry = {
                            "impl_id": impl_id,
                            "impl_title": record.get("impl_title") or "",
                            "env_id": record.get("env_id"),
                            "env_title": record.get("env_title") or "",
                            "impl_heuristics": [],
                        }
                        impl_records_by_id[impl_id] = entry
                    
                    # Merge env if we see it later.
                    if not entry.get("env_id") and record.get("env_id"):
                        entry["env_id"] = record.get("env_id")
                        entry["env_title"] = record.get("env_title") or ""
                    
                    entry["impl_heuristics"].extend(record.get("impl_heuristics") or [])
                
                # Build Principle heuristics.
                heuristics = self._build_heuristics(_dedupe_records_by_id(principle_heuristic_records))
                
                # Build Implementation objects (with env + heuristics).
                for impl_id, entry in impl_records_by_id.items():
                    impl_content = self._fetch_page_content(impl_id)
                    impl_heuristics = self._build_heuristics(_dedupe_records_by_id(entry.get("impl_heuristics") or []))
                    
                    environment = None
                    env_id = entry.get("env_id")
                    if env_id:
                        env_content = self._fetch_page_content(env_id)
                        environment = Environment(
                            id=env_id,
                            title=entry.get("env_title") or "",
                            overview=env_content.get("overview", ""),
                            content=env_content.get("content", ""),
                            requirements=env_content.get("overview", ""),
                        )
                    
                    implementations.append(
                        Implementation(
                            id=impl_id,
                            title=entry.get("impl_title") or "",
                            overview=impl_content.get("overview", ""),
                            content=impl_content.get("content", ""),
                            code_snippets=impl_content.get("code_snippets", []),
                            environment=environment,
                            heuristics=impl_heuristics,
                        )
                    )
                    
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
        
        tier3_source_pages: List[str] = []
        cfg = self._get_config()
        
        for query in search_queries:
            kg_result = self.kg.search(
                query,
                filters=KGSearchFilters(
                    top_k=cfg.controller.tier3_top_k,
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
                tier3_source_pages.append(item.id)
                
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
        #
        # Also: record Tier 3 provenance in metadata so log audits can confirm the
        # exact queries/pages used (important for PR review).
        tier3_queries_blob = " | ".join(search_queries)
        if tier3_queries_blob:
            existing_knowledge.query_used = (
                (existing_knowledge.query_used + " | " if existing_knowledge.query_used else "")
                + f"tier3_error: {tier3_queries_blob}"
            )
        if tier3_source_pages:
            for pid in tier3_source_pages:
                if pid not in existing_knowledge.source_pages:
                    existing_knowledge.source_pages.append(pid)
        
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
            cfg = self._get_config()
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
                model=self._get_config().controller.llm_model or self.DEFAULT_MODEL,
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


def _dedupe_records_by_id(records: List[Dict]) -> List[Dict]:
    """
    Dedupe Neo4j `collect(DISTINCT {id, title})`-style records by `id`.
    
    Some Cypher patterns still yield repeated maps across rows. We keep the
    first instance for stability.
    """
    seen: set[str] = set()
    deduped: List[Dict] = []
    for r in records or []:
        rid = (r or {}).get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        deduped.append(r)
    return deduped