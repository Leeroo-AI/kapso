# =============================================================================
# Knowledge Retriever - Unified retrieval exploiting wiki structure
# =============================================================================
#
# The wiki has a hierarchical structure:
#   Workflow → Steps → Principles → Implementations → Environments
#                   └→ Heuristics (tips attached to each level)
#
# RETRIEVAL STRATEGY:
#   1. SEMANTIC SEARCH: Find Workflow matching the goal
#   2. GRAPH TRAVERSAL: Once workflow found, traverse edges to get ALL linked:
#      - Principles (via STEP edge)
#      - Implementations (via IMPLEMENTED_BY edge from Principles)
#      - Heuristics (via USES_HEURISTIC edge)
#      - Environments (via REQUIRES_ENV edge from Implementations)
#
#   TIER 1: Exact workflow match → graph traversal for full knowledge
#   TIER 2: No workflow → synthesize from Principles (semantic search)
#   TIER 3: On error → targeted heuristic search
#
# KEY DESIGN: 
#   - Semantic search is ONLY for finding the workflow
#   - Once workflow is found, use GRAPH TRAVERSAL for everything else
#   - This ensures we get the CURATED knowledge path, not random semantic matches
# =============================================================================

import json
import logging
from src.knowledge.search.base import KGSearchFilters
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge.search.base import KnowledgeSearch
    from src.memory.context import WorkflowState, StepState
    from src.core.llm import LLMBackend
    from src.memory.context import WorkflowState, StepState

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    EXACT_WORKFLOW = "exact_workflow"       # Found workflow page in KG
    SYNTHESIZED_PLAN = "synthesized_plan"   # Synthesized from multiple pages
    ERROR_TARGETED = "error_targeted"       # Error-focused retrieval
    NO_RESULT = "no_result"                 # Nothing found


@dataclass
class RetrievalResult:
    """Result of a knowledge retrieval operation."""
    mode: RetrievalMode
    workflow: Optional["WorkflowState"] = None
    plan_steps: List[str] = field(default_factory=list)
    heuristics: List[str] = field(default_factory=list)  # From TIER 3 error retrieval only
    code_patterns: List[str] = field(default_factory=list)
    source_pages: List[str] = field(default_factory=list)
    query_used: str = ""
    confidence: float = 0.0


PROMPTS_DIR = Path(__file__).parent / "prompts"


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
    
    def _get_llm(self) -> "LLMBackend":
        if self._llm is None:
            from src.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    def retrieve(
        self, 
        goal: str, 
        last_error: Optional[str] = None,
        current_workflow: Optional["WorkflowState"] = None,
        exclude_workflow: Optional[str] = None,  # For pivot - exclude this workflow
    ) -> RetrievalResult:
        """
        Retrieve knowledge using tiered strategy.
        
        Args:
            goal: The goal to achieve
            last_error: Last error message (triggers TIER 3)
            current_workflow: Current workflow (for context in error retrieval)
            exclude_workflow: Workflow title to exclude (for pivot)
            
        Returns:
            RetrievalResult with workflow (steps have heuristics pre-loaded)
        """
        if not self.kg or not self.kg.is_enabled():
            logger.info("KG not available, creating minimal plan")
            return self._create_minimal_plan(goal)
        
        # TIER 3: Error-targeted retrieval (enriches current workflow)
        if last_error:
            return self._tier3_error_retrieval(goal, last_error, current_workflow)
        
        # TIER 1: Exact workflow match
        result = self._tier1_workflow_match(goal, exclude_workflow=exclude_workflow)
        if result.mode == RetrievalMode.EXACT_WORKFLOW:
            return result
        
        # TIER 2: Synthesize from pages
        return self._tier2_synthesize_plan(goal)
    
    # =========================================================================
    # TIER 1: Exact Workflow Match
    # =========================================================================
    
    def _tier1_workflow_match(self, goal: str, exclude_workflow: Optional[str] = None) -> RetrievalResult:
        """
        Find exact workflow match and parse its structure.
        Each step gets its linked heuristics loaded immediately.
        
        OPTIMIZED: Uses batch lookup for all linked heuristics in ONE query.
        """
        query = f"Workflow for: {goal}"
        kg_result = self.kg.search(
            query, 
            filters=KGSearchFilters(
                top_k=10,
                page_types=["Workflow"],
                min_score=0.5,
            ),
        )
        
        for item in kg_result.results:
            # Skip if this workflow should be excluded (pivot case)
            if exclude_workflow and item.page_title == exclude_workflow:
                logger.debug(f"Skipping excluded workflow: {item.page_title}")
                continue
                
            if item.page_type == "Workflow" and item.score >= self.WORKFLOW_MATCH_THRESHOLD:
                workflow = self._parse_workflow_with_heuristics(item)
                if workflow and workflow.steps:
                    # Just log summary - detailed logging happens in cognitive_controller
                    logger.debug(f"Workflow matched: {item.page_title} ({len(workflow.steps)} steps)")
                    
                    # All heuristics come from graph structure (via step heuristics)
                    # No separate "global heuristics" - trust the graph
                    
                    return RetrievalResult(
                        mode=RetrievalMode.EXACT_WORKFLOW,
                        workflow=workflow,
                        plan_steps=[s.title for s in workflow.steps],
                        heuristics=[],  # Heuristics are per-step, stored in workflow.steps
                        source_pages=[item.id],
                        query_used=query,
                        confidence=item.score,
                    )
        
        return RetrievalResult(mode=RetrievalMode.NO_RESULT, query_used=query)
    
    def _parse_workflow_with_heuristics(self, item) -> Optional["WorkflowState"]:
        """
        Parse workflow page extracting steps WITH their linked knowledge.
        
        UNIFIED RETRIEVAL:
        1. Parse workflow content for steps and inline links
        2. GRAPH TRAVERSAL to get: Principle → Implementation → Environment
        3. Combine into complete StepState with all linked knowledge
        
        Wiki format example:
            === Step 1: Load Base Model ===
            [[step::Principle/huggingface_peft_Base_Model_Loading]]
            Use AutoModelForCausalLM with device_map="auto"
        """
        from src.memory.context import WorkflowState, StepState
        
        content = item.content or ""
        
        # Split by step headers: === Step N: Title === or ### Step N: Title
        step_pattern = r'(?:===|###)\s*Step\s+(\d+):\s*([^=\n#]+)(?:===|###)?'
        step_matches = list(re.finditer(step_pattern, content))
        
        if not step_matches:
            logger.debug(f"No steps found in workflow: {item.page_title}")
            return None
        
        # PHASE 1: Extract all step content and linked items
        step_data = []
        all_linked_items = set()
        
        for i, match in enumerate(step_matches):
            step_num = int(match.group(1))
            step_title = match.group(2).strip()
            
            # Get content between this step and next
            start = match.end()
            end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(content)
            step_content = content[start:end]
            
            # Extract linked Principles (via [[step::...]])
            principle_links = re.findall(r'\[\[step::([^\]]+)\]\]', step_content)
            # Normalize to full IDs (e.g., "Principle/name" format)
            principle_id = None
            if principle_links:
                link = principle_links[0]
                if not link.startswith("Principle/"):
                    link = f"Principle/{link.replace('Principle:', '').replace('_', ' ')}"
                principle_id = link
            
            # Extract linked heuristics ONLY (uses:: links, NOT step:: which are Principles)
            # Bug fix: Previously captured [[step::...]] which are Principle links, not heuristics
            linked_heuristics = re.findall(r'\[\[uses(?:_heuristic)?::([^\]]+)\]\]', step_content)
            all_linked_items.update(linked_heuristics)
            
            # Extract bullet points as inline heuristics
            bullets = re.findall(r'[-*]\s+(.+)', step_content)
            inline_heuristics = [
                b.strip() for b in bullets 
                if 20 < len(b.strip()) < 200 and not b.strip().startswith('[[')
            ]
            
            # Extract description
            desc_match = re.search(r'^([^[\n*#].+)', step_content.strip())
            description = desc_match.group(1).strip() if desc_match else ""
            
            step_data.append({
                "number": step_num,
                "title": step_title,
                "principle_id": principle_id,
                "linked_heuristics": linked_heuristics,
                "inline_heuristics": inline_heuristics,
                "description": description[:200],
            })
        
        # PHASE 2: GRAPH TRAVERSAL - Get implementations, heuristics, environments
        # This uses the actual graph structure: Workflow → Principle → Implementation
        graph_data = self._traverse_workflow_graph(item.id)
        step_implementations = graph_data.get("step_implementations", {})
        step_heuristics = graph_data.get("step_heuristics", {})
        
        # Store graph_data for later use by get_implementation_context (for environments)
        self._last_graph_data = graph_data
        
        logger.debug(f"Graph traversal found {len(step_implementations)} implementations, "
                    f"{len(graph_data.get('environments', []))} environments")
        
        # PHASE 3: Build StepState objects with ALL linked knowledge
        steps = []
        for data in step_data:
            heuristics = []
            principle_id = data.get("principle_id")
            
            # Add heuristics from graph traversal (Principle → USES_HEURISTIC → Heuristic)
            # step_heuristics has TITLES from Neo4j; fetch CONTENT directly by title
            if principle_id and principle_id in step_heuristics:
                for h_title in step_heuristics[principle_id]:
                    content = self._get_heuristic_content(h_title)
                    heuristics.append(content if content else h_title)
            
            # Add resolved linked heuristics (from wiki content [[uses_heuristic::...]] links)
            for linked in data["linked_heuristics"]:
                content = self._get_heuristic_content(linked)
                if content:
                    heuristics.append(content)
            
            # Add inline heuristics
            heuristics.extend(data["inline_heuristics"])
            
            # Get implementation from graph traversal
            implementation = None
            if principle_id and principle_id in step_implementations:
                impl_info = step_implementations[principle_id]
                # Fetch full implementation content with code snippets
                impl_content = self._get_implementation_content(impl_info["id"])
                if impl_content:
                    implementation = impl_content
                else:
                    implementation = impl_info
            
            steps.append(StepState(
                number=data["number"],
                title=data["title"],
                status="pending",
                description=data["description"],
                heuristics=heuristics,  # Include ALL graph-linked heuristics
                principle_id=principle_id,
                implementation=implementation,
            ))
        
        steps.sort(key=lambda s: s.number)
        
        # Log what we found
        impl_count = sum(1 for s in steps if s.implementation)
        logger.info(f"  📚 Graph traversal: {impl_count}/{len(steps)} steps have implementations")
        
        return WorkflowState(
            id=item.id,
            title=item.page_title,
            source="kg_exact",
            confidence=item.score,
            steps=steps,
            current_step_index=0,
        )
    
    def _batch_fetch_heuristics(self, linked_names: List[str]) -> Dict[str, str]:
        """
        Batch fetch heuristics for all linked items in ONE query.
        
        Instead of N separate queries, we:
        1. Build a combined query with all names
        2. Do ONE search
        3. Match results back to the requested names
        
        Returns:
            Dict mapping linked_name -> heuristic_text
        """
        if not linked_names or not self.kg:
            return {}
        
        # Check cache first
        uncached = [n for n in linked_names if n not in self._heuristic_cache]
        
        if uncached:
            # Clean names for query
            clean_names = [
                re.sub(r'^(Principle|Heuristic):', '', name).strip()
                for name in uncached
            ]
            
            # Build combined query
            combined_query = " OR ".join(clean_names[:10])  # Limit to prevent huge queries
            
            logger.debug(f"Batch fetching {len(uncached)} heuristics in ONE query")
            
            result = self.kg.search(
                combined_query,
                filters=KGSearchFilters(
                    top_k=len(uncached) * 2,  # Get enough results
                    page_types=["Heuristic", "Principle"],
                ),
            )
            
            # Match results back to requested names
            for item in result.results:
                for original_name, clean_name in zip(uncached, clean_names):
                    if clean_name.lower() in item.page_title.lower():
                        text = item.overview or (item.content[:200] if item.content else None)
                        if text:
                            self._heuristic_cache[original_name] = text
                            break
        
        # Return all requested (cached + newly fetched)
        return {name: self._heuristic_cache.get(name, "") for name in linked_names if name in self._heuristic_cache}
    
    # NOTE: _fetch_global_heuristics removed - all heuristics now come from graph structure
    # (Workflow → Principle → USES_HEURISTIC → Heuristic)
    # This is cleaner: trust the KG structure, no separate semantic search fallback
    
    # =========================================================================
    # GRAPH TRAVERSAL - Get linked knowledge via edges
    # =========================================================================
    
    def _traverse_workflow_graph(self, workflow_id: str) -> Dict[str, any]:
        """
        Traverse the graph from a workflow to get ALL linked knowledge.
        
        Path: Workflow ──STEP──> Principle ──IMPLEMENTED_BY──> Implementation
                                     │                              │
                                     └──USES_HEURISTIC──> Heuristic │
                                                                     │
                                          └──REQUIRES_ENV──> Environment
        
        Returns:
            Dict with:
            - step_implementations: {principle_id: implementation_info}
            - step_heuristics: {principle_id: [heuristics]}
            - environments: [environment_info]
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            logger.debug("Neo4j not available for graph traversal")
            return {"step_implementations": {}, "step_heuristics": {}, "environments": []}
        
        step_implementations = {}
        step_heuristics = {}
        environments = []
        
        try:
            with self.kg._neo4j_driver.session() as session:
                # Query: Workflow → STEP → Principle → IMPLEMENTED_BY → Implementation
                # Also get REQUIRES_ENV from Implementations
                # NOTE: We only query IDs and titles from Neo4j.
                # Overviews and content are fetched from Weaviate separately.
                #
                # HEURISTICS: Can be linked from Workflow, Principle, OR Implementation
                # Per KG structure, heuristics can connect to any of these levels.
                query = """
                    MATCH (w:WikiPage {id: $workflow_id})-[:STEP]->(p:WikiPage)
                    OPTIONAL MATCH (p)-[:IMPLEMENTED_BY]->(impl:WikiPage)
                    OPTIONAL MATCH (impl)-[:REQUIRES_ENV]->(env:WikiPage)
                    
                    // Heuristics from Principle level
                    OPTIONAL MATCH (p)-[:USES_HEURISTIC]->(hp:WikiPage)
                    
                    // Heuristics from Implementation level
                    OPTIONAL MATCH (impl)-[:USES_HEURISTIC]->(hi:WikiPage)
                    
                    // Workflow-level heuristics (apply to all steps)
                    OPTIONAL MATCH (w)-[:USES_HEURISTIC]->(hw:WikiPage)
                    
                    RETURN 
                        p.id AS principle_id,
                        p.page_title AS principle_title,
                        impl.id AS impl_id,
                        impl.page_title AS impl_title,
                        env.id AS env_id,
                        env.page_title AS env_title,
                        collect(DISTINCT hp.page_title) + 
                        collect(DISTINCT hi.page_title) + 
                        collect(DISTINCT hw.page_title) AS heuristic_titles
                """
                result = session.run(query, workflow_id=workflow_id)
                
                seen_envs = set()
                for record in result:
                    principle_id = record["principle_id"]
                    
                    # Store implementation for this principle/step
                    # NOTE: Only ID and title from Neo4j; content from Weaviate later
                    if record["impl_id"]:
                        step_implementations[principle_id] = {
                            "id": record["impl_id"],
                            "title": record["impl_title"],
                        }
                    
                    # Store heuristic TITLES (content fetched from Weaviate later)
                    if record["heuristic_titles"]:
                        step_heuristics[principle_id] = [h for h in record["heuristic_titles"] if h]
                    
                    # Collect unique environments
                    if record["env_id"] and record["env_id"] not in seen_envs:
                        seen_envs.add(record["env_id"])
                        environments.append({
                            "id": record["env_id"],
                            "title": record["env_title"],
                        })
                
                # Log detailed mapping for debugging
                logger.debug(f"Graph traversal: {len(step_implementations)} implementations, "
                           f"{len(environments)} environments")
                for pid, impl in step_implementations.items():
                    principle_name = pid.split("/")[-1] if "/" in pid else pid
                    logger.debug(f"  → {principle_name} ──IMPLEMENTED_BY──> {impl['title']}")
                
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
        
        return {
            "step_implementations": step_implementations,
            "step_heuristics": step_heuristics,
            "environments": environments,
        }
    
    def _get_implementation_content(self, impl_id: str) -> Dict[str, any]:
        """
        Get full implementation page content including code snippets.
        
        Uses DIRECT FETCH by title (not semantic search).
        We already know the exact page from graph traversal.
        """
        if not self.kg:
            return {}
        
        # Extract page title from ID
        page_title = impl_id.replace("Implementation/", "").strip()
        
        # DIRECT FETCH by exact title - no semantic search needed!
        page = self.kg.get_page(page_title)
        
        if page:
            code_blocks = self._extract_code_blocks(page.content) if page.content else []
            return {
                "title": page.page_title,
                "overview": page.overview or "",
                "content": page.content[:1000] if page.content else "",
                "code_snippets": code_blocks[:3],
            }
        
        # Fallback: try with underscores replaced by spaces (ID normalization)
        alt_title = page_title.replace("_", " ")
        if alt_title != page_title:
            page = self.kg.get_page(alt_title)
            if page:
                code_blocks = self._extract_code_blocks(page.content) if page.content else []
                return {
                    "title": page.page_title,
                    "overview": page.overview or "",
                    "content": page.content[:1000] if page.content else "",
                    "code_snippets": code_blocks[:3],
                }
        
        logger.warning(f"Could not fetch implementation page: {page_title}")
        return {}
    
    def _get_heuristic_content(self, title: str) -> Optional[str]:
        """
        Get heuristic content by direct page fetch.
        
        Simple approach: We know the title from Neo4j, just fetch the page.
        Returns the overview/content, or None if not found.
        """
        if not self.kg or not title:
            return None
        
        # Check cache first
        if title in self._heuristic_cache:
            return self._heuristic_cache[title]
        
        # Clean the title (remove type prefix if present)
        clean_title = title.replace("Heuristic/", "").replace("Principle:", "").strip()
        
        # Direct fetch by title
        page = self.kg.get_page(clean_title)
        if page:
            content = page.overview or (page.content[:300] if page.content else None)
            if content:
                self._heuristic_cache[title] = content
                return content
        
        # Try with underscores replaced by spaces (ID normalization)
        alt_title = clean_title.replace("_", " ")
        if alt_title != clean_title:
            page = self.kg.get_page(alt_title)
            if page:
                content = page.overview or (page.content[:300] if page.content else None)
                if content:
                    self._heuristic_cache[title] = content
                    return content
        
        return None
    
    # =========================================================================
    # TIER 2: Synthesize Plan from Principles
    # =========================================================================
    
    def _tier2_synthesize_plan(self, goal: str) -> RetrievalResult:
        """
        Synthesize a plan from relevant Principles.
        
        IMPROVED APPROACH:
        1. Search for PRINCIPLES (the theory layer) - not all page types
        2. For each Principle, GRAPH TRAVERSE to get linked knowledge:
           - Implementation (via IMPLEMENTED_BY)
           - Heuristics (via USES_HEURISTIC)
           - Environment (via REQUIRES_ENV)
        3. LLM orders Principles into a workflow sequence
        4. Each step = 1 Principle + its curated linked knowledge
        
        This ensures synthesized workflows have the same quality as exact matches.
        """
        # Step 1: Search for PRINCIPLES specifically
        kg_result = self.kg.search(
            goal, 
            filters=KGSearchFilters(
                top_k=10, 
                page_types=["Principle"],  # Focus on theory layer
                min_score=0.4,
            ),
        )
        
        if not kg_result.results:
            return self._create_minimal_plan(goal)
        
        # Step 2: Graph traverse from each Principle to get linked knowledge
        principle_data = []
        for item in kg_result.results:
            if item.page_type != "Principle":
                continue
            
            # Get linked Implementation, Heuristics, Environment via graph
            linked = self._traverse_principle_graph(item.id)
            
            principle_data.append({
                "id": item.id,
                "title": item.page_title,
                "overview": item.overview or "",
                "implementation": linked.get("implementation"),
                "heuristics": linked.get("heuristics", []),
                "environment": linked.get("environment"),
            })
        
        if not principle_data:
            return self._create_minimal_plan(goal)
        
        # Step 3: LLM orders Principles into workflow sequence
        principles_summary = "\n".join([
            f"- {p['title']}: {p['overview'][:100]}"
            for p in principle_data[:7]
        ])
        ordered_indices = self._order_principles_with_llm(goal, principles_summary, len(principle_data))
        
        # Step 4: Build steps from ordered Principles with their linked knowledge
        from src.memory.context import WorkflowState, StepState
        
        steps = []
        for i, idx in enumerate(ordered_indices):
            if idx >= len(principle_data):
                continue
            p = principle_data[idx]
            
            steps.append(StepState(
                number=i + 1,
                title=p["title"].split("/")[-1].replace("_", " "),  # Clean title
                status="pending",
                description=p["overview"][:200],
                principle_id=p["id"],
                implementation=p["implementation"],
                heuristics=p["heuristics"],  # Curated from graph!
            ))
        
        if not steps:
            return self._create_minimal_plan(goal)
        
        workflow = WorkflowState(
            id=f"Synthesized/{goal[:30].replace(' ', '_')}",
            title=f"Plan: {goal[:50]}",
            source="kg_synthesized",
            confidence=self.SYNTHESIZE_CONFIDENCE,
            steps=steps,
        )
        
        # Collect all heuristics for the result
        all_heuristics = []
        for step in steps:
            all_heuristics.extend(step.heuristics)
        
        return RetrievalResult(
            mode=RetrievalMode.SYNTHESIZED_PLAN,
            workflow=workflow,
            plan_steps=[s.title for s in steps],
            heuristics=all_heuristics,
            source_pages=[p["id"] for p in principle_data[:5]],
            query_used=goal,
            confidence=self.SYNTHESIZE_CONFIDENCE,
        )
    
    def _traverse_principle_graph(self, principle_id: str) -> Dict[str, any]:
        """
        Traverse graph from a single Principle to get linked knowledge.
        
        Returns:
            Dict with implementation, heuristics, environment
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            return {}
        
        result = {"implementation": None, "heuristics": [], "environment": None}
        
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
                        env.page_title AS env_title,
                        collect(DISTINCT h.page_title) + collect(DISTINCT hi.page_title) AS heuristic_titles
                """
                records = session.run(query, principle_id=principle_id)
                
                for record in records:
                    if record["impl_id"]:
                        impl_content = self._get_implementation_content(record["impl_id"])
                        result["implementation"] = impl_content if impl_content else {
                            "id": record["impl_id"],
                            "title": record["impl_title"],
                        }
                    
                    if record["env_title"]:
                        result["environment"] = {"title": record["env_title"]}
                    
                    # Fetch heuristic content
                    for h_title in record["heuristic_titles"]:
                        if h_title:
                            content = self._get_heuristic_content(h_title)
                            if content:
                                result["heuristics"].append(content)
                    break  # Only one record expected
                    
        except Exception as e:
            logger.warning(f"Principle graph traversal failed: {e}")
        
        return result
    
    def _order_principles_with_llm(self, goal: str, principles: str, count: int) -> List[int]:
        """Use LLM to order Principles into logical workflow sequence."""
        prompt = f"""Given this goal: {goal}

And these available Principles (concepts):
{principles}

Order them into a logical workflow sequence (first to last).
Return ONLY a JSON array of indices (0-indexed), e.g. [2, 0, 1, 3]
Include only the most relevant principles (max 6).
"""
        try:
            response = self._get_llm().llm_completion(
                model=self.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            indices = json.loads(response)
            return [int(i) for i in indices if isinstance(i, (int, float)) and 0 <= i < count][:6]
        except Exception as e:
            logger.warning(f"LLM ordering failed: {e}")
            return list(range(min(count, 5)))  # Fallback: use first 5 in order
    
    def _synthesize_with_llm(self, goal: str, knowledge: str) -> List[str]:
        """Use LLM to synthesize a step-by-step plan."""
        prompt_file = PROMPTS_DIR / "synthesize_plan.md"
        if not prompt_file.exists():
            return []
        
        prompt = prompt_file.read_text().replace("{goal}", goal).replace("{knowledge_pages}", knowledge)
        
        try:
            response = self._get_llm().llm_completion(
                model=self.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            steps = json.loads(response)
            return [str(s) for s in steps[:7]] if isinstance(steps, list) else []
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return []
    
    # =========================================================================
    # TIER 3: Error-Targeted Retrieval with Alternative Implementations
    # =========================================================================
    
    def _tier3_error_retrieval(
        self, 
        goal: str, 
        error: str,
        current_workflow: Optional["WorkflowState"]
    ) -> RetrievalResult:
        """
        Enhanced error-targeted retrieval.
        
        Algorithm:
        1. LLM infers which step(s) are likely failing based on error
        2. Graph query finds ALTERNATIVE implementations for those steps
        3. Semantic search finds error-related heuristics
        4. Combine into enriched context for retry
        
        This gives the agent concrete alternatives to try, not just hints.
        """
        logger.info(f"TIER 3: Error retrieval triggered")
        
        heuristics = []
        code_patterns = []
        alternative_implementations = []
        
        # Step 1: LLM infers failing step(s) if we have workflow context
        if current_workflow and current_workflow.steps:
            inferred = self._infer_failing_steps(goal, error, current_workflow)
            
            if inferred:
                failing_principles = inferred.get("failing_steps", [])
                search_terms = inferred.get("suggested_search_terms", [])
                
                logger.info(f"TIER 3: LLM inferred {len(failing_principles)} failing step(s)")
                
                # Step 2: Query graph for alternative implementations
                for step_info in failing_principles:
                    principle_id = step_info.get("principle_id")
                    step_num = step_info.get("step_number")
                    
                    if principle_id:
                        # Get current implementation to exclude
                        current_impl_id = None
                        for step in current_workflow.steps:
                            if step.principle_id == principle_id and step.implementation:
                                current_impl_id = step.implementation.get("id")
                                break
                        
                        # Find alternative implementations
                        alts = self._find_alternative_implementations(
                            principle_id, 
                            exclude_impl_id=current_impl_id
                        )
                        
                        for alt in alts:
                            alt["for_step"] = step_num
                            alt["reasoning"] = step_info.get("reasoning", "")
                            alternative_implementations.append(alt)
                        
                        logger.info(f"TIER 3: Found {len(alts)} alternatives for {principle_id}")
                
                # Use LLM-suggested search terms for semantic search
                if search_terms:
                    error_query = " ".join(search_terms[:3])
                else:
                    error_query = self._extract_error_keywords(error)
            else:
                error_query = self._extract_error_keywords(error)
        else:
            error_query = self._extract_error_keywords(error)
        
        # Step 3: Semantic search for error-related heuristics
        query = f"{goal} {error_query}"
        
        kg_result = self.kg.search(
            query, 
            filters=KGSearchFilters(
                top_k=10, 
                page_types=["Heuristic", "Implementation"],
            ),
            context=error,
        )
        
        for item in kg_result.results:
            if item.page_type == "Heuristic":
                if item.overview:
                    heuristics.append(item.overview)
                if item.content:
                    code_blocks = self._extract_code_blocks(item.content)
                    code_patterns.extend(code_blocks[:2])
            elif item.page_type == "Implementation" and item.content:
                code_blocks = self._extract_code_blocks(item.content)
                code_patterns.extend(code_blocks[:1])
        
        logger.info(f"TIER 3: Found {len(alternative_implementations)} alternative implementations, "
                   f"{len(heuristics)} heuristics, {len(code_patterns)} code patterns")
        
        # Build enriched result
        result = RetrievalResult(
            mode=RetrievalMode.ERROR_TARGETED,
            workflow=current_workflow,
            heuristics=heuristics[:5],
            code_patterns=code_patterns[:5],
            source_pages=[item.id for item in kg_result.results[:5]],
            query_used=query,
            confidence=kg_result.results[0].score if kg_result.results else 0.0,
        )
        
        # Attach alternative implementations to workflow steps
        if alternative_implementations and current_workflow:
            result.workflow = self._enrich_workflow_with_alternatives(
                current_workflow, 
                alternative_implementations
            )
        
        return result
    
    def _infer_failing_steps(
        self, 
        goal: str, 
        error: str, 
        workflow: "WorkflowState"
    ) -> Optional[Dict]:
        """
        Use LLM to infer which step(s) are likely failing based on error.
        
        Returns dict with:
        - failing_steps: List of {step_number, principle_id, confidence, reasoning}
        - error_category: memory|type|value|import|runtime|unknown
        - suggested_search_terms: Terms to use for semantic search
        """
        prompt_file = PROMPTS_DIR / "infer_failing_step.md"
        if not prompt_file.exists():
            logger.warning("infer_failing_step.md prompt not found")
            return None
        
        # Build workflow steps description (same context format as decisions)
        steps_desc = []
        for step in workflow.steps:
            impl_name = ""
            if step.implementation:
                impl_name = f" (using: {step.implementation.get('title', 'unknown')})"
            steps_desc.append(
                f"Step {step.number}: {step.title}{impl_name}\n"
                f"  Principle: {step.principle_id or 'N/A'}\n"
                f"  Description: {step.description[:100] if step.description else 'N/A'}"
            )
        
        # Build context (similar to what we send to agent/decisions)
        context = f"Goal: {goal}\nWorkflow: {workflow.title}"
        
        prompt = prompt_file.read_text()
        prompt = prompt.replace("{context}", context)
        prompt = prompt.replace("{error}", error[:1000])  # Limit error length
        prompt = prompt.replace("{workflow_steps}", "\n\n".join(steps_desc))
        
        try:
            response = self._get_llm().llm_completion(
                model=self.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            result = json.loads(response)
            return result
        except Exception as e:
            logger.warning(f"Failed to infer failing step: {e}")
            return None
    
    def _find_alternative_implementations(
        self, 
        principle_id: str, 
        exclude_impl_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Query graph for alternative implementations of a Principle.
        
        Returns list of implementation dicts with title, overview, code_snippets.
        """
        if not self.kg or not hasattr(self.kg, '_neo4j_driver') or not self.kg._neo4j_driver:
            return []
        
        alternatives = []
        
        try:
            with self.kg._neo4j_driver.session() as session:
                # Query: Find all implementations of this principle
                query = """
                    MATCH (p:WikiPage {id: $principle_id})-[:IMPLEMENTED_BY]->(impl:WikiPage)
                    WHERE impl.id <> $exclude_id OR $exclude_id IS NULL
                    RETURN impl.id AS impl_id, impl.page_title AS impl_title
                """
                result = session.run(
                    query, 
                    principle_id=principle_id,
                    exclude_id=exclude_impl_id or ""
                )
                
                for record in result:
                    impl_id = record["impl_id"]
                    impl_title = record["impl_title"]
                    
                    # Fetch full implementation content
                    impl_content = self._get_implementation_content(impl_id)
                    
                    if impl_content:
                        alternatives.append({
                            "id": impl_id,
                            "title": impl_title,
                            "overview": impl_content.get("overview", ""),
                            "code_snippets": impl_content.get("code_snippets", []),
                            "principle_id": principle_id,
                        })
                    else:
                        alternatives.append({
                            "id": impl_id,
                            "title": impl_title,
                            "overview": "",
                            "code_snippets": [],
                            "principle_id": principle_id,
                        })
                        
        except Exception as e:
            logger.warning(f"Failed to find alternative implementations: {e}")
        
        return alternatives
    
    def _enrich_workflow_with_alternatives(
        self, 
        workflow: "WorkflowState", 
        alternatives: List[Dict]
    ) -> "WorkflowState":
        """
        Add alternative implementations to workflow steps.
        
        Modifies the workflow to include alternatives for steps that might be failing.
        """
        from src.memory.context import WorkflowState, StepState
        
        # Group alternatives by step number
        alts_by_step = {}
        for alt in alternatives:
            step_num = alt.get("for_step")
            if step_num:
                if step_num not in alts_by_step:
                    alts_by_step[step_num] = []
                alts_by_step[step_num].append(alt)
        
        # Create new steps with alternatives
        new_steps = []
        for step in workflow.steps:
            # Copy step and add alternatives if available
            new_step = StepState(
                number=step.number,
                title=step.title,
                status=step.status,
                description=step.description,
                heuristics=step.heuristics.copy() if step.heuristics else [],
                principle_id=step.principle_id,
                implementation=step.implementation,
            )
            
            # Add alternatives as additional heuristics/info
            if step.number in alts_by_step:
                for alt in alts_by_step[step.number]:
                    alt_hint = (
                        f"ALTERNATIVE APPROACH: {alt['title']}\n"
                        f"Reasoning: {alt.get('reasoning', 'Previous implementation failed')}\n"
                        f"{alt.get('overview', '')}"
                    )
                    new_step.heuristics.append(alt_hint)
                    
                    # Add code snippets as patterns
                    for snippet in alt.get("code_snippets", [])[:2]:
                        new_step.heuristics.append(f"Alternative code:\n```\n{snippet}\n```")
            
            new_steps.append(new_step)
        
        # Create new workflow with enriched steps
        return WorkflowState(
            id=workflow.id,
            title=workflow.title,
            source=workflow.source,
            confidence=workflow.confidence,
            steps=new_steps,
            current_step_index=workflow.current_step_index,
        )
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        Extract code blocks from content (supports WikiText and Markdown).
        
        WikiText format: <syntaxhighlight lang="python">...</syntaxhighlight>
        Markdown format: ```python ... ```
        """
        blocks = []
        
        # WikiText format
        wiki_blocks = re.findall(
            r'<syntaxhighlight[^>]*>(.*?)</syntaxhighlight>', 
            content, 
            re.DOTALL
        )
        blocks.extend(wiki_blocks)
        
        # Markdown format
        md_blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
        blocks.extend(md_blocks)
        
        return blocks
    
    def _extract_error_keywords(self, error: str) -> str:
        """Extract key terms from error message."""
        patterns = [
            r"(CUDA out of memory)",
            r"(RuntimeError: [\w\s]+)",
            r"(TypeError: [\w\s]+)",
            r"(ValueError: [\w\s]+)",
            r"(AttributeError: [\w\s]+)",
            r"(ImportError: [\w\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                return match.group(1)[:50]
        
        return error[:50]
    
    # =========================================================================
    # Fallback
    # =========================================================================
    
    def _create_minimal_plan(self, goal: str) -> RetrievalResult:
        """Create minimal plan when KG unavailable or nothing found."""
        from src.memory.context import WorkflowState, StepState
        
        steps = [
            StepState(
                number=1, 
                title="Analyze the problem", 
                status="pending",
                heuristics=["Break down the problem into sub-tasks"],
            ),
            StepState(
                number=2, 
                title="Implement solution", 
                status="pending",
                heuristics=["Start with the simplest working approach"],
            ),
            StepState(
                number=3, 
                title="Test and validate", 
                status="pending",
                heuristics=["Verify output matches expected format"],
            ),
        ]
        
        workflow = WorkflowState(
            id=f"Minimal/{goal[:30].replace(' ', '_')}",
            title=f"Plan: {goal[:50]}",
            source="fallback",
            confidence=0.3,
            steps=steps,
        )
        
        return RetrievalResult(
            mode=RetrievalMode.SYNTHESIZED_PLAN,
            workflow=workflow,
            plan_steps=[s.title for s in steps],
            confidence=0.3,
        )

    # =========================================================================
    # TWO-STAGE RETRIEVAL
    # =========================================================================
    # Stage 1 (Planning): Workflow, Principles, Heuristics → for high-level guidance
    # Stage 2 (Implementation): Implementation, Environment, Code → for concrete code
    # =========================================================================
    
    def get_planning_context(self, goal: str, exclude_workflow: Optional[str] = None) -> Dict[str, any]:
        """
        Stage 1: Get planning-level context.
        
        Retrieves:
        - Workflow (if exact match found)
        - Principles (theoretical concepts)
        - Heuristics (tips and best practices)
        
        Returns dict suitable for `additional_info` in ContextData.
        """
        if not self.kg or not self.kg.is_enabled():
            logger.info("KG not available for planning context")
            return {"workflow": None, "principles": [], "heuristics": []}
        
        # Try to get workflow first (existing TIER 1)
        workflow_result = self._tier1_workflow_match(goal, exclude_workflow)
        
        # Get principles
        principles = self._fetch_principles(goal)
        
        # Heuristics come from graph structure (workflow steps → USES_HEURISTIC)
        # If no workflow, heuristics are empty (TIER 2 synthesizes from principles)
        heuristics = []
        if workflow_result.workflow:
            for step in workflow_result.workflow.steps:
                heuristics.extend(step.heuristics)
        
        return {
            "workflow": workflow_result.workflow,
            "retrieval_mode": workflow_result.mode.value,
            "principles": principles,
            "heuristics": heuristics,
            "confidence": workflow_result.confidence,
        }
    
    def get_implementation_context(
        self, 
        goal: str, 
        workflow: Optional["WorkflowState"] = None,
        step_title: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get implementation-level context.
        
        UNIFIED APPROACH:
        1. If workflow is provided, extract implementations from steps (already graph-traversed)
        2. Otherwise, fall back to semantic search
        
        This ensures we get the CURATED implementations linked in the graph,
        not random semantic matches.
        
        Args:
            goal: The overall goal
            workflow: Optional - workflow with pre-loaded implementations from graph
            step_title: Optional - specific step being implemented (narrows search)
            
        Returns dict suitable for `kg_code_results` in ContextData.
        """
        implementations = []
        code_snippets = []
        environment = []
        
        # PREFERRED: Use implementations from workflow (already graph-traversed)
        if workflow and workflow.steps:
            for step in workflow.steps:
                if step.implementation:
                    impl = step.implementation
                    implementations.append({
                        "title": impl.get("title", f"Step {step.number}"),
                        "overview": impl.get("overview", ""),
                        "content": impl.get("content", "")[:500],
                    })
                    # Add code snippets from this implementation
                    if impl.get("code_snippets"):
                        code_snippets.extend(impl["code_snippets"])
            
            # Get environments from graph traversal (if available)
            if hasattr(self, '_last_graph_data') and self._last_graph_data:
                environment = self._last_graph_data.get("environments", [])
            
            logger.info(f"  📦 Implementation context (graph): {len(implementations)} pages, {len(code_snippets)} snippets")
            
            # If we got implementations from graph, return them
            if implementations:
                return {
                    "implementations": implementations,
                    "code_snippets": code_snippets[:10],
                    "environment": environment[:3],
                }
        
        # FALLBACK: Semantic search (when no workflow or workflow has no implementations)
        if not self.kg or not self.kg.is_enabled():
            logger.info("KG not available for implementation context")
            return {"implementations": [], "code_snippets": [], "environment": []}
        
        logger.info("  📦 Using semantic search for implementations (no graph data)")
        
        # Build focused query
        query = step_title if step_title else goal
        
        # Search for Implementation pages
        impl_result = self.kg.search(
            query,
            filters=KGSearchFilters(
                top_k=10,
                page_types=["Implementation"],
                min_score=0.4,
            ),
        )
        
        for item in impl_result.results[:5]:
            if item.page_type == "Implementation":
                implementations.append({
                    "title": item.page_title,
                    "overview": item.overview or "",
                    "content": (item.content[:500] if item.content else ""),
                })
                
                if item.content:
                    code_blocks = self._extract_code_blocks(item.content)
                    code_snippets.extend(code_blocks[:2])
        
        # Search for Environment pages
        env_result = self.kg.search(
            query,
            filters=KGSearchFilters(
                top_k=5,
                page_types=["Environment"],
                min_score=0.3,
            ),
        )
        
        for item in env_result.results[:3]:
            if item.page_type == "Environment":
                environment.append({
                    "title": item.page_title,
                    "requirements": item.overview or "",
                })
        
        logger.info(f"  📦 Implementation context (semantic): {len(implementations)} pages, {len(code_snippets)} snippets")
        
        return {
            "implementations": implementations,
            "code_snippets": code_snippets[:10],
            "environment": environment,
        }
    
    def _fetch_principles(self, goal: str) -> List[Dict[str, str]]:
        """Fetch principle pages relevant to the goal."""
        if not self.kg:
            return []
            
        result = self.kg.search(
            goal,
            filters=KGSearchFilters(
                top_k=5,
                page_types=["Principle"],
                min_score=0.4,
            ),
        )
        
        principles = []
        for item in result.results[:3]:
            if item.page_type == "Principle":
                principles.append({
                    "title": item.page_title,
                    "overview": item.overview or "",
                })
        
        return principles