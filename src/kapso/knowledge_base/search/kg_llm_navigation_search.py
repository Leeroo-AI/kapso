# KG LLM Navigation Search
#
# Knowledge Graph search with LLM-guided navigation.
# Uses Neo4j for storage and LLM to intelligently navigate
# the graph structure, selecting relevant neighbor nodes
# at each step based on the query context.
#
# Registered as "kg_llm_navigation" via the factory decorator.

import json
import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from kapso.core.llm import LLMBackend
from kapso.knowledge_base.search.base import KGEditInput, KnowledgeSearch, KGOutput, KGResultItem, KGSearchFilters, WikiPage
from kapso.knowledge_base.search.factory import register_knowledge_search


@register_knowledge_search("kg_llm_navigation")
class KGLLMNavigationSearch(KnowledgeSearch):
    """
    Knowledge Graph search with LLM-guided navigation.
    
    Uses Neo4j for indexing and LLM to intelligently navigate
    the graph structure. At each step, the LLM selects which
    neighbor nodes to explore based on the query context.
    
    Config params (from knowledge_search.yaml):
        - search_top_k: Number of top nodes to start search from
        - navigation_steps: How many hops to navigate from initial nodes
        - expansion_limit: Max nodes to expand at each navigation step
        - search_node_type: Type of nodes to search (e.g., "specialization")
    """
    
    def __init__(
        self,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize KG LLM Navigation search."""
        super().__init__(enabled=enabled, params=params)
        # Extract params with defaults
        self.search_top_k = self.params.get("search_top_k", 1)
        self.navigation_steps = self.params.get("navigation_steps", 3)
        self.expansion_limit = self.params.get("expansion_limit", 3)
        self.search_node_type = self.params.get("search_node_type", "specialization")
        self.navigations_model = self.params.get("navigations_model", "gpt-5-mini")
        
        # Neo4j connection params
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # Lazy-load driver and LLM only when enabled
        self._driver = None
        self._llm = None
        
        if self._enabled:
            self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize Neo4j driver and LLM backend."""
        try:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            self._llm = LLMBackend()
            self._setup_constraints()
        except Exception:
            self._driver = None
            self._llm = None
    
    def _setup_constraints(self) -> None:
        """Set up Neo4j constraints for node uniqueness."""
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run("""
                CREATE CONSTRAINT node_id_unique IF NOT EXISTS
                FOR (n:Node)
                REQUIRE n.id IS UNIQUE
            """)
    
    # =========================================================================
    # Indexing Methods
    # =========================================================================
    
    def index(self, data: Dict[str, Any]) -> None:
        """
        Index knowledge graph data into Neo4j.
        
        Args:
            data: Dictionary with 'nodes' and 'edges' keys
                  - nodes: {id: {name, type, content, ...}}
                  - edges: [{source, target, relationship, ...}]
        """
        if not self._driver:
            return
        
        with self._driver.session() as session:
            # Index nodes
            for node_id, node_data in data.get('nodes', {}).items():
                properties = node_data.copy()
                properties['id'] = node_id
                
                session.run("""
                    MERGE (n:Node {id: $id})
                    SET n += $properties
                """, id=node_id, properties=properties)
            
            # Index edges
            for edge in data.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                relationship = edge.get('relationship', 'RELATES_TO')
                properties = {
                    k: v for k, v in edge.items() 
                    if k not in ['source', 'target', 'relationship']
                }
                
                session.run("""
                    MATCH (source:Node {id: $source})
                    MATCH (target:Node {id: $target})
                    MERGE (source)-[r:""" + relationship + """]->(target)
                    SET r += $properties
                """, source=source, target=target, properties=properties)
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    # =========================================================================
    # Search Methods
    # =========================================================================
    
    def search(
        self, 
        query: str, 
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> KGOutput:
        """
        Search the knowledge graph using LLM-guided navigation.
        
        Args:
            query: The search query (typically problem description)
            filters: Optional filters (top_k, min_score, page_types, domains)
            context: Optional additional context (e.g., last experiment info)
            
        Returns:
            KGOutput with ranked and filtered results
        """
        # Use default filters if not provided
        filters = filters or KGSearchFilters()
        
        if not self._enabled or not self._driver:
            return KGOutput(query=query, filters=filters)
        
        # Build query prompt
        query_prompt = f"Problem: \n{query}"
        if context:
            query_prompt += f"\n\nLast Experiment: \n{context}"
        
        try:
            result_items = self._retrieve_navigate(
                query_prompt,
                filters=filters,
                search_top_k=self.search_top_k,
                navigation_steps=self.navigation_steps,
                expansion_limit=self.expansion_limit,
                search_node_type=self.search_node_type,
            )
            
            return KGOutput(
                query=query,
                filters=filters,
                results=result_items,
                total_found=len(result_items),
                search_metadata={
                    "search_type": "kg_llm_navigation",
                    "search_top_k": self.search_top_k,
                    "navigation_steps": self.navigation_steps,
                    "expansion_limit": self.expansion_limit,
                }
            )
        except Exception:
            return KGOutput(query=query, filters=filters)
    
    def _retrieve_navigate(
        self, 
        query: str,
        filters: KGSearchFilters,
        search_top_k: int = 1,
        navigation_steps: int = 2,
        expansion_limit: int = 3,
        search_node_type: Optional[str] = None,         
    ) -> List[KGResultItem]:
        """
        Retrieve knowledge using LLM-guided graph navigation.
        
        1. Search for initial nodes matching the query
        2. Use LLM to select which neighbors to explore
        3. Repeat for N navigation steps
        4. Apply filters and return list of KGResultItem
        
        Args:
            query: Search query
            filters: KGSearchFilters with top_k, min_score, page_types, domains
            search_top_k: Initial nodes to find (navigation param)
            navigation_steps: Graph navigation depth
            expansion_limit: Max nodes per navigation step
            search_node_type: Starting node type filter
        """
        # Find starting nodes
        selected_nodes = self._keyword_search(query, top_k=search_top_k, node_type=search_node_type)
        navigation_parents = selected_nodes
        navigated_node_ids = [node['id'] for node in selected_nodes]
        
        # Track scores: initial nodes get higher scores, later nodes get lower
        node_scores = {node['id']: 1.0 for node in selected_nodes}

        try: 
            for step in range(navigation_steps):
                navigation_childs = []
                neighbor_info = []
                
                # Score decay for each navigation step (clamped to min 0.1)
                step_score = max(0.1, 1.0 - (step + 1) * 0.15)

                # Collect neighbors of current nodes
                for node in navigation_parents:
                    neighbors = self._get_neighbor_nodes(node['id'])
                    for neighbor in neighbors:
                        if neighbor['id'] not in navigated_node_ids:
                            navigated_node_ids.append(neighbor['id'])
                            navigation_childs.append(neighbor)
                            neighbor_info.append(f"{neighbor.get('name')}")
                            node_scores[neighbor['id']] = step_score
                
                if len(navigation_childs) == 0:
                    break

                # Use LLM to select relevant neighbors
                prompt = f"""
                    You are a world-class researcher navigating a knowledge graph. You have already explored these nodes:
                    {chr(10).join(f"- {node.get('name')} : {node.get('content')}" for node in selected_nodes)}

                    Based on the type and especially the data type of the problem you want to solve <problem>\n{query}\n</problem> and previously navigated nodes, which of these new nodes would be most relevant to explore? Note that you can choose at most {expansion_limit} nodes to explore. note that you should only choose nodes that are relevant and not just slightly similar.
                    Respond with a JSON list of node names you want to add to your exploration, or an empty list if none are relevant. Note that you should not 

                    You have the following new neighbor nodes available for your current exploration:
                    {chr(10).join(neighbor_info)}

                    Example response: ["node1", "node3"]
                """
                selected_neighbor_names = list(self._llm_call(prompt))
                
                # Filter selected neighbors
                new_nodes = []
                for neighbor in navigation_childs:
                    if neighbor.get('name') in selected_neighbor_names:
                        new_nodes.append(neighbor)
                
                selected_nodes.extend(new_nodes)
                navigation_parents = new_nodes
            
            # Convert to KGResultItem list
            result_items = []
            for node in selected_nodes:
                node_id = node.get('id', '')
                node_type = node.get('type', 'unknown')
                node_domains = node.get('domains', [])
                score = node_scores.get(node_id, 0.5)
                
                # Apply page_types filter
                if filters.page_types and node_type not in filters.page_types:
                    continue
                
                # Apply domains filter (match any)
                if filters.domains:
                    if not any(d in node_domains for d in filters.domains):
                        continue
                
                # Apply min_score filter
                if filters.min_score is not None and score < filters.min_score:
                    continue
                
                result_items.append(KGResultItem(
                    id=node_id,
                    score=score,
                    page_title=node.get('name', ''),
                    page_type=node_type,
                    overview=node.get('overview', ''),
                    content=node.get('content', '') if filters.include_content else '',
                    metadata={
                        k: v for k, v in node.items() 
                        if k not in ('id', 'name', 'type', 'overview', 'content')
                    }
                ))
            
            # Sort by score descending
            result_items.sort(key=lambda x: x.score, reverse=True)
            
            # Apply top_k limit
            return result_items[:filters.top_k]
            
        except Exception as e:
            print(f"Error in retrieve_navigate: {e}")
            return []
    
    def _keyword_search(
        self, 
        query: str, 
        top_k: int = 10, 
        node_type: Optional[str] = None
    ) -> List[Any]:
        """Search nodes by keyword matching."""
        results = self._search_by_keyword(query, top_k, node_type)
        
        # Deduplicate results
        seen_ids = set()
        unique_results = []
        for result in results:
            if result['id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['id'])
                
        return unique_results[:top_k]

    def _search_by_keyword(
        self, 
        query: str, 
        top_k: int, 
        node_type: Optional[str] = None
    ) -> List[Any]:
        """Execute keyword search query on Neo4j."""
        if not self._driver:
            return []
            
        with self._driver.session() as session:
            words = query.lower().split()
            if not words:
                return []            
            
            if node_type:
                result = session.run("""
                    MATCH (n:Node)
                    WHERE n.type = $type AND ANY(word IN $words WHERE 
                        n.content CONTAINS word OR 
                        n.name CONTAINS word
                    )
                    RETURN n
                    LIMIT $top_k
                """, words=words, top_k=top_k, type=node_type)
            else:
                result = session.run("""
                    MATCH (n:Node)
                    WHERE ANY(word IN $words WHERE 
                        n.content CONTAINS word OR 
                        n.name CONTAINS word
                    )
                    RETURN n
                    LIMIT $top_k
                """, words=words, top_k=top_k)
            return [record['n'] for record in result]

    def _get_neighbor_nodes(self, node_id: str) -> List[Any]:
        """Get all neighbor nodes of a given node."""
        if not self._driver:
            return []
            
        with self._driver.session() as session:
            result = session.run("""
                MATCH (n:Node {id: $node_id})-[r]-(neighbor:Node)
                RETURN neighbor
            """, node_id=node_id)
            
            return [dict(record['neighbor']) for record in result]

    def _llm_call(self, query: str) -> Any:
        """Call LLM and parse JSON response."""
        if not self._llm:
            return []
        messages = [{"role": "user", "content": query}]
        response = self._llm.llm_completion(
            model=self.navigations_model,
            messages=messages,
        )
        return json.loads(response)
    
    def get_page(self, page_title: str) -> Optional[WikiPage]:
        """
        Retrieve a wiki page by its title.
        
        Looks up the page in Neo4j by exact name match.
        
        Args:
            page_title: Exact title of the page to retrieve
            
        Returns:
            WikiPage if found, None otherwise
        """
        if not self._driver:
            return None
        
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (n:Node {name: $name}) RETURN n",
                    name=page_title
                )
                record = result.single()
                
                if record:
                    node = dict(record["n"])
                    return WikiPage(
                        id=node.get("id", ""),
                        page_title=node.get("name", ""),
                        page_type=node.get("type", ""),
                        overview=node.get("overview", ""),
                        content=node.get("content", ""),
                        domains=node.get("domains", []) if isinstance(node.get("domains"), list) else [],
                    )
                
                return None
                
        except Exception:
            return None
    
    def edit(self, data: KGEditInput) -> bool:
        """
        Edit an existing wiki page.
        
        Note: KGLLMNavigationSearch uses a different data model (nodes/edges dict)
        than the wiki-based KGGraphSearch. This implementation only updates
        the Neo4j node properties.
        
        Args:
            data: KGEditInput with page_id and fields to update
            
        Returns:
            True if successful, False if page not found
        """
        if not self._driver:
            return False
        
        updates = data.get_updates()
        if not updates:
            return True
        
        try:
            with self._driver.session() as session:
                # Map field names to Neo4j node properties
                neo4j_updates = {}
                field_mapping = {
                    "page_title": "name",
                    "page_type": "type",
                    "overview": "overview",
                    "content": "content",
                    "domains": "domains",
                }
                
                for our_field, neo4j_field in field_mapping.items():
                    if our_field in updates:
                        neo4j_updates[neo4j_field] = updates[our_field]
                
                if neo4j_updates:
                    # Update by page_id (which maps to node id or name)
                    result = session.run(
                        "MATCH (n:Node) WHERE n.id = $id OR n.name = $id "
                        "SET n += $updates RETURN n",
                        id=data.page_id,
                        updates=neo4j_updates,
                    )
                    return result.single() is not None
                
                return True
                
        except Exception as e:
            print(f"Error in edit: {e}")
            return False
    
    # =========================================================================
    # Index Metadata Methods (for .index files)
    # =========================================================================
    
    def get_backend_refs(self) -> Dict[str, Any]:
        """
        Return backend-specific references for index file.
        
        Returns Neo4j URI and node label used by this backend.
        """
        return {
            "neo4j_uri": self.neo4j_uri,
            "node_label": "Node",  # This backend uses :Node label
        }
    
    def validate_backend_data(self) -> bool:
        """
        Check if Neo4j has indexed Node data.
        
        Returns:
            True if Neo4j has Node entries, False otherwise
        """
        if not self._driver:
            return False
        try:
            with self._driver.session() as session:
                result = session.run("MATCH (n:Node) RETURN count(n) as count")
                return result.single()["count"] > 0
        except Exception:
            return False
    
    def get_indexed_count(self) -> int:
        """
        Get the number of indexed nodes.
        
        Returns:
            Count of Node entries in Neo4j
        """
        if not self._driver:
            return 0
        try:
            with self._driver.session() as session:
                result = session.run("MATCH (n:Node) RETURN count(n) as count")
                return result.single()["count"]
        except Exception:
            return 0
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()


# =============================================================================
# Lazy Singleton for backward compatibility
# =============================================================================

_kg_search_instance = None

def get_kg_llm_navigation_search() -> KGLLMNavigationSearch:
    """Get the KGLLMNavigationSearch instance (lazy initialization)."""
    global _kg_search_instance
    if _kg_search_instance is None:
        _kg_search_instance = KGLLMNavigationSearch()
    return _kg_search_instance


class _KGSearchProxy:
    """Proxy class that lazy loads the KGLLMNavigationSearch."""
    def __getattr__(self, name):
        return getattr(get_kg_llm_navigation_search(), name)


# For backward compatibility with code that imports kg_agent
kg_search = _KGSearchProxy()


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Test the KG LLM Navigation Search."""
    try:
        search = KGLLMNavigationSearch()
        search.clear()
        
        # Load test data
        kg_data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'benchmarks', 'mle', 'data', 'kg_data.json'
        )
        with open(kg_data_path, 'r') as f:
            graph_data = json.load(f)
        
        # Index the data
        search.index(graph_data)
        
        # Test query
        query = """
            The task type is **Classification**.
            The data type is **Tabular with Image data**.
            
            Task: Classify samples into multiple categories using numeric features or image data.
            Dataset: Contains training CSV with features and labels, test CSV for predictions, and image files.
        """
        result = search.search(query)
        print("Search Results:")
        print(result.to_context_string())
        print(f"\nTotal found: {result.total_found}")
        print(f"\nTop result: {result.top_result}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        search.close()


if __name__ == "__main__":
    main()

