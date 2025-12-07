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

from src.core.llm import LLMBackend
from src.knowledge.search.base import KnowledgeSearch, KnowledgeResult
from src.knowledge.search.factory import register_knowledge_search


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
        
        # Neo4j connection params
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # Lazy-load driver and LLM only when enabled
        self._driver = None
        self._llm = None
        
        if self.enabled:
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
    
    def search(self, query: str, context: Optional[str] = None) -> KnowledgeResult:
        """
        Search the knowledge graph using LLM-guided navigation.
        
        Args:
            query: The search query (typically problem description)
            context: Optional additional context (e.g., last experiment info)
            
        Returns:
            KnowledgeResult with text and code results
        """
        if not self.enabled or not self._driver:
            return KnowledgeResult()
        
        # Build query prompt
        query_prompt = f"Problem: \n{query}"
        if context:
            query_prompt += f"\n\nLast Experiment: \n{context}"
        
        try:
            text_results, code_results = self._retrieve_navigate(
                query_prompt,
                search_top_k=self.search_top_k,
                navigation_steps=self.navigation_steps,
                expansion_limit=self.expansion_limit,
                search_node_type=self.search_node_type,
            )
            
            return KnowledgeResult(
                text_results=text_results or "",
                code_results=code_results or "",
                metadata={
                    "search_type": "kg_llm_navigation",
                    "search_top_k": self.search_top_k,
                    "navigation_steps": self.navigation_steps,
                }
            )
        except Exception:
            return KnowledgeResult()
    
    def _retrieve_navigate(
        self, 
        query: str,
        search_top_k: int = 1,
        navigation_steps: int = 2,
        expansion_limit: int = 3,
        search_node_type: Optional[str] = None,         
    ) -> tuple:
        """
        Retrieve knowledge using LLM-guided graph navigation.
        
        1. Search for initial nodes matching the query
        2. Use LLM to select which neighbors to explore
        3. Repeat for N navigation steps
        4. Return aggregated knowledge from visited nodes
        """
        # Find starting nodes
        selected_nodes = self._keyword_search(query, top_k=search_top_k, node_type=search_node_type)
        navigation_parents = selected_nodes
        navigated_node_ids = [node['id'] for node in selected_nodes]

        try: 
            for step in range(navigation_steps):
                navigation_childs = []
                neighbor_info = []

                # Collect neighbors of current nodes
                for node in navigation_parents:
                    neighbors = self._get_neighbor_nodes(node['id'])
                    for neighbor in neighbors:
                        if neighbor['id'] not in navigated_node_ids:
                            navigated_node_ids.append(neighbor['id'])
                            navigation_childs.append(neighbor)
                            neighbor_info.append(f"{neighbor.get('name')}")
                
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
            
            # Format results
            kg_code_results = "\n\n".join([
                f"{node.get('name')} : {node.get('content')}" 
                for node in selected_nodes if node.get('type') == 'code'
            ])
            kg_all_results = "\n\n".join([
                f"{node.get('name')} : {node.get('content')}" 
                for node in selected_nodes
            ])
            return kg_all_results, kg_code_results
            
        except Exception as e:
            print(f"Error in retrieve_navigate: {e}")
            return "No graph results.", ""
    
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
            model="gpt-5-mini",
            messages=messages,
        )
        return json.loads(response)
    
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
        print("Text Results:")
        print(result.text_results)
        print("\nCode Results:")
        print(result.code_results)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        search.close()


if __name__ == "__main__":
    main()

