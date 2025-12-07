#!/usr/bin/env python3
"""
Wiki Search - Provides keyword, semantic, and hybrid search over wiki graph.

Supports two modes:
1. Neo4j-only: Traditional search with embeddings in Neo4j
2. Weaviate + Neo4j: Vector search in Weaviate, graph enrichment from Neo4j

Usage:
    from wiki_search import WikiSearch
    
    # Neo4j-only mode
    searcher = WikiSearch(neo4j_uri, neo4j_user, neo4j_password)
    
    # Weaviate + Neo4j mode (recommended for scale)
    searcher = WikiSearch(
        neo4j_uri, neo4j_user, neo4j_password,
        use_weaviate=True,
        weaviate_url="http://localhost:8080"
    )
    results = searcher.hybrid_search("machine learning", limit=10)
"""

import os
import logging
import math
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiSearch:
    """Search interface for wiki graph with optional Weaviate vector search"""
    
    def __init__(self, uri: str, user: str, password: str, 
                 domain: str,
                 database: str = "neo4j",
                 openai_api_key: Optional[str] = None,
                 use_weaviate: bool = False,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_collection: str = "WikiPages"):
        """
        Initialize search interface.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            domain: Wiki domain to search within (e.g., 'acme.leeroo.com')
            database: Neo4j database name (default: "neo4j")
            openai_api_key: Optional OpenAI API key for semantic search
            use_weaviate: Whether to use Weaviate for vector search
            weaviate_url: Weaviate server URL
            weaviate_collection: Name of Weaviate collection
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.domain = domain  # Used for filtering searches to specific wiki
        self.database = database
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Weaviate configuration
        self.use_weaviate = use_weaviate
        self.weaviate_collection_name = weaviate_collection
        self.weaviate_client = None
        
        # Check if semantic search is available
        self.semantic_enabled = False
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                self.semantic_enabled = True
                logger.info("Semantic search enabled")
            except ImportError:
                logger.warning("OpenAI library not installed - semantic search disabled")
        
        # Initialize Weaviate if requested
        if use_weaviate and self.semantic_enabled:
            try:
                import weaviate
                
                # Connect to Weaviate
                self.weaviate_client = weaviate.connect_to_local(
                    host=weaviate_url.replace("http://", "").replace(":8080", ""),
                    port=8080
                )
                logger.info(f"Connected to Weaviate at {weaviate_url}")
                
            except ImportError:
                logger.warning("Weaviate client not installed - falling back to Neo4j-only")
                self.use_weaviate = False
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {e}")
                self.use_weaviate = False
    
    def close(self):
        """Close connections"""
        self.driver.close()
        if self.weaviate_client:
            self.weaviate_client.close()
    
    def keyword_search(self, query: str, limit: int = 10, page_type: Optional[str] = None) -> List[Dict]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query
            limit: Maximum results
            page_type: Optional filter by page type (concept/implementation/workflow/resource)
            
        Returns:
            List of matching pages with scores
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Build WHERE clause for keyword matching
        conditions = []
        for word in words:
            conditions.append(f"""
                toLower(p.name) CONTAINS '{word}' OR 
                toLower(p.id) CONTAINS '{word}'
            """)
        
        where_clause = " OR ".join(conditions) if conditions else "TRUE"
        
        # Add page_type filter if provided
        type_filter = ""
        if page_type:
            type_filter = "AND p.type = $page_type"
        
        with self.driver.session(database=self.database) as session:
            cypher = f"""
                MATCH (p:WikiPage)
                WHERE p.domain = $domain {type_filter} AND ({where_clause})
                WITH p, 
                     CASE 
                       WHEN toLower(p.name) = $query_str THEN 2.0
                       WHEN toLower(p.name) CONTAINS $query_str THEN 1.5
                       ELSE 1.0
                     END AS score
                RETURN p.id, p.title, p.name, p.type, score
                ORDER BY score DESC
                LIMIT $limit
            """
            
            params = {'query_str': query_lower, 'domain': self.domain, 'limit': limit}
            if page_type:
                params['page_type'] = page_type
            
            result = session.run(cypher, **params)
            
            results = []
            for record in result:
                results.append({
                    'id': record['p.id'],  # Domain-prefixed ID
                    'title': record['p.title'],  # Original title
                    'name': record['p.name'],
                    'type': record['p.type'],
                    'score': float(record['score']),
                    'match_type': 'keyword'
                })
            
            return results
    
    def semantic_search(self, query: str, limit: int = 10, 
                       retrieval_query: Optional[str] = None,
                       page_type: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            limit: Maximum results
            retrieval_query: Optional Cypher query for graph enrichment (Weaviate mode)
            page_type: Optional filter by page type (concept/implementation/workflow/resource)
            
        Returns:
            List of similar pages with similarity scores
        """
        if not self.semantic_enabled:
            logger.warning("Semantic search not available - falling back to keyword search")
            return self.keyword_search(query, limit, page_type)
        
        # Use Weaviate if available
        if self.use_weaviate and self.weaviate_client:
            return self._weaviate_semantic_search(query, limit, retrieval_query, page_type)
        
        # Generate query embedding for Neo4j-only mode
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return self.keyword_search(query, limit, page_type)
        
        # Add page_type filter if provided
        type_filter = ""
        if page_type:
            type_filter = "AND p.type = $page_type"
        
        with self.driver.session(database=self.database) as session:
            # Check if vector search is available
            try:
                # Neo4j 5.11+ with vector search
                cypher = f"""
                    MATCH (p:WikiPage)
                    WHERE p.domain = $domain {type_filter} AND p.embedding IS NOT NULL
                    WITH p, gds.similarity.cosine(p.embedding, $embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN p.id, p.title, p.name, p.type, similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                """
                params = {'embedding': query_embedding, 'domain': self.domain, 'limit': limit}
                if page_type:
                    params['page_type'] = page_type
                result = session.run(cypher, **params)
                
            except Exception as e:
                # Fallback: Manual cosine similarity (slower)
                logger.debug(f"Vector search not available: {e}")
                cypher = f"""
                    MATCH (p:WikiPage)
                    WHERE p.domain = $domain {type_filter} AND p.embedding IS NOT NULL
                    RETURN p.id, p.title, p.name, p.type, p.embedding
                """
                params = {'domain': self.domain}
                if page_type:
                    params['page_type'] = page_type
                result = session.run(cypher, **params)
                
                # Calculate similarities manually
                pages_with_sim = []
                for record in result:
                    if record['p.embedding']:
                        similarity = self._cosine_similarity(
                            query_embedding, 
                            record['p.embedding']
                        )
                        if similarity > 0.5:
                            pages_with_sim.append({
                                'id': record['p.id'],
                                'name': record['p.name'],
                                'type': record['p.type'],
                                'similarity': similarity
                            })
                
                # Sort and limit
                pages_with_sim.sort(key=lambda x: x['similarity'], reverse=True)
                result = pages_with_sim[:limit]
                
                return [{
                    'id': r['id'],
                    'name': r['name'],
                    'type': r['type'],
                    'score': r['similarity'],
                    'match_type': 'semantic'
                } for r in result]
            
            # Process vector search results
            results = []
            for record in result:
                results.append({
                    'id': record['p.id'],
                    'name': record['p.name'],
                    'type': record['p.type'],
                    'score': float(record['similarity']),
                    'match_type': 'semantic'
                })
            
            return results
    
    def hybrid_search(self, query: str, limit: int = 10, 
                     semantic_weight: float = 0.6,
                     page_type: Optional[str] = None) -> List[Dict]:
        """
        Perform hybrid search combining keyword and semantic.
        
        Args:
            query: Search query
            limit: Maximum results
            semantic_weight: Weight for semantic scores (0-1)
            page_type: Optional filter by page type (concept/implementation/workflow/resource)
            
        Returns:
            Combined and ranked results
        """
        # Get keyword results
        keyword_results = self.keyword_search(query, limit * 2, page_type)
        
        # Get semantic results if available
        if self.semantic_enabled:
            semantic_results = self.semantic_search(query, limit * 2, page_type=page_type)
        else:
            semantic_results = []
        
        # Combine results
        combined = {}
        
        # Add keyword results
        keyword_weight = 1.0 - semantic_weight
        for result in keyword_results:
            page_id = result['id']
            combined[page_id] = {
                **result,
                'keyword_score': result['score'],
                'semantic_score': 0.0,
                'final_score': result['score'] * keyword_weight,
                'match_types': ['keyword']
            }
        
        # Add/update with semantic results
        for result in semantic_results:
            page_id = result['id']
            if page_id in combined:
                combined[page_id]['semantic_score'] = result['score']
                combined[page_id]['final_score'] += result['score'] * semantic_weight
                combined[page_id]['match_types'].append('semantic')
            else:
                combined[page_id] = {
                    **result,
                    'keyword_score': 0.0,
                    'semantic_score': result['score'],
                    'final_score': result['score'] * semantic_weight,
                    'match_types': ['semantic']
                }
        
        # Sort by final score
        ranked = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        
        return ranked[:limit]
    
    def find_similar(self, page_id: str, limit: int = 5) -> List[Dict]:
        """
        Find pages similar to a given page.
        
        Args:
            page_id: ID of the reference page
            limit: Maximum results
            
        Returns:
            List of similar pages
        """
        with self.driver.session(database=self.database) as session:
            # Try semantic similarity first
            if self.semantic_enabled:
                result = session.run("""
                    MATCH (p:WikiPage {id: $id})
                    WHERE p.embedding IS NOT NULL
                    RETURN p.embedding
                """, id=page_id).single()
                
                if result and result['p.embedding']:
                    # Find similar by embedding
                    try:
                        cypher = """
                            MATCH (p:WikiPage)
                            WHERE p.id <> $id AND p.embedding IS NOT NULL
                            WITH p, gds.similarity.cosine(p.embedding, $embedding) AS similarity
                            WHERE similarity > 0.6
                            RETURN p.id, p.name, p.type, similarity
                            ORDER BY similarity DESC
                            LIMIT $limit
                        """
                        result = session.run(cypher,
                                           id=page_id,
                                           embedding=result['p.embedding'],
                                           limit=limit)
                        
                        similar = []
                        for record in result:
                            similar.append({
                                'id': record['p.id'],
                                'name': record['p.name'],
                                'type': record['p.type'],
                                'similarity': float(record['similarity']),
                                'method': 'semantic'
                            })
                        
                        if similar:
                            return similar
                    except:
                        pass
            
            # Fallback: Find similar by shared relationships
            cypher = """
                MATCH (p1:WikiPage {id: $id})-[r1]->(shared)<-[r2]-(p2:WikiPage)
                WHERE p1.id <> p2.id
                WITH p2, COUNT(DISTINCT shared) as shared_count
                ORDER BY shared_count DESC
                LIMIT $limit
                MATCH (p2)
                RETURN p2.id, p2.name, p2.type, shared_count
            """
            
            result = session.run(cypher, id=page_id, limit=limit)
            
            similar = []
            for record in result:
                similar.append({
                    'id': record['p2.id'],
                    'name': record['p2.name'],
                    'type': record['p2.type'],
                    'shared_connections': record['shared_count'],
                    'method': 'graph'
                })
            
            return similar
    
    def graph_search(self, start_id: str, relationship: Optional[str] = None,
                     direction: str = 'outgoing', depth: int = 1) -> List[Dict]:
        """
        Search graph by traversing relationships.
        
        Args:
            start_id: Starting page ID
            relationship: Optional relationship type to follow
            direction: 'outgoing', 'incoming', or 'both'
            depth: Maximum traversal depth
            
        Returns:
            List of connected pages
        """
        # Build relationship pattern
        if relationship:
            rel_pattern = f"[:{relationship}*1..{depth}]"
        else:
            rel_pattern = f"[*1..{depth}]"
        
        # Build direction pattern
        if direction == 'outgoing':
            pattern = f"(start)-{rel_pattern}->(connected)"
        elif direction == 'incoming':
            pattern = f"(start)<-{rel_pattern}-(connected)"
        else:
            pattern = f"(start)-{rel_pattern}-(connected)"
        
        with self.driver.session(database=self.database) as session:
            cypher = f"""
                MATCH (start:WikiPage {{id: $start_id, domain: $domain}})
                MATCH {pattern}
                WHERE connected.domain = $domain
                RETURN DISTINCT connected.id as id, 
                       connected.title as title,
                       connected.name as name, 
                       connected.type as type,
                       length(shortest((start)-[*]-(connected))) as distance
                ORDER BY distance, name
            """
            
            # Create domain-prefixed ID for start node
            prefixed_start_id = f"{self.domain}:{start_id}"
            result = session.run(cypher, start_id=prefixed_start_id, domain=self.domain)
            
            connected = []
            for record in result:
                connected.append({
                    'id': record['id'],
                    'name': record['name'],
                    'type': record['type'],
                    'distance': record['distance']
                })
            
            return connected
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if not self.semantic_enabled:
            return None
        
        try:
            import openai
            
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text[:8000]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _weaviate_semantic_search(self, query: str, limit: int = 10, 
                                  retrieval_query: Optional[str] = None,
                                  page_type: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search using Weaviate with Neo4j graph enrichment.
        
        Args:
            query: Search query
            limit: Maximum results
            retrieval_query: Optional Cypher query to enrich results with graph context
            page_type: Optional filter by page type (concept/implementation/workflow/resource)
            
        Returns:
            List of results enriched with graph data
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Search in Weaviate
            collection = self.weaviate_client.collections.get(self.weaviate_collection_name)
            
            # Build where filter for page_type if provided
            where_filter = None
            if page_type:
                where_filter = {
                    "path": ["type"],
                    "operator": "Equal",
                    "valueText": page_type
                }
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                where=where_filter,
                return_properties=["text", "title", "type", "neo4j_id", "sections"]
            )
            
            results = []
            for item in response.objects:
                result = {
                    'id': item.properties.get('neo4j_id'),
                    'name': item.properties.get('title'),
                    'type': item.properties.get('type'),
                    'score': item.metadata.distance if hasattr(item.metadata, 'distance') else 0.0,
                    'text': item.properties.get('text', '')[:500],  # Preview
                    'match_type': 'semantic',
                    'source': 'weaviate'
                }
                
                # Enrich with graph data if retrieval query provided
                if retrieval_query and result['id']:
                    enrichment = self._enrich_with_graph(result['id'], retrieval_query)
                    result.update(enrichment)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            # Fallback to Neo4j search
            return self._neo4j_semantic_search(query, limit)
    
    def _enrich_with_graph(self, node_id: str, retrieval_query: str) -> Dict[str, Any]:
        """
        Enrich a result with graph data from Neo4j.
        
        Args:
            node_id: Node ID to enrich
            retrieval_query: Cypher query to get additional context
            
        Returns:
            Dictionary with additional graph data
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Default retrieval query if none provided
                if not retrieval_query:
                    retrieval_query = """
                        MATCH (node:WikiPage {id: $node_id})
                        OPTIONAL MATCH (node)-[r]->(related:WikiPage)
                        RETURN node, collect({
                            type: type(r),
                            target: related.name,
                            target_id: related.id
                        }) as relationships
                    """
                
                result = session.run(retrieval_query, node_id=node_id).single()
                
                if result:
                    return {
                        'graph_context': {
                            'relationships': result.get('relationships', []),
                            'node_properties': dict(result.get('node', {})) if result.get('node') else {}
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Graph enrichment failed: {e}")
        
        return {}
    
    def _neo4j_semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Fallback semantic search using Neo4j embeddings (original implementation).
        """
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return []
        
        # Continue with Neo4j vector search (reuse existing code from semantic_search)