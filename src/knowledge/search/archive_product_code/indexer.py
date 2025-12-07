#!/usr/bin/env python3
"""
Wiki Indexer - Indexes wiki pages to Neo4j with Weaviate vector storage.

Supports two modes:
1. Weaviate + Neo4j: Graph in Neo4j, vectors in Weaviate (default, recommended for scale)
2. Neo4j-only: Graph + embeddings in Neo4j (legacy mode)

Usage:
    from wiki_indexer import WikiIndexer
    
    # Default mode: Weaviate + Neo4j with embeddings (recommended)
    indexer = WikiIndexer(neo4j_uri, neo4j_user, neo4j_password, domain="acme.leeroo.com")
    indexer.index_page(page_title, wikitext, use_embeddings=True)
    
    # Without embeddings (text-only indexing)
    indexer.index_page(page_title, wikitext, use_embeddings=False)
    
    # Neo4j-only mode without Weaviate (legacy)
    indexer = WikiIndexer(
        neo4j_uri, neo4j_user, neo4j_password,
        domain="acme.leeroo.com",
        use_weaviate=False
    )
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from neo4j import GraphDatabase

# Import parser
from .parser import WikiParser, WikiNode, WikiEdge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiIndexer:
    """Indexes wiki pages and relationships into Neo4j with Weaviate vector storage (default)"""
    
    def __init__(self, uri: str, user: str, password: str, 
                 domain: str,
                 database: str = "neo4j",
                 openai_api_key: Optional[str] = None,
                 use_weaviate: bool = True,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_collection: str = "WikiPages"):
        """
        Initialize indexer with Neo4j and Weaviate.
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
            domain: Wiki domain (e.g., 'acme.leeroo.com') - used for prefixing IDs
            database: Neo4j database name (default: "neo4j")
            openai_api_key: Optional OpenAI API key for embeddings
            use_weaviate: Whether to use Weaviate for vector storage (default: True)
            weaviate_url: Weaviate server URL
            weaviate_collection: Name of Weaviate collection for wiki pages
        """
        # Note: database parameter is not set on driver, but on session
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.domain = domain  # Used for prefixing all node IDs
        self.parser = WikiParser()
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Weaviate configuration
        self.use_weaviate = use_weaviate
        self.weaviate_collection_name = weaviate_collection
        self.weaviate_client = None
        
        # Initialize OpenAI if available
        self.embeddings_enabled = False
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                self.embeddings_enabled = True
                logger.info("OpenAI embeddings enabled")
            except ImportError:
                logger.warning("OpenAI library not installed - embeddings disabled")
        
        # Initialize Weaviate if requested
        if use_weaviate:
            try:
                import weaviate
                import weaviate.classes as wvc
                
                # Connect to Weaviate
                self.weaviate_client = weaviate.connect_to_local(
                    host=weaviate_url.replace("http://", "").replace(":8080", ""),
                    port=8080
                )
                
                # Create collection if it doesn't exist
                self._ensure_weaviate_collection()
                logger.info(f"Connected to Weaviate at {weaviate_url}")
                
            except ImportError:
                logger.warning("Weaviate client not installed - pip install weaviate-client")
                self.use_weaviate = False
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {e}")
                self.use_weaviate = False
        
        logger.info(f"Connected to Neo4j at {uri}")
        if self.use_weaviate:
            logger.info("Using Weaviate for vector storage")
    
    def close(self):
        """Close connections"""
        self.driver.close()
        if self.weaviate_client:
            self.weaviate_client.close()
    
    def _ensure_weaviate_collection(self):
        """Create Weaviate collection if it doesn't exist"""
        try:
            import weaviate.classes as wvc
            
            # Check if collection exists
            collections = self.weaviate_client.collections.list_all()
            if self.weaviate_collection_name in collections:
                logger.info(f"Weaviate collection '{self.weaviate_collection_name}' already exists")
                return
            
            # Create collection with properties for wiki pages
            self.weaviate_client.collections.create(
                name=self.weaviate_collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide embeddings
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="neo4j_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="sections", data_type=wvc.config.DataType.TEXT),  # JSON string
                ]
            )
            logger.info(f"Created Weaviate collection '{self.weaviate_collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create Weaviate collection: {e}")
    
    def create_indexes(self):
        """Create Neo4j indexes and constraints"""
        with self.driver.session(database=self.database) as session:
            # Composite constraint on (domain, title) for uniqueness per wiki
            try:
                session.run("""
                    CREATE CONSTRAINT page_domain_title IF NOT EXISTS
                    FOR (p:WikiPage) REQUIRE (p.domain, p.title) IS NODE KEY
                """)
            except:
                # Fallback for older Neo4j versions
                session.run("""
                    CREATE CONSTRAINT page_id IF NOT EXISTS
                    FOR (p:WikiPage) REQUIRE p.id IS UNIQUE
                """)
            
            # Standard indexes
            session.run("CREATE INDEX page_domain IF NOT EXISTS FOR (p:WikiPage) ON (p.domain)")
            session.run("CREATE INDEX page_type IF NOT EXISTS FOR (p:WikiPage) ON (p.type)")
            session.run("CREATE INDEX page_name IF NOT EXISTS FOR (p:WikiPage) ON (p.name)")
            
            # Vector index for embeddings (if Neo4j supports it)
            if self.embeddings_enabled:
                try:
                    session.run("""
                        CREATE VECTOR INDEX page_embeddings IF NOT EXISTS
                        FOR (p:WikiPage)
                        ON (p.embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 3072,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    logger.info("Created vector index for embeddings")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {e}")
            
            logger.info("Created Neo4j indexes and constraints")
    
    def index_page(self, page_title: str, wikitext: str, 
                   use_embeddings: bool = True) -> Dict:
        """
        Index a wiki page with its relationships.
        
        Args:
            page_title: Title of the wiki page
            wikitext: Raw wikitext content
            use_embeddings: Whether to generate embeddings (default: True)
            
        Returns:
            Dictionary with indexing results
        """
        # Parse the page
        node, edges = self.parser.parse(page_title, wikitext)
        
        # Index the node
        node_result = self._index_node(node, wikitext, use_embeddings)
        
        # Index the edges
        edges_result = self._index_edges(edges)
        
        return {
            'node': node_result,
            'edges': edges_result,
            'statistics': {
                'type': node.type.value,
                'sections': len(node.sections),
                'categories': len(node.categories),
                'outgoing_links': len(edges),
                'has_embedding': node_result.get('has_embedding', False)
            }
        }
    
    def _index_node(self, node: WikiNode, wikitext: str, 
                    use_embeddings: bool) -> Dict:
        """Index a node in Neo4j and optionally Weaviate"""
        # Create domain-prefixed ID
        prefixed_id = f"{self.domain}:{node.id}"
        
        properties = {
            'id': prefixed_id,
            'domain': self.domain,
            'title': node.id,  # Original page title without prefix
            'type': node.type.value,
            'name': node.name,
            'categories': node.categories,
            'section_count': len(node.sections),
            'has_code': node.metadata.get('code_blocks', 0) > 0,
            'has_prerequisites': node.metadata.get('has_prerequisites', False),
            'has_examples': node.metadata.get('has_examples', False)
        }
        
        embedding = None
        has_embedding = False
        
        # Generate embedding if requested and available
        if use_embeddings and self.embeddings_enabled:
            embedding = self._generate_embedding(node, wikitext)
            has_embedding = embedding is not None
            
            if self.use_weaviate and embedding:
                # Store embedding in Weaviate
                self._index_to_weaviate(node, embedding, wikitext)
                # Don't store large embedding in Neo4j if using Weaviate
                properties['has_embedding'] = True
                properties['weaviate_indexed'] = True
            elif embedding and not self.use_weaviate:
                # Store embedding in Neo4j (original behavior)
                properties['embedding'] = embedding
                properties['has_embedding'] = True
            else:
                properties['has_embedding'] = False
        
        # Store in Neo4j (graph structure and metadata)
        with self.driver.session(database=self.database) as session:
            query = """
                MERGE (p:WikiPage {id: $id})
                SET p += $properties
                RETURN p.id
            """
            result = session.run(query, id=prefixed_id, properties=properties).single()
            
        logger.info(f"Indexed node: {prefixed_id} ({node.type.value}) - Weaviate: {self.use_weaviate and has_embedding}")
        
        return {
            'id': prefixed_id,
            'type': node.type.value,
            'has_embedding': has_embedding,
            'weaviate_indexed': self.use_weaviate and has_embedding
        }
    
    def _index_to_weaviate(self, node: WikiNode, embedding: List[float], wikitext: str):
        """Store node embedding in Weaviate"""
        try:
            collection = self.weaviate_client.collections.get(self.weaviate_collection_name)
            
            # Prepare text for storage
            embedding_text = self._create_embedding_text(node, wikitext)
            
            # Prepare properties
            weaviate_properties = {
                "text": embedding_text,
                "title": node.name,
                "type": node.type.value,
                "neo4j_id": node.id,
                "sections": json.dumps(list(node.sections.keys()))  # Store section names
            }
            
            # Check if object already exists (update vs create)
            # For simplicity, we'll delete and recreate
            try:
                # Try to delete existing
                collection.data.delete_many(
                    where={"path": ["neo4j_id"], "operator": "Equal", "valueText": node.id}
                )
            except:
                pass  # Object might not exist
            
            # Add object with embedding
            collection.data.insert(
                properties=weaviate_properties,
                vector=embedding
            )
            
            logger.debug(f"Indexed to Weaviate: {node.id}")
            
        except Exception as e:
            logger.error(f"Failed to index to Weaviate: {e}")
    
    def _index_edges(self, edges: List[WikiEdge]) -> Dict:
        """Index edges in Neo4j"""
        relationship_counts = {}
        
        with self.driver.session(database=self.database) as session:
            for edge in edges:
                # Create domain-prefixed IDs
                source_prefixed = f"{self.domain}:{edge.source_id}"
                target_prefixed = f"{self.domain}:{edge.target_id}"
                
                # Ensure target node exists with domain
                session.run("""
                    MERGE (p:WikiPage {id: $id})
                    ON CREATE SET p.type = 'unknown', 
                                  p.name = $title,
                                  p.title = $title,
                                  p.domain = $domain
                """, id=target_prefixed, title=edge.target_id, domain=self.domain)
                
                # Create relationship
                rel_type = edge.relationship.name
                query = f"""
                    MATCH (source:WikiPage {{id: $source_id}})
                    MATCH (target:WikiPage {{id: $target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.section = $section,
                        r.link_text = $link_text,
                        r.context = $context,
                        r.domain = $domain
                    RETURN r
                """
                
                session.run(query,
                          source_id=source_prefixed,
                          target_id=target_prefixed,
                          section=edge.section,
                          link_text=edge.link_text,
                          context=edge.context[:200],  # Limit context length
                          domain=self.domain)
                
                # Count relationships by type
                rel_value = edge.relationship.value
                relationship_counts[rel_value] = relationship_counts.get(rel_value, 0) + 1
        
        logger.info(f"Indexed {len(edges)} edges")
        
        return {
            'total': len(edges),
            'by_type': relationship_counts
        }
    
    def _generate_embedding(self, node: WikiNode, wikitext: str) -> Optional[List[float]]:
        """Generate OpenAI embedding for a node"""
        if not self.embeddings_enabled:
            return None
        
        try:
            import openai
            
            # Create embedding text
            embedding_text = self._create_embedding_text(node, wikitext)
            
            # Generate embedding
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=embedding_text[:8000]  # Limit length
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for {node.id} (dim: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for {node.id}: {e}")
            return None
    
    def _create_embedding_text(self, node: WikiNode, wikitext: str) -> str:
        """Create text representation for embedding"""
        parts = []
        
        # Add title and type
        parts.append(f"Title: {node.name}")
        parts.append(f"Type: {node.type.value}")
        
        # Add categories
        if node.categories:
            parts.append(f"Categories: {', '.join(node.categories)}")
        
        # Add key sections
        priority_sections = ['Overview', 'Introduction', 'Description']
        for section_name in priority_sections:
            if section_name in node.sections:
                content = self._clean_wikitext(node.sections[section_name])
                if content:
                    parts.append(f"{section_name}: {content[:500]}")
        
        # Add some other sections
        other_sections = [s for s in node.sections.keys() if s not in priority_sections]
        for section_name in other_sections[:2]:
            content = self._clean_wikitext(node.sections[section_name])
            if content:
                parts.append(f"{section_name}: {content[:300]}")
        
        return "\n\n".join(parts)
    
    def _clean_wikitext(self, text: str) -> str:
        """Remove wiki markup for cleaner text"""
        import re
        
        # Remove wiki links but keep text
        text = re.sub(r'\[\[(?:[^\|]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'''?", '', text)
        text = re.sub(r'^[*#]+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session(database=self.database) as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("""
                MATCH (p:WikiPage)
                RETURN p.type as type, COUNT(p) as count
                ORDER BY count DESC
            """)
            stats['nodes_by_type'] = {record['type']: record['count'] for record in result}
            
            # Count relationships
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(r) as count
                ORDER BY count DESC
            """)
            stats['edges_by_type'] = {record['type']: record['count'] for record in result}
            
            # Total counts
            result = session.run("MATCH (p:WikiPage) RETURN COUNT(p) as count").single()
            stats['total_nodes'] = result['count']
            
            result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count").single()
            stats['total_edges'] = result['count']
            
            # Embeddings count
            if self.embeddings_enabled:
                result = session.run("""
                    MATCH (p:WikiPage)
                    WHERE p.embedding IS NOT NULL
                    RETURN COUNT(p) as count
                """).single()
                stats['nodes_with_embeddings'] = result['count']
            
            return stats
    
    def rebuild_index(self, pages: List[Dict[str, str]], use_embeddings: bool = True):
        """
        Rebuild entire index from scratch.
        
        Args:
            pages: List of dicts with 'title' and 'content' keys
            use_embeddings: Whether to generate embeddings (default: True)
        """
        # Clear existing data
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing graph")
        
        # Recreate indexes
        self.create_indexes()
        
        # Index all pages
        for i, page in enumerate(pages, 1):
            logger.info(f"Indexing {i}/{len(pages)}: {page['title']}")
            self.index_page(page['title'], page['content'], use_embeddings)
        
        # Final statistics
        stats = self.get_statistics()
        logger.info(f"Rebuild complete: {stats['total_nodes']} nodes, {stats['total_edges']} edges")