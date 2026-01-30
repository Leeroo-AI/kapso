# KG Graph Search
#
# Hybrid search using Weaviate for embeddings and Neo4j for graph structure.
# 
# Architecture:
# - Weaviate: Stores embeddings (from page overview) for semantic search
# - Neo4j: Stores graph structure (nodes + edges) for connection enrichment
#
# Search Flow:
# 1. Query -> Generate embedding -> Weaviate vector search
# 2. Get top-k similar pages from Weaviate
# 3. Enrich results with connected pages from Neo4j
# 4. Return KGOutput with results and connections
#
# Environment Variables:
# - OPENAI_API_KEY: For text-embedding-3-large
# - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD: Graph database
# - WEAVIATE_URL: Vector database (default: http://localhost:8080)

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional dependencies (may not be installed)
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    GraphDatabase = None
    HAS_NEO4J = False

try:
    import weaviate
    import weaviate.classes as wvc
    HAS_WEAVIATE = True
except ImportError:
    weaviate = None
    wvc = None
    HAS_WEAVIATE = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

from kapso.knowledge_base.search.base import (
    KGEditInput,
    KGIndexInput,
    KGOutput,
    KGResultItem,
    KGSearchFilters,
    KnowledgeSearch,
    PageType,
    WikiPage,
)
from kapso.knowledge_base.search.factory import register_knowledge_search
from kapso.core.llm import LLMBackend

logger = logging.getLogger(__name__)


# =============================================================================
# ID Normalization
# =============================================================================
# Wiki files use spaces in names: "huggingface peft LoRA Configuration.md"
# But wiki links use underscores: [[step::Principle:huggingface_peft_LoRA_Configuration]]
# This function normalizes both to use spaces for consistent matching.

def normalize_page_id(page_id: str) -> str:
    """
    Normalize a page ID for consistent matching.
    
    Converts underscores to spaces in the page name part (after the type prefix).
    Example: "Principle/huggingface_peft_LoRA_Configuration" 
          -> "Principle/huggingface peft LoRA Configuration"
    
    This ensures wiki links match actual wiki file names.
    """
    if "/" in page_id:
        page_type, name = page_id.split("/", 1)
        # Convert underscores to spaces in the name part only
        normalized_name = name.replace("_", " ")
        return f"{page_type}/{normalized_name}"
    else:
        # No type prefix, just normalize the whole string
        return page_id.replace("_", " ")


# =============================================================================
# Wiki Parser Functions
# =============================================================================

# Type subdirectory names mapped to PageType values.
# Used by the wiki parser to determine page types from directory structure.
_TYPE_SUBDIRS = {
    "workflows": PageType.WORKFLOW.value,
    "principles": PageType.PRINCIPLE.value,
    "implementations": PageType.IMPLEMENTATION.value,
    "environments": PageType.ENVIRONMENT.value,
    "heuristics": PageType.HEURISTIC.value,
}

# Reverse mapping: PageType value -> subdirectory name
# Used when writing back to source files
_TYPE_TO_SUBDIR = {
    PageType.WORKFLOW.value: "workflows",
    PageType.PRINCIPLE.value: "principles",
    PageType.IMPLEMENTATION.value: "implementations",
    PageType.ENVIRONMENT.value: "environments",
    PageType.HEURISTIC.value: "heuristics",
}


def parse_wiki_directory(
    wiki_dir: Path,
    domain_file: str = "domain_tag.txt",
) -> List[WikiPage]:
    """
    Parse wiki files from a directory into WikiPage objects.
    
    Supports two directory structures:
    1. Flat structure: *.mediawiki files in root
    2. Type subdirectories: workflows/, principles/, etc. with *.md files
    
    Args:
        wiki_dir: Path to directory containing wiki files
        domain_file: Name of file containing default domain tags
        
    Returns:
        List of parsed WikiPage objects
    
    Example:
        # Type subdirectories
        pages = parse_wiki_directory(Path("data/wikis"))
        
        # Flat .mediawiki structure
        pages = parse_wiki_directory(Path("data/wikis/allenai_allennlp"))
    """
    wiki_dir = Path(wiki_dir)
    
    # Check if directory uses type subdirectories
    has_type_subdirs = any((wiki_dir / subdir).exists() for subdir in _TYPE_SUBDIRS)
    
    if has_type_subdirs:
        return _parse_type_subdirectories(wiki_dir)
    else:
        return _parse_flat_mediawiki(wiki_dir, domain_file)


def _parse_type_subdirectories(wiki_dir: Path) -> List[WikiPage]:
    """
    Parse wiki files organized in type subdirectories.
    
    Structure: wiki_dir/workflows/*.md, wiki_dir/principles/*.md, etc.
    Each subdirectory name determines the page type.
    """
    pages = []
    
    for subdir_name, page_type in _TYPE_SUBDIRS.items():
        subdir_path = wiki_dir / subdir_name
        if not subdir_path.exists():
            continue
        
        # Parse all .md files in this type subdirectory
        for wiki_file in sorted(subdir_path.glob("*.md")):
            page = _parse_typed_md_file(wiki_file, page_type)
            if page:
                pages.append(page)
    
    return pages


def _parse_typed_md_file(file_path: Path, page_type: str) -> Optional[WikiPage]:
    """
    Parse a .md file from a type subdirectory.
    
    The page type is determined by the parent directory name.
    The file uses MediaWiki formatting inside .md extension.
    """
    content = file_path.read_text(encoding="utf-8")
    filename = file_path.stem  # e.g., "QLoRA_Finetuning"
    
    # Construct identifier from type and filename
    identifier = f"{page_type}/{filename}"
    
    # Extract title (fallback to filename with underscores replaced)
    title = _extract_title(content, filename)
    
    # Extract overview/definition
    overview = _extract_overview(content)
    
    # Extract domains from [[domain::...]] tags
    domains = _extract_domain_tags(content)
    
    # Extract last updated
    last_updated = _extract_last_updated_tag(content)
    
    # Extract sources
    sources = _extract_sources(content)
    
    # Extract semantic links
    outgoing_links = _extract_links(content)
    
    return WikiPage(
        id=identifier,
        page_title=title,
        page_type=page_type,
        overview=overview,
        content=content,
        domains=domains,
        sources=sources,
        last_updated=last_updated,
        outgoing_links=outgoing_links,
    )


def _parse_flat_mediawiki(wiki_dir: Path, domain_file: str) -> List[WikiPage]:
    """
    Parse flat directory with .mediawiki files.
    
    Original parser for legacy structure.
    """
    pages = []
    
    # Load default domains from domain_tag.txt
    default_domains = []
    domain_path = wiki_dir / domain_file
    if domain_path.exists():
        default_domains = [
            d.strip() for d in domain_path.read_text().splitlines()
            if d.strip()
        ]
    
    # Extract repo_id from directory name
    repo_id = wiki_dir.name
    
    # Parse each .mediawiki file
    for wiki_file in sorted(wiki_dir.glob("*.mediawiki")):
        page = parse_wiki_file(wiki_file, repo_id, default_domains)
        if page:
            pages.append(page)
    
    return pages


def parse_wiki_file(
    file_path: Path,
    repo_id: str,
    default_domains: List[str] = None,
) -> Optional[WikiPage]:
    """
    Parse a single .mediawiki file into a WikiPage.
    
    Extracts:
    - Page type from filename prefix (e.g., "Workflow_Model_Training" -> "Workflow")
    - Metadata from wikitable (Identifier, Domains, Last Updated, etc.)
    - Overview/Definition section
    - Full content
    - Semantic links [[edge_type::Type:Target]]
    
    Args:
        file_path: Path to .mediawiki file
        repo_id: Repository/namespace identifier
        default_domains: Default domain tags if not specified in file
        
    Returns:
        WikiPage object, or None if file cannot be parsed
    """
    content = file_path.read_text(encoding="utf-8")
    filename = file_path.stem  # e.g., "Workflow_Model_Training"
    
    # Extract page type from filename prefix
    page_type = _extract_page_type(filename)
    if not page_type:
        return None  # Skip non-standard files (e.g., domain_tag.txt)
    
    # Extract identifier
    identifier = _extract_identifier(content, filename, repo_id)
    
    # Extract title from first header
    title = _extract_title(content, filename)
    
    # Extract overview/definition
    overview = _extract_overview(content)
    
    # Extract domains from metadata or use defaults
    domains = _extract_domains(content) or default_domains or []
    
    # Extract last updated
    last_updated = _extract_last_updated(content)
    
    # Extract sources (repo URLs, etc.)
    sources = _extract_sources(content)
    
    # Extract semantic links
    outgoing_links = _extract_links(content)
    
    return WikiPage(
        id=identifier,
        page_title=title,
        page_type=page_type,
        overview=overview,
        content=content,
        domains=domains,
        sources=sources,
        last_updated=last_updated,
        outgoing_links=outgoing_links,
    )


def _extract_page_type(filename: str) -> Optional[str]:
    """
    Extract page type from filename prefix.
    
    Mapping follows wiki_structure definitions:
    - Workflow_ -> Workflow (The Recipe)
    - Principle_ -> Principle (The Theory)
    - Implementation_ -> Implementation (The Code)
    - Environment_ -> Environment (The Context)
    - Heuristic_ -> Heuristic (The Wisdom)
    """
    type_mapping = {
        "Workflow_": PageType.WORKFLOW.value,
        "Principle_": PageType.PRINCIPLE.value,
        "Implementation_": PageType.IMPLEMENTATION.value,
        "Environment_": PageType.ENVIRONMENT.value,
        "Heuristic_": PageType.HEURISTIC.value,
        # Additional types found in wikis
        "Artifact_": "Artifact",
        "Resource_": "Resource",
    }
    
    for prefix, page_type in type_mapping.items():
        if filename.startswith(prefix):
            return page_type
    return None


def _extract_title(content: str, filename: str) -> str:
    """
    Extract title from = Title = header or generate from filename.
    
    Example: "= Workflow: Model Training =" -> "Workflow: Model Training"
    """
    # Look for top-level header
    match = re.search(r'^= ([^=]+) =$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Fallback: convert filename to title
    # "Workflow_Model_Training" -> "Workflow Model Training"
    return filename.replace("_", " ")


def _extract_overview(content: str) -> str:
    """
    Extract overview/definition from first content section.
    
    Looks for:
    - == Overview == section
    - == Definition == section
    """
    patterns = [
        r'== Overview ==\s*\n+(.+?)(?=\n==|\n\{\{|\Z)',
        r'== Definition ==\s*\n+(.+?)(?=\n==|\n\{\{|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            overview = match.group(1).strip()
            # Clean up wiki formatting
            overview = re.sub(r'\[\[Category:[^\]]+\]\]', '', overview)
            overview = re.sub(r'\n+', ' ', overview)  # Collapse newlines
            return overview.strip()
    
    return ""


def _extract_identifier(content: str, filename: str, repo_id: str) -> str:
    """
    Extract identifier from metadata or construct from filename.
    
    Looks for: | Identifier || value in wikitable
    Fallback: repo_id/name_from_filename
    """
    # Look for Identifier in metadata table
    match = re.search(r'\|\|?\s*Identifier\s*\n\|\|?\s*([^\n|]+)', content)
    if match:
        return f"{repo_id}/{match.group(1).strip()}"
    
    # Construct from filename: remove type prefix
    parts = filename.split("_", 1)
    name = parts[1] if len(parts) > 1 else filename
    return f"{repo_id}/{name}"


def _extract_domains(content: str) -> List[str]:
    """
    Extract domain tags from metadata table.
    
    Looks for: | Domain(s) || value1, value2
    """
    # Try different patterns for domain row
    patterns = [
        r'Domain\(s\)\s*\n\|\|?\s*([^\n|]+)',
        r'\|\|?\s*Domain\(s\)\s*\n\|\|?\s*([^\n|]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            domains_str = match.group(1).strip()
            return [d.strip() for d in domains_str.split(",") if d.strip()]
    
    return []


def _extract_domain_tags(content: str) -> List[str]:
    """
    Extract domain tags from [[domain::...]] wiki links.
    
    Used in data/wikis format where domains are in metadata table like:
    | [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
    """
    domains = []
    
    # Pattern: [[domain::Name]]
    pattern = r'\[\[domain::([^\]]+)\]\]'
    for match in re.finditer(pattern, content):
        domain = match.group(1).strip()
        # Convert underscores to spaces for consistency
        domain = domain.replace("_", " ")
        if domain and domain not in domains:
            domains.append(domain)
    
    return domains


def _extract_last_updated_tag(content: str) -> Optional[str]:
    """
    Extract last updated from [[last_updated::...]] wiki link.
    
    Format: [[last_updated::2025-12-12 00:00 GMT]]
    """
    match = re.search(r'\[\[last_updated::([^\]]+)\]\]', content)
    if match:
        return match.group(1).strip()
    return None


def _extract_last_updated(content: str) -> Optional[str]:
    """
    Extract last updated timestamp from metadata.
    
    Looks for: | Last Updated || YYYY-MM-DD HH:MM GMT
    """
    match = re.search(r'Last Updated\s*\n\|\|?\s*([^\n|]+)', content)
    if match:
        return match.group(1).strip()
    return None


def _extract_sources(content: str) -> List[Dict[str, str]]:
    """
    Extract source URLs from metadata.
    
    Looks for:
    - Repo URL links
    - Knowledge Sources links
    """
    sources = []
    
    # Extract repo URL: [https://github.com/... repo_name]
    repo_match = re.search(r'Repo URL\s*\n\|\|?\s*\[([^\s\]]+)\s+([^\]]+)\]', content)
    if repo_match:
        sources.append({
            "type": "Repo",
            "url": repo_match.group(1),
            "title": repo_match.group(2).strip(),
        })
    
    # Extract knowledge sources: [[source::Type|Title|URL]]
    source_pattern = r'\[\[source::(\w+)\|([^|]+)\|([^\]]+)\]\]'
    for match in re.finditer(source_pattern, content):
        sources.append({
            "type": match.group(1),
            "title": match.group(2),
            "url": match.group(3),
        })
    
    return sources


def _extract_links(content: str) -> List[Dict[str, str]]:
    """
    Extract semantic wiki links (graph edges).
    
    Patterns:
    - [[step::Principle:repo/Name]]
    - [[realized_by::Implementation:repo/Name]]
    - [[implemented_by::Implementation:repo/Name]]
    - [[uses_heuristic::Heuristic:repo/Name]]
    - [[requires_env::Environment:repo/Name]]
    - [[consumes::Artifact:repo/Name]]
    - [[produces::Artifact:repo/Name]]
    """
    links = []
    
    # Pattern: [[edge_type::TargetType:target_id]]
    pattern = r'\[\[(\w+)::(\w+):([^\]]+)\]\]'
    
    for match in re.finditer(pattern, content):
        links.append({
            "edge_type": match.group(1),
            "target_type": match.group(2),
            "target_id": match.group(3),
        })
    
    return links


# =============================================================================
# KG Graph Search Implementation
# =============================================================================

@register_knowledge_search("kg_graph_search")
class KGGraphSearch(KnowledgeSearch):
    """
    Hybrid search using Weaviate (embeddings) + Neo4j (graph structure).
    
    Weaviate handles semantic search via embeddings on page overview.
    Neo4j stores graph structure for connection enrichment.
    
    Environment Variables:
        OPENAI_API_KEY: For text-embedding-3-large embeddings
        NEO4J_URI: Neo4j connection URI (e.g., bolt://localhost:7687)
        NEO4J_USER: Neo4j username
        NEO4J_PASSWORD: Neo4j password
        WEAVIATE_URL: Weaviate server URL (default: http://localhost:8080)
    """
    
    # =========================================================================
    # Class Constants
    # =========================================================================
    
    # Graph edge types from wiki structure (non-configurable)
    # Note: STEP removed - Workflows now link to GitHub repos instead of Principles
    EDGE_TYPES = ["IMPLEMENTED_BY", "USES_HEURISTIC", "REQUIRES_ENV"]
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize KG Graph Search.
        
        Args:
            params: Configuration parameters (defaults from knowledge_search.yaml):
                - embedding_model: OpenAI embedding model
                - weaviate_collection: Weaviate collection name
                - include_connected_pages: Whether to include graph connections
                - use_llm_reranker: Whether to use LLM reranker
                - reranker_model: LLM model for reranking
        """
        super().__init__(params=params)
        
        # Extract params (defaults come from knowledge_search.yaml via factory)
        self.embedding_model = self.params.get("embedding_model")
        self.weaviate_collection = self.params.get("weaviate_collection")
        self.reranker_model = self.params.get("reranker_model")
        self.include_connected_pages = self.params.get("include_connected_pages", True)
        self.use_llm_reranker = self.params.get("use_llm_reranker", True)
        
        # Truncation limits MUST be config-driven (never hardcoded in code).
        # These exist only to satisfy upstream API constraints and prompt size.
        self.embedding_max_input_chars = self._coerce_positive_int(
            self.params.get("embedding_max_input_chars")
        )
        self.reranker_overview_max_chars = self._coerce_positive_int(
            self.params.get("reranker_overview_max_chars")
        )
        
        # Clients (initialized lazily)
        self._neo4j_driver = None
        self._weaviate_client = None
        self._openai_client = None
        self._llm_backend = None
        
        # Cached pages (for save/load)
        self._pages: List[WikiPage] = []
        
        self._initialize_clients()
    
    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        """Parse an optional positive int from config; return None if unset/invalid."""
        try:
            if value is None:
                return None
            n = int(value)
            return n if n > 0 else None
        except Exception:
            return None
    
    @staticmethod
    def _maybe_truncate(text: str, max_chars: Optional[int]) -> str:
        """
        Optionally truncate text to max_chars.
        
        This is intentionally centralized and config-driven. Do NOT add ad-hoc
        `[:N]` slices in other code paths.
        """
        if not text:
            return ""
        if max_chars is None:
            return text
        return text[:max_chars]
    
    # =========================================================================
    # Client Initialization
    # =========================================================================
    
    def _initialize_clients(self) -> None:
        """Initialize Neo4j, Weaviate, OpenAI, and LLM clients."""
        self._initialize_neo4j()
        self._initialize_weaviate()
        self._initialize_openai()
        self._initialize_llm()
    
    def _initialize_neo4j(self) -> None:
        """Initialize Neo4j driver."""
        if not HAS_NEO4J:
            logger.warning("neo4j package not installed. Graph features disabled.")
            return
            
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Connected to Neo4j at {uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
    
    def _initialize_weaviate(self) -> None:
        """Initialize Weaviate client."""
        if not HAS_WEAVIATE:
            logger.warning("weaviate-client package not installed.")
            return
            
        try:
            url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
            # Parse host and port from URL
            host = url.replace("http://", "").replace("https://", "").split(":")[0]
            port = 8080
            if ":" in url.replace("http://", "").replace("https://", ""):
                port = int(url.split(":")[-1])
            
            self._weaviate_client = weaviate.connect_to_local(host=host, port=port)
            logger.info(f"Connected to Weaviate at {url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI client."""
        if not HAS_OPENAI:
            logger.warning("openai package not installed.")
            return
            
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. Embeddings disabled.")
                return
            
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM backend for reranking."""
        if self.use_llm_reranker:
            self._llm_backend = LLMBackend()
            logger.info(f"LLM reranker initialized (model: {self.reranker_model})")
    
    # =========================================================================
    # Index Methods
    # =========================================================================
    
    def index(self, data: KGIndexInput) -> None:
        """
        Index wiki pages into Weaviate and Neo4j.
        
        Args:
            data: KGIndexInput with wiki_dir or pages
            
        If data.wiki_dir is provided, parses wiki files from the directory.
        If data.pages is provided, indexes the pre-parsed WikiPage objects.
        If data.persist_path is set, saves the parsed pages for later loading.
        """
        # Get pages from input
        if data.pages:
            pages = data.pages
        elif data.wiki_dir:
            logger.info(f"Parsing wiki pages from {data.wiki_dir}")
            pages = parse_wiki_directory(data.wiki_dir)
            logger.info(f"Parsed {len(pages)} wiki pages")
        else:
            raise ValueError("Must provide either wiki_dir or pages")
        
        # Store for caching
        self._pages = pages
        
        # Index to Neo4j (graph structure)
        self._index_to_neo4j(pages)
        
        # Index to Weaviate (embeddings)
        self._index_to_weaviate(pages)
        
        # Save parsed pages if persist_path provided
        if data.persist_path:
            self._save_parsed_pages(data.persist_path)
        
        logger.info(f"Indexed {len(pages)} pages to Neo4j and Weaviate")
    
    def _index_to_neo4j(self, pages: List[WikiPage]) -> None:
        """
        Index pages to Neo4j (graph structure only).
        
        Creates WikiPage nodes with minimal properties for filtering.
        Creates edges based on outgoing_links.
        """
        if not self._neo4j_driver:
            logger.warning("Neo4j not available. Skipping graph indexing.")
            return
        
        with self._neo4j_driver.session() as session:
            # Create indexes first
            self._create_neo4j_indexes(session)
            
            # Index nodes
            for page in pages:
                self._create_neo4j_node(session, page)
            
            # Index edges
            for page in pages:
                self._create_neo4j_edges(session, page)
        
        logger.info(f"Indexed {len(pages)} nodes to Neo4j")
    
    def _create_neo4j_indexes(self, session) -> None:
        """Create Neo4j indexes for efficient querying."""
        try:
            session.run(
                "CREATE INDEX page_id IF NOT EXISTS FOR (p:WikiPage) ON (p.id)"
            )
            session.run(
                "CREATE INDEX page_type IF NOT EXISTS FOR (p:WikiPage) ON (p.page_type)"
            )
            logger.debug("Created Neo4j indexes")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def _create_neo4j_node(self, session, page: WikiPage) -> None:
        """Create a WikiPage node in Neo4j."""
        query = """
            MERGE (p:WikiPage {id: $id})
            SET p.page_type = $page_type,
                p.page_title = $page_title,
                p.domains = $domains
        """
        session.run(
            query,
            id=page.id,
            page_type=page.page_type,
            page_title=page.page_title,
            domains=page.domains,
        )
    
    def _create_neo4j_edges(self, session, page: WikiPage) -> None:
        """
        Create edges from page's outgoing_links.
        
        SPECIAL HANDLING FOR HEURISTIC BACKLINKS:
        When a Heuristic page declares [[uses_heuristic::Principle:X]], it's a
        "backlink" - meaning "Principle X uses me". Per the KG spec, the correct
        edge direction is Principle → USES_HEURISTIC → Heuristic.
        
        So when we see a uses_heuristic link FROM a Heuristic page, we create
        the edge in REVERSE direction (target → source) instead of (source → target).
        """
        for link in page.outgoing_links:
            edge_type = link.get("edge_type", "RELATED").upper()
            target_type = link.get("target_type", "")
            target_id = link.get("target_id", "")
            
            # Construct and NORMALIZE target page ID
            # Wiki links use underscores but file names use spaces
            raw_target_id = f"{target_type}/{target_id}"
            target_page_id = normalize_page_id(raw_target_id)
            
            # Also normalize the title for display
            normalized_title = target_id.replace("_", " ")
            
            # Map edge types to Neo4j relationship types
            neo4j_rel_type = self._map_edge_type(edge_type)
            
            # SPECIAL CASE: Heuristic backlinks
            # When a Heuristic page declares [[uses_heuristic::Principle:X]],
            # it means "Principle X uses this heuristic", so we reverse the edge.
            is_heuristic_backlink = (
                page.page_type == "Heuristic" and 
                edge_type.lower() == "uses_heuristic" and
                target_type in ("Workflow", "Principle", "Implementation")
            )
            
            if is_heuristic_backlink:
                # Create REVERSE edge: target → USES_HEURISTIC → source (this heuristic)
                query = f"""
                MERGE (target:WikiPage {{id: $target_id}})
                    ON CREATE SET target.page_type = $target_type, target.page_title = $target_title
                    WITH target
                    MATCH (source:WikiPage {{id: $source_id}})
                    MERGE (target)-[r:{neo4j_rel_type}]->(source)
                """
                logger.debug(f"Creating reverse heuristic edge: {target_page_id} → {page.id}")
            else:
                # Normal edge: source → target
                query = f"""
                    MERGE (target:WikiPage {{id: $target_id}})
                    ON CREATE SET target.page_type = $target_type, target.page_title = $target_title
                    WITH target
                    MATCH (source:WikiPage {{id: $source_id}})
                    MERGE (source)-[r:{neo4j_rel_type}]->(target)
                """
            
            try:
                session.run(
                    query,
                    source_id=page.id,
                    target_id=target_page_id,
                    target_type=target_type,
                    target_title=normalized_title,
                )
            except Exception as e:
                logger.warning(f"Failed to create edge {page.id} -> {target_page_id}: {e}")
    
    def _map_edge_type(self, edge_type: str) -> str:
        """Map wiki edge types to Neo4j relationship types."""
        mapping = {
            # Note: "step" removed - Workflows now use github_url instead
            "implemented_by": "IMPLEMENTED_BY",
            "uses_heuristic": "USES_HEURISTIC",
            "requires_env": "REQUIRES_ENV",
            "realized_by": "IMPLEMENTED_BY",
            "consumes": "CONSUMES",
            "produces": "PRODUCES",
            "github_url": "GITHUB_URL",  # External link to GitHub repository
        }
        return mapping.get(edge_type.lower(), "RELATED")
    
    def _index_to_weaviate(self, pages: List[WikiPage]) -> None:
        """
        Index pages to Weaviate with embeddings.
        
        Generates embeddings from page overview using OpenAI.
        """
        if not self._weaviate_client:
            logger.warning("Weaviate not available. Skipping vector indexing.")
            return
        
        if not self._openai_client:
            logger.warning("OpenAI not available. Cannot generate embeddings.")
            return
        
        # Ensure collection exists
        self._ensure_weaviate_collection()
        
        # Get collection
        collection = self._weaviate_client.collections.get(self.weaviate_collection)
        
        # Index each page
        indexed_count = 0
        for page in pages:
            try:
                # Generate embedding from overview
                embedding = self._generate_embedding(page.overview)
                if not embedding:
                    continue
                
                # Prepare properties
                properties = {
                    "page_id": page.id,
                    "page_title": page.page_title,
                    "page_type": page.page_type,
                    "overview": page.overview,
                    "content": page.content,
                    "domains": page.domains,
                }
                
                # Insert with named vector (Weaviate 1.27+ with vectorizer_config=None
                # creates a named vector "default", so we must use this format)
                collection.data.insert(
                    properties=properties,
                    vector={"default": embedding},
                )
                indexed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to index page {page.id} to Weaviate: {e}")
        
        logger.info(f"Indexed {indexed_count} pages to Weaviate")
    
    def _ensure_weaviate_collection(self) -> None:
        """Create Weaviate collection if it doesn't exist."""
        if not HAS_WEAVIATE or not self._weaviate_client:
            return
            
        try:
            # Check if collection exists
            collections = self._weaviate_client.collections.list_all()
            if self.weaviate_collection in [c.name for c in collections.values()]:
                logger.debug(f"Collection '{self.weaviate_collection}' already exists")
                return
            
            # Create collection with no auto-vectorizer (we provide embeddings)
            # vectorizer_config=None creates a named vector "default" in Weaviate 1.27+
            self._weaviate_client.collections.create(
                name=self.weaviate_collection,
                vectorizer_config=None,
                properties=[
                    wvc.config.Property(name="page_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="page_title", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="page_type", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="overview", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="domains", data_type=wvc.config.DataType.TEXT_ARRAY),
                ],
            )
            logger.info(f"Created Weaviate collection '{self.weaviate_collection}'")
            
        except Exception as e:
            logger.error(f"Failed to ensure Weaviate collection: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI."""
        if not self._openai_client or not text:
            return None
        
        try:
            input_text = self._maybe_truncate(text, self.embedding_max_input_chars)
            response = self._openai_client.embeddings.create(
                model=self.embedding_model,
                input=input_text,
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    # =========================================================================
    # Parsed Data Persistence (Internal)
    # =========================================================================
    
    def _save_parsed_pages(self, path: Union[str, Path]) -> None:
        """
        Save parsed WikiPages to JSON file (internal use).
        
        Note: This only saves the parsed page data, not the actual indexed
        data in Weaviate/Neo4j. Used internally during indexing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "pages": [p.to_dict() for p in self._pages],
            "metadata": {
                "collection": self.weaviate_collection,
                "embedding_model": self.embedding_model,
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self._pages)} pages to {path}")
    
    def _load_parsed_pages(self, path: Union[str, Path]) -> None:
        """
        Load parsed WikiPages from JSON file (internal use).
        
        Note: This only loads the parsed page data into memory, not the
        actual indexed data. If Weaviate/Neo4j don't have the data,
        you'll need to call index() again.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._pages = [WikiPage.from_dict(p) for p in data.get("pages", [])]
        
        logger.info(f"Loaded {len(self._pages)} pages from {path}")
    
    # =========================================================================
    # Search Methods
    # =========================================================================
    
    def search(
        self,
        query: str,
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
        use_llm_reranker: Optional[bool] = None,
    ) -> KGOutput:
        """
        Search for relevant wiki pages.
        
        1. Semantic search in Weaviate using query embedding
        2. LLM reranking based on query relevance (if enabled)
        3. Apply filters (page_type, domains, min_score)
        4. Enrich with connected pages from Neo4j
        
        Args:
            query: Search query text
            filters: Optional filters (top_k, min_score, page_types, domains)
            context: Optional additional context (unused currently)
            use_llm_reranker: Override config-level use_llm_reranker setting.
                              If None, uses the instance setting from params.
            
        Returns:
            KGOutput with ranked results and graph connections
        """
        # Default filters
        if filters is None:
            filters = KGSearchFilters()
        
        # Resolve use_llm_reranker: parameter override > instance setting
        should_rerank = use_llm_reranker if use_llm_reranker is not None else self.use_llm_reranker
        
        # Semantic search in Weaviate (get more results for reranking)
        search_top_k = filters.top_k * 2 if should_rerank else filters.top_k
        search_filters = KGSearchFilters(
            top_k=search_top_k,
            min_score=filters.min_score,
            page_types=filters.page_types,
            domains=filters.domains,
            include_content=filters.include_content,
        )
        results = self._semantic_search(query, search_filters)
        
        # LLM reranking if enabled (check resolved should_rerank value)
        if should_rerank and self._llm_backend and len(results) > 0:
            results = self._rerank_results(query, results, filters.top_k)
        
        # Enrich with Neo4j connections if enabled
        if self.include_connected_pages and self._neo4j_driver:
            for result in results:
                connected = self._get_connected_pages(result.id)
                result.metadata["connected_pages"] = connected
        
        return KGOutput(
            query=query,
            filters=filters,
            results=results,
            total_found=len(results),
            search_metadata={
                "backend": "kg_graph_search",
                "collection": self.weaviate_collection,
                "reranked": should_rerank,
            },
        )
    
    def _rerank_results(
        self,
        query: str,
        results: List[KGResultItem],
        top_k: int,
    ) -> List[KGResultItem]:
        """
        Rerank search results using LLM.
        
        Uses the LLM to evaluate relevance of each result to the query
        and reorder them based on the LLM's assessment.
        
        Args:
            query: Original search query
            results: List of KGResultItem from semantic search
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of KGResultItem (top_k items)
        """
        if not results:
            return results
        
        # Build context for LLM
        pages_info = []
        for i, result in enumerate(results):
            overview_preview = self._maybe_truncate(result.overview or "", self.reranker_overview_max_chars)
            pages_info.append(
                f"[{i}] {result.page_title} ({result.page_type})\n"
                f"    Overview: {overview_preview}..."
            )
        
        pages_text = "\n\n".join(pages_info)
        
        # Reranking prompt
        system_prompt = """You are an expert search result reranker for a machine learning knowledge base.

Your task is to rerank wiki pages based on their relevance to a user's query. Consider:

1. **Direct Relevance**: Does the page directly answer or address the query?
2. **Page Type Appropriateness**: 
   - Workflow pages are best for "how to" questions
   - Principle pages are best for "what is" or theoretical questions
   - Implementation pages are best for code/API questions
   - Heuristic pages are best for best practices and optimization tips
   - Environment pages are best for setup/installation questions
3. **Specificity**: Prefer pages that specifically match the query over general pages
4. **Completeness**: Prefer pages whose overview indicates comprehensive coverage

Output your ranking as a comma-separated list of page indices inside <output_order> tags.
Only include pages that are genuinely relevant. Exclude irrelevant pages entirely.

Example:
<output_order>2,0,4,1</output_order>"""

        user_message = f"""Rerank the following pages for this query:

**Query:** {query}

**Candidate Pages:**
{pages_text}

Analyze each page's relevance to the query, then provide your ranking.
Return at most {top_k} page indices, ordered from most to least relevant.
Only include pages that would actually help answer the query.

<output_order>YOUR_COMMA_SEPARATED_INDICES_HERE</output_order>"""

        try:
            response = self._llm_backend.llm_completion_with_system_prompt(
                model=self.reranker_model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0,
            )
            
            # Parse response - extract indices
            indices = self._parse_rerank_response(response, len(results))
            
            # Reorder results based on LLM ranking
            reranked = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(results):
                    # Update score based on rerank position
                    result = results[idx]
                    # New score: weighted combination of semantic score and rank position
                    rank_score = 1.0 - (len(reranked) * 0.1)  # Decay by position
                    result.score = max(0.1, min(1.0, (result.score + rank_score) / 2))
                    reranked.append(result)
            
            logger.info(f"Reranked {len(results)} results -> {len(reranked)} (model: {self.reranker_model})")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]
    
    def _parse_rerank_response(self, response: str, max_idx: int) -> List[int]:
        """Parse LLM reranking response to extract indices from <output_order> tags."""
        indices = []
        
        # Try to extract from <output_order> tags first
        tag_match = re.search(r'<output_order>(.*?)</output_order>', response, re.DOTALL)
        if tag_match:
            content = tag_match.group(1).strip()
        else:
            # Fallback to entire response
            content = response
        
        # Extract numbers from content
        numbers = re.findall(r'\d+', content)
        
        for num_str in numbers:
            try:
                idx = int(num_str)
                if 0 <= idx < max_idx and idx not in indices:
                    indices.append(idx)
            except ValueError:
                continue
        
        return indices
    
    def _semantic_search(
        self,
        query: str,
        filters: KGSearchFilters,
    ) -> List[KGResultItem]:
        """
        Perform semantic search in Weaviate.
        
        Args:
            query: Search query text
            filters: Search filters
            
        Returns:
            List of KGResultItem ranked by similarity
        """
        if not self._weaviate_client or not self._openai_client:
            logger.warning("Weaviate or OpenAI not available for semantic search")
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return []
        
        try:
            # Get collection
            collection = self._weaviate_client.collections.get(self.weaviate_collection)
            
            # Build Weaviate filters
            weaviate_filters = self._build_weaviate_filters(filters)
            
            # Search with near_vector (target named vector "default" for Weaviate 1.27+)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                target_vector="default",
                limit=filters.top_k,
                filters=weaviate_filters,
                return_metadata=["distance"],
            )
            
            # Convert to KGResultItem
            results = []
            for obj in response.objects:
                props = obj.properties
                
                # Convert distance to score (cosine similarity)
                distance = obj.metadata.distance if obj.metadata else 0
                score = max(0.0, min(1.0, 1.0 - distance))
                
                # Apply min_score filter
                if filters.min_score and score < filters.min_score:
                    continue
                
                results.append(KGResultItem(
                    id=props.get("page_id", ""),
                    score=score,
                    page_title=props.get("page_title", ""),
                    page_type=props.get("page_type", ""),
                    overview=props.get("overview", ""),
                    content=props.get("content", "") if filters.include_content else "",
                    metadata={
                        "domains": props.get("domains", []),
                    },
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _build_weaviate_filters(self, filters: KGSearchFilters):
        """Build Weaviate filter object from KGSearchFilters."""
        filter_conditions = []
        
        # Filter by page_type
        if filters.page_types:
            type_filters = [
                wvc.query.Filter.by_property("page_type").equal(pt)
                for pt in filters.page_types
            ]
            if len(type_filters) == 1:
                filter_conditions.append(type_filters[0])
            else:
                # OR together multiple page types
                combined = type_filters[0]
                for tf in type_filters[1:]:
                    combined = combined | tf
                filter_conditions.append(combined)
        
        # Filter by domains
        if filters.domains:
            domain_filters = [
                wvc.query.Filter.by_property("domains").contains_any([d])
                for d in filters.domains
            ]
            if len(domain_filters) == 1:
                filter_conditions.append(domain_filters[0])
            else:
                combined = domain_filters[0]
                for df in domain_filters[1:]:
                    combined = combined | df
                filter_conditions.append(combined)
        
        # Combine all filters with AND
        if not filter_conditions:
            return None
        
        result = filter_conditions[0]
        for fc in filter_conditions[1:]:
            result = result & fc
        
        return result
    
    def _get_connected_pages(self, page_id: str) -> List[Dict[str, str]]:
        """
        Get connected pages from Neo4j.
        
        Returns both outgoing and incoming connections.
        """
        if not self._neo4j_driver:
            return []
        
        connected = []
        
        try:
            with self._neo4j_driver.session() as session:
                # Get outgoing connections
                outgoing_query = """
                    MATCH (p:WikiPage {id: $page_id})-[r]->(target:WikiPage)
                    RETURN target.id AS id, target.page_title AS title, 
                           target.page_type AS type, type(r) AS edge_type
                """
                result = session.run(outgoing_query, page_id=page_id)
                for record in result:
                    connected.append({
                        "id": record["id"],
                        "title": record["title"],
                        "type": record["type"],
                        "edge_type": record["edge_type"],
                        "direction": "outgoing",
                    })
                
                # Get incoming connections
                incoming_query = """
                    MATCH (source:WikiPage)-[r]->(p:WikiPage {id: $page_id})
                    RETURN source.id AS id, source.page_title AS title,
                           source.page_type AS type, type(r) AS edge_type
                """
                result = session.run(incoming_query, page_id=page_id)
                for record in result:
                    connected.append({
                        "id": record["id"],
                        "title": record["title"],
                        "type": record["type"],
                        "edge_type": record["edge_type"],
                        "direction": "incoming",
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to get connected pages for {page_id}: {e}")
        
        return connected
    
    # =========================================================================
    # Page Retrieval
    # =========================================================================
    
    def get_page(self, page_title: str) -> Optional[WikiPage]:
        """
        Retrieve a wiki page by its title.
        
        Looks up the page in Weaviate by exact title match.
        
        Args:
            page_title: Exact title of the page to retrieve
            
        Returns:
            WikiPage if found, None otherwise
        """
        if not HAS_WEAVIATE or not self._weaviate_client:
            logger.warning("Weaviate not available for page retrieval")
            return None
        
        try:
            collection = self._weaviate_client.collections.get(self.weaviate_collection)

            # NOTE: although this method is named `get_page(page_title)`, the
            # cognitive retrieval stack often has a stable page *id* (e.g.
            # "Principle/huggingface peft LoRA Configuration") from Neo4j.
            #
            # To keep the interface stable while making retrieval robust, we
            # support BOTH lookup modes:
            # - If the input looks like a typed wiki ID ("Workflow/...", "Principle/...", ...),
            #   we attempt an exact `page_id` match first.
            # - Otherwise we fall back to the original `page_title` match.
            response = None
            if "/" in page_title:
                prefix = page_title.split("/", 1)[0]
                if prefix in PageType.values():
                    response = collection.query.fetch_objects(
                        filters=wvc.query.Filter.by_property("page_id").equal(page_title),
                        limit=1,
                        include_vector=False,
                    )
            
            # Fallback / default: Query by exact page_title match
            if response is None or not response.objects:
                response = collection.query.fetch_objects(
                    filters=wvc.query.Filter.by_property("page_title").equal(page_title),
                    limit=1,
                    include_vector=False,
                )
            
            if response.objects:
                obj = response.objects[0]
                props = obj.properties
                return WikiPage(
                    id=props.get("page_id", ""),
                    page_title=props.get("page_title", ""),
                    page_type=props.get("page_type", ""),
                    overview=props.get("overview", ""),
                    content=props.get("content", ""),
                    domains=props.get("domains", []),
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get page '{page_title}': {e}")
            return None
    
    # =========================================================================
    # Edit Methods
    # =========================================================================
    
    def edit(self, data: KGEditInput) -> bool:
        """
        Edit an existing wiki page and update ALL storage layers.
        
        Updates (in order):
        1. Raw source file (.md or .mediawiki)
        2. Persist JSON cache
        3. Weaviate (embeddings + properties)
        4. Neo4j (node properties + edges)
        5. Internal memory cache
        
        Args:
            data: KGEditInput with page_id and fields to update
            
        Returns:
            True if successful, False if page not found
        """
        from datetime import datetime, timezone
        
        updates = data.get_updates()
        
        # Add timestamp if auto_timestamp enabled
        if data.auto_timestamp:
            updates["last_updated"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M GMT"
            )
        
        # Track results for each layer
        results = {
            "source_file": None,
            "persist_cache": None,
            "weaviate": None,
            "neo4j": None,
        }
        
        # =====================================================================
        # 1. Update Raw Source File
        # =====================================================================
        if data.update_source_files and data.wiki_dir:
            try:
                results["source_file"] = self._update_source_file(
                    data.page_id, 
                    updates, 
                    data.wiki_dir,
                )
            except Exception as e:
                logger.error(f"Failed to update source file for {data.page_id}: {e}")
                results["source_file"] = False
        
        # =====================================================================
        # 2. Update Persist JSON Cache
        # =====================================================================
        if data.update_persist_cache and data.persist_path:
            try:
                results["persist_cache"] = self._update_persist_cache(
                    data.page_id,
                    updates,
                    data.persist_path,
                )
            except Exception as e:
                logger.error(f"Failed to update persist cache for {data.page_id}: {e}")
                results["persist_cache"] = False
        
        # =====================================================================
        # 3. Update Weaviate
        # =====================================================================
        if self._weaviate_client:
            try:
                results["weaviate"] = self._update_weaviate_page(
                    data.page_id, 
                    updates,
                    reembed=data.requires_reembedding,
                )
            except Exception as e:
                logger.error(f"Failed to update Weaviate for {data.page_id}: {e}")
                results["weaviate"] = False
        
        # =====================================================================
        # 4. Update Neo4j
        # =====================================================================
        if self._neo4j_driver:
            try:
                results["neo4j"] = self._update_neo4j_page(
                    data.page_id,
                    updates,
                    rebuild_edges=data.requires_edge_rebuild,
                )
            except Exception as e:
                logger.error(f"Failed to update Neo4j for {data.page_id}: {e}")
                results["neo4j"] = False
        
        # =====================================================================
        # 5. Update Internal Cache
        # =====================================================================
        self._update_cached_page(data.page_id, updates)
        
        # Determine overall success (at least one layer updated)
        success = any(v is True for v in results.values())
        
        if success:
            updated_layers = [k for k, v in results.items() if v is True]
            logger.info(f"Edited page {data.page_id}: {list(updates.keys())} -> {updated_layers}")
        else:
            logger.warning(f"Edit failed for {data.page_id}: {results}")
        
        return success
    
    # =========================================================================
    # Add Page Method (for new pages)
    # =========================================================================
    
    def add_page(
        self,
        page: WikiPage,
        wiki_dir: Path,
        persist_path: Optional[Path] = None,
    ) -> bool:
        """
        Add a new page to ALL storage layers.
        
        Creates (in order):
        1. Raw source file (.md) in wiki_dir
        2. Persist JSON cache entry (if persist_path provided)
        3. Weaviate (embeddings + properties)
        4. Neo4j (node + edges)
        5. Internal memory cache
        
        Args:
            page: WikiPage object to add
            wiki_dir: Directory to write the source .md file
            persist_path: Optional path to JSON cache file
            
        Returns:
            True if successful, False on failure
            
        Example:
            page = WikiPage(
                id="Principle/My_New_Concept",
                page_title="My New Concept",
                page_type="Principle",
                overview="A brief overview...",
                content="Full content...",
            )
            success = search.add_page(page, Path("data/wikis"))
        """
        results = {
            "source_file": None,
            "persist_cache": None,
            "weaviate": None,
            "neo4j": None,
        }
        
        # =====================================================================
        # 1. Create Source File
        # =====================================================================
        try:
            results["source_file"] = self._create_source_file(page, wiki_dir)
        except Exception as e:
            logger.error(f"Failed to create source file for {page.id}: {e}")
            results["source_file"] = False
        
        # =====================================================================
        # 2. Update Persist JSON Cache
        # =====================================================================
        if persist_path:
            try:
                results["persist_cache"] = self._add_to_persist_cache(page, persist_path)
            except Exception as e:
                logger.error(f"Failed to update persist cache for {page.id}: {e}")
                results["persist_cache"] = False
        
        # =====================================================================
        # 3. Index to Weaviate
        # =====================================================================
        if self._weaviate_client:
            try:
                results["weaviate"] = self._index_single_page_to_weaviate(page)
            except Exception as e:
                logger.error(f"Failed to index {page.id} to Weaviate: {e}")
                results["weaviate"] = False
        
        # =====================================================================
        # 4. Index to Neo4j
        # =====================================================================
        if self._neo4j_driver:
            try:
                results["neo4j"] = self._index_single_page_to_neo4j(page)
            except Exception as e:
                logger.error(f"Failed to index {page.id} to Neo4j: {e}")
                results["neo4j"] = False
        
        # =====================================================================
        # 5. Update Internal Cache
        # =====================================================================
        if self._pages is None:
            self._pages = []
        self._pages.append(page)
        
        # Determine overall success
        success = any(v is True for v in results.values())
        
        if success:
            created_layers = [k for k, v in results.items() if v is True]
            logger.info(f"Added page {page.id} to: {created_layers}")
        else:
            logger.warning(f"Add page failed for {page.id}: {results}")
        
        return success
    
    def _create_source_file(self, page: WikiPage, wiki_dir: Path) -> bool:
        """
        Create a new source .md file for a WikiPage.
        
        Creates the file in the appropriate type subdirectory:
        wiki_dir/{type_subdir}/{Title}.md
        
        The page.content is written directly as it should already be in the
        correct MediaWiki format (following sections_definition.md structure).
        
        Args:
            page: WikiPage to write
            wiki_dir: Root wiki directory
            
        Returns:
            True if file was created successfully
        """
        # Get subdirectory for this page type
        subdir = _TYPE_TO_SUBDIR.get(page.page_type)
        if not subdir:
            logger.warning(f"Unknown page type: {page.page_type}")
            # Fall back to using page_type as subdir name
            subdir = page.page_type.lower() + "s"
        
        # Create directory if needed
        target_dir = wiki_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename from page title (use spaces, not underscores)
        filename = page.page_title.replace("_", " ") + ".md"
        file_path = target_dir / filename
        
        # Write content directly - it should already be in proper MediaWiki format
        # (following the sections_definition.md structure for this page type)
        file_path.write_text(page.content, encoding="utf-8")
        logger.info(f"Created source file: {file_path}")
        
        return True
    
    def _format_page_as_markdown(self, page: WikiPage) -> str:
        """
        Format a WikiPage as markdown content for source file.
        
        DEPRECATED: This method is no longer used. Pages should be written
        with their content directly, as it should already be in the correct
        MediaWiki format (following sections_definition.md structure).
        
        This method is kept for potential edge cases where structured data
        needs to be converted to a file format, but the preferred approach
        is to have ingestors produce properly formatted content.
        
        Creates a structured markdown file with metadata table,
        overview section, and main content.
        """
        parts = []
        
        # Title
        parts.append(f"# {page.page_title}")
        parts.append("")
        
        # Metadata table
        parts.append("| Property | Value |")
        parts.append("|----------|-------|")
        parts.append(f"| **Type** | {page.page_type} |")
        
        if page.domains:
            domain_links = ", ".join(f"[[domain::{d}]]" for d in page.domains)
            parts.append(f"| **Domains** | {domain_links} |")
        
        if page.last_updated:
            parts.append(f"| **Last Updated** | [[last_updated::{page.last_updated}]] |")
        
        parts.append("")
        
        # Sources section (if any)
        if page.sources:
            parts.append("## Knowledge Sources")
            for src in page.sources:
                src_type = src.get("type", "Doc")
                src_title = src.get("title", "")
                src_url = src.get("url", "")
                parts.append(f"* [[source::{src_type}|{src_title}|{src_url}]]")
            parts.append("")
        
        # Overview section
        parts.append("## Overview")
        parts.append(page.overview)
        parts.append("")
        
        # Main content
        parts.append("## Content")
        parts.append(page.content)
        parts.append("")
        
        # Outgoing links section (if any)
        if page.outgoing_links:
            parts.append("## Related Pages")
            for link in page.outgoing_links:
                edge_type = link.get("edge_type", "related")
                target_type = link.get("target_type", "")
                target_id = link.get("target_id", "")
                parts.append(f"* [[{edge_type}::{target_type}:{target_id}]]")
            parts.append("")
        
        return "\n".join(parts)
    
    def _add_to_persist_cache(self, page: WikiPage, persist_path: Path) -> bool:
        """
        Add a page to the persist JSON cache.
        
        Loads existing cache, appends the new page, and saves back.
        
        Args:
            page: WikiPage to add
            persist_path: Path to JSON cache file
            
        Returns:
            True if successful
        """
        # Load existing cache or create new
        if persist_path.exists():
            with open(persist_path, 'r') as f:
                data = json.load(f)
        else:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pages": [],
                "metadata": {
                    "collection": self.weaviate_collection,
                    "embedding_model": self.embedding_model,
                },
            }
        
        # Check if page already exists (by id)
        existing_ids = {p.get("id") for p in data.get("pages", [])}
        if page.id in existing_ids:
            # Update existing entry
            data["pages"] = [
                page.to_dict() if p.get("id") == page.id else p
                for p in data["pages"]
            ]
            logger.debug(f"Updated existing page in cache: {page.id}")
        else:
            # Append new page
            data["pages"].append(page.to_dict())
            logger.debug(f"Added new page to cache: {page.id}")
        
        # Save back
        with open(persist_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    def _index_single_page_to_weaviate(self, page: WikiPage) -> bool:
        """
        Index a single page to Weaviate.
        
        Generates embedding and inserts to collection.
        
        Args:
            page: WikiPage to index
            
        Returns:
            True if successful
        """
        if not self._weaviate_client or not self._openai_client:
            return False
        
        # Ensure collection exists
        self._ensure_weaviate_collection()
        
        # Generate embedding from overview
        embedding = self._generate_embedding(page.overview)
        if not embedding:
            logger.warning(f"Failed to generate embedding for {page.id}")
            return False
        
        # Prepare properties
        properties = {
            "page_id": page.id,
            "page_title": page.page_title,
            "page_type": page.page_type,
            "overview": page.overview,
            "content": page.content,
            "domains": page.domains,
        }
        
        # Get collection
        collection = self._weaviate_client.collections.get(self.weaviate_collection)
        
        # Insert with named vector
        collection.data.insert(
            properties=properties,
            vector={"default": embedding},
        )
        
        logger.debug(f"Indexed page to Weaviate: {page.id}")
        return True
    
    def _index_single_page_to_neo4j(self, page: WikiPage) -> bool:
        """
        Index a single page to Neo4j.
        
        Creates/merges node and creates edges.
        
        Args:
            page: WikiPage to index
            
        Returns:
            True if successful
        """
        if not self._neo4j_driver:
            return False
        
        with self._neo4j_driver.session() as session:
            # Create/merge node
            self._create_neo4j_node(session, page)
            
            # Create edges
            self._create_neo4j_edges(session, page)
        
        logger.debug(f"Indexed page to Neo4j: {page.id}")
        return True
    
    def page_exists(self, page_id: str) -> bool:
        """
        Check if a page exists in the knowledge graph.
        
        Checks Weaviate first (faster), falls back to Neo4j.
        
        Args:
            page_id: Page ID to check (e.g., "Principle/My_Concept")
            
        Returns:
            True if page exists
        """
        # Check Weaviate first
        if self._weaviate_client:
            try:
                collection = self._weaviate_client.collections.get(self.weaviate_collection)
                response = collection.query.fetch_objects(
                    filters=wvc.query.Filter.by_property("page_id").equal(page_id),
                    limit=1,
                )
                if response.objects:
                    return True
            except Exception:
                pass
        
        # Fall back to Neo4j
        if self._neo4j_driver:
            try:
                with self._neo4j_driver.session() as session:
                    result = session.run(
                        "MATCH (p:WikiPage {id: $id}) RETURN p LIMIT 1",
                        id=page_id,
                    )
                    if result.single():
                        return True
            except Exception:
                pass
        
        return False
    
    # =========================================================================
    # Source File Update Methods
    # =========================================================================
    
    def _update_source_file(
        self,
        page_id: str,
        updates: Dict[str, Any],
        wiki_dir: Path,
    ) -> bool:
        """
        Update the raw source file (.md or .mediawiki).
        
        Handles two structures:
        1. Type subdirectories: wiki_dir/{type_subdir}/{filename}.md
        2. Flat structure: wiki_dir/{Type_Filename}.mediawiki
        """
        # Determine file path from page_id
        file_path = self._resolve_source_file_path(page_id, wiki_dir)
        
        if not file_path or not file_path.exists():
            logger.warning(f"Source file not found for {page_id}")
            return False
        
        # If full content is provided, just replace the file
        if "content" in updates:
            new_content = self._inject_metadata_into_content(
                updates["content"],
                updates,
            )
            file_path.write_text(new_content, encoding="utf-8")
            logger.info(f"Replaced source file: {file_path}")
            return True
        
        # Otherwise, patch specific parts of the existing file
        current_content = file_path.read_text(encoding="utf-8")
        new_content = self._patch_file_content(current_content, updates)
        file_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Patched source file: {file_path}")
        return True
    
    def _resolve_source_file_path(
        self,
        page_id: str,
        wiki_dir: Path,
    ) -> Optional[Path]:
        """
        Resolve the source file path from page_id.
        
        Page ID format: "{PageType}/{filename}" or "{repo_id}/{name}"
        
        Note: Files may be created with spaces in filename (e.g., "RAG Chain Pattern.md")
        but page_id uses underscores (e.g., "Principle/RAG_Chain_Pattern").
        This method tries both formats.
        
        Returns:
            Path to source file, or None if not found
        """
        # Check if wiki_dir uses type subdirectories
        has_type_subdirs = any((wiki_dir / subdir).exists() for subdir in _TYPE_SUBDIRS)
        
        if has_type_subdirs:
            # Type subdirectory structure: page_id = "Workflow/QLoRA_Finetuning"
            parts = page_id.split("/", 1)
            if len(parts) != 2:
                return None
            
            page_type, filename = parts
            subdir = _TYPE_TO_SUBDIR.get(page_type)
            if not subdir:
                return None
            
            # Try with underscores first (page_id format)
            file_path = wiki_dir / subdir / f"{filename}.md"
            if file_path.exists():
                return file_path
            
            # Try with spaces (how _create_source_file creates them)
            filename_with_spaces = filename.replace("_", " ")
            file_path = wiki_dir / subdir / f"{filename_with_spaces}.md"
            if file_path.exists():
                return file_path
            
            return None
        
        else:
            # Flat structure: page_id = "repo_id/Name" -> "Type_Name.mediawiki"
            # Need to find by matching parsed id
            parts = page_id.split("/", 1)
            if len(parts) != 2:
                return None
            
            repo_id, name = parts
            
            # Search for file that would produce this id
            for wiki_file in wiki_dir.glob("*.mediawiki"):
                file_parts = wiki_file.stem.split("_", 1)
                if len(file_parts) > 1 and file_parts[1] == name:
                    return wiki_file
            
            return None
    
    def _patch_file_content(
        self,
        current_content: str,
        updates: Dict[str, Any],
    ) -> str:
        """
        Patch specific parts of wiki file content.
        
        Handles metadata table updates (domains, sources, last_updated)
        and overview section updates.
        """
        new_content = current_content
        
        # Update domains: [[domain::X]], [[domain::Y]] format
        if "domains" in updates:
            new_domains = updates["domains"]
            domain_line = ", ".join(f"[[domain::{d.replace(' ', '_')}]]" for d in new_domains)
            
            # Pattern to find the domains row in wikitable
            pattern = r'(\|\| \[\[domain::.*?\]\](?:, \[\[domain::.*?\]\])*)'
            replacement = f'|| {domain_line}'
            new_content = re.sub(pattern, replacement, new_content)
        
        # Update last_updated: [[last_updated::YYYY-MM-DD HH:MM GMT]]
        if "last_updated" in updates:
            new_timestamp = updates["last_updated"]
            pattern = r'\[\[last_updated::[^\]]+\]\]'
            replacement = f'[[last_updated::{new_timestamp}]]'
            new_content = re.sub(pattern, replacement, new_content)
        
        # Update sources: [[source::Type|Title|URL]]
        if "sources" in updates:
            new_sources = updates["sources"]
            source_lines = []
            for src in new_sources:
                src_type = src.get("type", "Doc")
                src_title = src.get("title", "")
                src_url = src.get("url", "")
                source_lines.append(f"* [[source::{src_type}|{src_title}|{src_url}]]")
            
            # Replace entire Knowledge Sources section content
            sources_text = "\n".join(source_lines)
            pattern = r'(\! Knowledge Sources\n\|\|\n)(\* \[\[source::.*?\]\]\n?)+'
            replacement = f'\\1{sources_text}\n'
            new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
        
        # Update overview section
        if "overview" in updates:
            new_overview = updates["overview"]
            # Replace content between "== Overview ==" and next "==" or "==="
            pattern = r'(== Overview ==\n)(.+?)(\n===|\n==|\n\{\{|\Z)'
            replacement = f'\\1{new_overview}\n\\3'
            new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
        
        return new_content
    
    def _inject_metadata_into_content(
        self,
        content: str,
        updates: Dict[str, Any],
    ) -> str:
        """
        Ensure metadata tags in content match updates.
        
        When full content is replaced, this ensures the metadata tags
        (domains, last_updated, etc.) reflect the update values.
        """
        new_content = content
        
        # If updates include these fields, patch them into the new content
        if "domains" in updates:
            new_content = self._patch_file_content(new_content, {"domains": updates["domains"]})
        
        if "last_updated" in updates:
            new_content = self._patch_file_content(new_content, {"last_updated": updates["last_updated"]})
        
        if "sources" in updates:
            new_content = self._patch_file_content(new_content, {"sources": updates["sources"]})
        
        return new_content
    
    # =========================================================================
    # Persist Cache Update Methods
    # =========================================================================
    
    def _update_persist_cache(
        self,
        page_id: str,
        updates: Dict[str, Any],
        persist_path: Path,
    ) -> bool:
        """
        Update the persisted JSON cache file.
        """
        if not persist_path.exists():
            logger.warning(f"Persist cache not found: {persist_path}")
            return False
        
        # Load existing cache
        with open(persist_path, 'r') as f:
            data = json.load(f)
        
        # Find and update the page
        pages = data.get("pages", [])
        found = False
        for page_dict in pages:
            if page_dict.get("id") == page_id:
                for field, value in updates.items():
                    if field in page_dict:
                        page_dict[field] = value
                found = True
                break
        
        if not found:
            logger.warning(f"Page {page_id} not found in persist cache")
            return False
        
        # Write back
        with open(persist_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Updated persist cache: {persist_path}")
        return True
    
    # =========================================================================
    # Weaviate Update Methods
    # =========================================================================
    
    def _update_weaviate_page(
        self, 
        page_id: str, 
        updates: Dict[str, Any],
        reembed: bool = False,
    ) -> bool:
        """Update a page in Weaviate."""
        if not HAS_WEAVIATE or not self._weaviate_client:
            return False
        
        collection = self._weaviate_client.collections.get(self.weaviate_collection)
        
        # Find the object by page_id
        response = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("page_id").equal(page_id),
            limit=1,
        )
        
        if not response.objects:
            return False
        
        obj = response.objects[0]
        obj_uuid = obj.uuid
        
        # Prepare property updates (map our field names to Weaviate properties)
        weaviate_updates = {}
        field_mapping = {
            "page_title": "page_title",
            "page_type": "page_type", 
            "overview": "overview",
            "content": "content",
            "domains": "domains",
        }
        
        for our_field, weaviate_field in field_mapping.items():
            if our_field in updates:
                weaviate_updates[weaviate_field] = updates[our_field]
        
        # Generate new embedding if overview changed
        new_vector = None
        if reembed and "overview" in updates:
            new_vector = self._generate_embedding(updates["overview"])
        
        # Update the object
        if new_vector:
            collection.data.update(
                uuid=obj_uuid,
                properties=weaviate_updates,
                vector=new_vector,
            )
        elif weaviate_updates:
            collection.data.update(
                uuid=obj_uuid,
                properties=weaviate_updates,
            )
        
        return True
    
    # =========================================================================
    # Neo4j Update Methods
    # =========================================================================
    
    def _update_neo4j_page(
        self,
        page_id: str,
        updates: Dict[str, Any],
        rebuild_edges: bool = False,
    ) -> bool:
        """Update a page in Neo4j."""
        if not HAS_NEO4J or not self._neo4j_driver:
            return False
        
        with self._neo4j_driver.session() as session:
            # Check if page exists
            check_result = session.run(
                "MATCH (p:WikiPage {id: $id}) RETURN p",
                id=page_id,
            )
            if not check_result.single():
                return False
            
            # Update node properties
            neo4j_updates = {}
            for field in ["page_title", "page_type", "domains"]:
                if field in updates:
                    neo4j_updates[field] = updates[field]
            
            if neo4j_updates:
                session.run(
                    "MATCH (p:WikiPage {id: $id}) SET p += $updates",
                    id=page_id,
                    updates=neo4j_updates,
                )
            
            # Rebuild edges if outgoing_links changed
            if rebuild_edges and "outgoing_links" in updates:
                # Delete existing outgoing edges
                session.run(
                    "MATCH (p:WikiPage {id: $id})-[r]->() DELETE r",
                    id=page_id,
                )
                
                # Create new edges (with normalized IDs)
                for link in updates["outgoing_links"]:
                    edge_type = link.get("edge_type", "RELATED").upper()
                    target_type = link.get("target_type", "")
                    target_id = link.get("target_id", "")
                    
                    # Normalize ID: convert underscores to spaces
                    raw_target_id = f"{target_type}/{target_id}"
                    target_page_id = normalize_page_id(raw_target_id)
                    normalized_title = target_id.replace("_", " ")
                    
                    neo4j_rel_type = self._map_edge_type(edge_type)
                    
                    query = f"""
                        MERGE (target:WikiPage {{id: $target_id}})
                        ON CREATE SET target.page_type = $target_type, target.page_title = $target_title
                        WITH target
                        MATCH (source:WikiPage {{id: $source_id}})
                        MERGE (source)-[r:{neo4j_rel_type}]->(target)
                    """
                    session.run(
                        query,
                        source_id=page_id,
                        target_id=target_page_id,
                        target_type=target_type,
                        target_title=normalized_title,
                    )
            
            return True
    
    # =========================================================================
    # Internal Cache Update Methods
    # =========================================================================
    
    def _update_cached_page(self, page_id: str, updates: Dict[str, Any]) -> None:
        """Update page in internal memory cache."""
        for page in self._pages:
            if page.id == page_id:
                for field, value in updates.items():
                    if hasattr(page, field):
                        setattr(page, field, value)
                break
    
    # =========================================================================
    # Index Metadata Methods (for .index files)
    # =========================================================================
    
    def get_backend_refs(self) -> Dict[str, Any]:
        """
        Return backend-specific references for index file.
        
        Returns Weaviate collection name and embedding model used.
        """
        return {
            "weaviate_collection": self.weaviate_collection,
            "embedding_model": self.embedding_model,
        }
    
    def validate_backend_data(self) -> bool:
        """
        Check if Weaviate collection has indexed data.
        
        Returns:
            True if collection exists and has data, False otherwise
        """
        if not self._weaviate_client:
            return False
        try:
            collection = self._weaviate_client.collections.get(self.weaviate_collection)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count > 0
        except Exception:
            return False
    
    def get_indexed_count(self) -> int:
        """
        Get the number of indexed pages.
        
        Returns:
            Count from internal cache, or queries Weaviate if cache is empty
        """
        # First check internal cache
        if self._pages:
            return len(self._pages)
        
        # Otherwise query Weaviate
        if not self._weaviate_client:
            return 0
        try:
            collection = self._weaviate_client.collections.get(self.weaviate_collection)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count
        except Exception:
            return 0
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def clear(self) -> None:
        """Clear all indexed data from Weaviate and Neo4j."""
        # Clear Weaviate collection
        if self._weaviate_client:
            try:
                # Use default collection name if not set
                collection_name = self.weaviate_collection or "KGWikiPages"
                self._weaviate_client.collections.delete(collection_name)
                logger.info(f"Deleted Weaviate collection '{collection_name}'")
            except Exception as e:
                logger.warning(f"Could not delete Weaviate collection: {e}")
        
        # Clear Neo4j data
        if self._neo4j_driver:
            try:
                with self._neo4j_driver.session() as session:
                    session.run("MATCH (n:WikiPage) DETACH DELETE n")
                logger.info("Cleared Neo4j WikiPage nodes")
            except Exception as e:
                logger.warning(f"Could not clear Neo4j: {e}")
        
        self._pages = []
    
    def close(self) -> None:
        """Close all connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None
        
        if self._weaviate_client:
            self._weaviate_client.close()
            self._weaviate_client = None
        
        # Close the OpenAI client if it exposes a close() method.
        #
        # Why:
        # - The OpenAI SDK uses an underlying HTTP client (often httpx) which
        #   can keep sockets open if not explicitly closed.
        # - Our tests intentionally create/close KnowledgeSearch instances, and
        #   unclosed sockets show up as ResourceWarning noise in CI/log audits.
        # - We treat this as part of the KnowledgeSearch resource lifecycle.
        if self._openai_client:
            try:
                if hasattr(self._openai_client, "close"):
                    self._openai_client.close()
            except Exception:
                # Best-effort cleanup. Never fail callers during shutdown.
                pass
            self._openai_client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-close connections."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - auto-close connections when garbage collected."""
        self.close()


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the KG Graph Search with data/wikis."""
    print("=" * 60)
    print("KG Graph Search Test")
    print("=" * 60)
    
    # Initialize search
    search = KGGraphSearch(params={})
    
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-index":
        print("\nSkipping indexing (--skip-index flag)")
    else:
        # Index data/wikis
        wiki_dir = Path("data/wikis")
        persist_path = Path("data/indexes/wikis.json")
        
        if wiki_dir.exists():
            print(f"\nIndexing wiki pages from {wiki_dir}...")
            search.index(KGIndexInput(
                wiki_dir=wiki_dir,
                persist_path=persist_path,
            ))
        else:
            print(f"\nWarning: {wiki_dir} not found")
            sys.exit(1)
    
    # Test queries
    test_cases = [
        # (query, expected_type, page_types_filter)
        ("How to fine-tune LLM with limited GPU memory?", "Workflow", ["Workflow"]),
        ("preference learning for AI alignment", "Principle", None),
        ("LoRA rank selection best practices", "Heuristic", ["Heuristic"]),
        ("training configuration hyperparameters", None, None),
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Queries")
    print("=" * 60)
    
    for query, expected_type, page_types in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        if page_types:
            print(f"Filter: page_types={page_types}")
        
        filters = KGSearchFilters(
            top_k=3,
            page_types=page_types,
        )
        
        result = search.search(query, filters)
        
        print(f"\nResults ({result.total_found} found):")
        for item in result:
            print(f"  - {item.page_title} ({item.page_type})")
            print(f"    Score: {item.score:.3f}")
            if item.metadata.get("connected_pages"):
                conn_count = len(item.metadata["connected_pages"])
                print(f"    Connected: {conn_count} pages")
        
        if expected_type and result.top_result:
            if result.top_result.page_type == expected_type:
                print(f"  ✓ Expected type '{expected_type}' found")
            else:
                print(f"  ✗ Expected '{expected_type}', got '{result.top_result.page_type}'")
    
    # Cleanup
    print("\n" + "=" * 60)
    search.close()
    print("Test complete!")

