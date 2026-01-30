"""
Shared backend singletons for the Gated MCP Server.

Provides lazy, thread-safe initialization of:
- KGGraphSearch: Shared by kg, idea, and code gates
- Researcher: Used by research gate

Backends are initialized on first access, not at server startup.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Singletons
# =============================================================================

_kg_search_backend = None
_kg_search_lock = threading.Lock()

_researcher_backend = None
_researcher_lock = threading.Lock()

# Index metadata (populated when KG backend is initialized from .index file)
_index_metadata = None
_index_data_source: Optional[str] = None


def get_kg_search_backend():
    """
    Get or create the KGGraphSearch backend singleton.
    
    Thread-safe lazy initialization. The backend is configured from:
    1. KG_INDEX_PATH env var (if set, reads .index file for config)
    2. Default kg_graph_search backend
    
    Returns:
        KGGraphSearch instance
        
    Raises:
        RuntimeError: If backend initialization fails
    """
    global _kg_search_backend, _index_metadata, _index_data_source
    
    if _kg_search_backend is not None:
        return _kg_search_backend
    
    with _kg_search_lock:
        # Double-check after acquiring lock
        if _kg_search_backend is not None:
            return _kg_search_backend
        
        try:
            from kapso.knowledge_base.search.factory import KnowledgeSearchFactory
            from kapso.knowledge_base.search.base import KGIndexMetadata
            
            # Check for index file configuration
            index_path_str = os.getenv("KG_INDEX_PATH")
            backend_type = "kg_graph_search"
            backend_refs = {}
            
            if index_path_str:
                try:
                    index_path = Path(index_path_str).expanduser().resolve()
                    if index_path.exists():
                        index_data = json.loads(index_path.read_text(encoding="utf-8"))
                        _index_metadata = KGIndexMetadata.from_dict(index_data)
                        _index_data_source = _index_metadata.data_source
                        
                        # Backend type from index file
                        backend_type = (_index_metadata.search_backend or "").strip() or "kg_graph_search"
                        backend_refs = _index_metadata.backend_refs or {}
                        
                        logger.info(
                            f"Initializing KGGraphSearch from KG_INDEX_PATH={index_path}: "
                            f"backend={backend_type}, data_source={_index_data_source}"
                        )
                    else:
                        logger.warning(f"KG_INDEX_PATH not found: {index_path}")
                except Exception as e:
                    logger.warning(f"Failed to read KG_INDEX_PATH: {e}")
            
            # Create the backend
            logger.info(f"Creating KGGraphSearch backend: {backend_type}")
            _kg_search_backend = KnowledgeSearchFactory.create(backend_type, params=backend_refs)
            logger.info("KGGraphSearch backend initialized successfully")
            
            return _kg_search_backend
            
        except Exception as e:
            logger.error(f"KGGraphSearch initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"KGGraphSearch initialization failed: {e}") from e


def get_researcher_backend():
    """
    Get or create the Researcher backend singleton.
    
    Thread-safe lazy initialization.
    
    Returns:
        Researcher instance
        
    Raises:
        RuntimeError: If backend initialization fails
    """
    global _researcher_backend
    
    if _researcher_backend is not None:
        return _researcher_backend
    
    with _researcher_lock:
        # Double-check after acquiring lock
        if _researcher_backend is not None:
            return _researcher_backend
        
        try:
            from kapso.researcher.researcher import Researcher
            
            logger.info("Creating Researcher backend")
            _researcher_backend = Researcher()
            logger.info("Researcher backend initialized successfully")
            
            return _researcher_backend
            
        except Exception as e:
            logger.error(f"Researcher initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Researcher initialization failed: {e}") from e


def get_index_data_source() -> Optional[str]:
    """
    Get the data source path from the loaded index metadata.
    
    Returns:
        Data source path if available, None otherwise
    """
    return _index_data_source


def reset_backends() -> None:
    """
    Reset all backend singletons.
    
    Useful for testing. Closes any open connections.
    """
    global _kg_search_backend, _researcher_backend, _index_metadata, _index_data_source
    
    with _kg_search_lock:
        if _kg_search_backend is not None:
            try:
                _kg_search_backend.close()
            except Exception as e:
                logger.warning(f"Error closing KGGraphSearch: {e}")
            _kg_search_backend = None
        _index_metadata = None
        _index_data_source = None
    
    with _researcher_lock:
        _researcher_backend = None
    
    logger.info("All backends reset")
