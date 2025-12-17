# Paper Ingestor
#
# Extracts knowledge from research papers (PDFs).
# Parses sections, abstracts, formulas, and key findings.
#
# Part of Stage 1 of the knowledge learning pipeline.

import logging
from typing import Any, Dict, List, Optional

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_ingestor("paper")
class PaperIngestor(Ingestor):
    """
    Extract knowledge from research papers and PDFs.
    
    Extracts knowledge from:
    - Abstract and introduction
    - Methodology sections
    - Key findings and conclusions
    - Formulas and algorithms
    
    Input formats:
        Source.Paper("./paper.pdf")
        {"path": "./paper.pdf"}
        {"url": "https://arxiv.org/pdf/..."}
    
    Example:
        ingestor = PaperIngestor()
        pages = ingestor.ingest(Source.Paper("./research.pdf"))
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize PaperIngestor.
        
        Args:
            params: Optional parameters (for future use)
        """
        super().__init__(params)
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "paper"
    
    def _normalize_source(self, source: Any) -> Dict[str, Any]:
        """
        Normalize source input to a dict.
        
        Args:
            source: Source.Paper object or dict
            
        Returns:
            Dict with path or url key
        """
        if hasattr(source, 'to_dict'):
            return source.to_dict()
        elif isinstance(source, dict):
            return source
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Extract knowledge from a research paper.
        
        Args:
            source: Source.Paper object or dict with:
                - path: Local file path (optional)
                - url: Remote PDF URL (optional)
            
        Returns:
            List of proposed WikiPage objects
        """
        # Normalize source to dict
        source_data = self._normalize_source(source)
        
        path = source_data.get("path", source_data.get("url", ""))
        
        if not path:
            raise ValueError("Paper path or URL is required")
        
        # TODO: Implement actual PDF parsing with Claude Code agent
        # 1. Load PDF (local or download from URL)
        # 2. Extract text using PyPDF2 or pdfplumber
        # 3. Run Claude Code agent to analyze and extract structured knowledge
        # 4. Return WikiPages for Principles, Implementations, Heuristics
        
        logger.warning(f"[PaperIngestor] Not yet implemented. Source: {path}")
        
        # Placeholder: Return empty list
        # In a full implementation, this would use an agent similar to RepoIngestor
        return []


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the PaperIngestor."""
    import sys
    
    print("=" * 60)
    print("PaperIngestor Test")
    print("=" * 60)
    
    test_path = sys.argv[1] if len(sys.argv) > 1 else "./test_paper.pdf"
    
    print(f"\nTest paper: {test_path}")
    print("-" * 60)
    
    ingestor = PaperIngestor()
    
    try:
        pages = ingestor.ingest({"path": test_path})
        print(f"\nExtracted {len(pages)} proposed pages")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

