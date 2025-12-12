# Wiki Parser
#
# Parses .mediawiki files into WikiPage objects for indexing.
# Handles the wiki structure defined in src/knowledge/wiki_structure/.

import re
from pathlib import Path
from typing import List, Optional, Dict

from src.knowledge.search.base import WikiPage, PageType


def parse_wiki_directory(
    wiki_dir: Path,
    domain_file: str = "domain_tag.txt",
) -> List[WikiPage]:
    """
    Parse all .mediawiki files in a directory into WikiPage objects.
    
    Args:
        wiki_dir: Path to directory containing .mediawiki files
        domain_file: Name of file containing default domain tags
        
    Returns:
        List of parsed WikiPage objects
    
    Example:
        pages = parse_wiki_directory(Path("data/wikis/allenai_allennlp"))
    """
    wiki_dir = Path(wiki_dir)
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
