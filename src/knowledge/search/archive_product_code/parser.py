#!/usr/bin/env python3
"""
Wiki Parser - Extracts graph structure (nodes and edges) from MediaWiki wikitext.

Aligned with DE Wiki Guideline:
- 4 Node Types: Concept, Implementation, Workflow, Resource
- Semantic properties (SMW): [[property::Target]] format
- Edge relationships: implements, extends, precedes, related_to, consumes, produces
- Page type identification by title prefix (Concept:, Implementation:, Workflow:, Resource:)

Usage:
    from wiki_parser import WikiParser
    
    parser = WikiParser()
    node, edges = parser.parse(page_title, wikitext)
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class NodeType(Enum):
    """Wiki page types - aligned with DE Wiki guideline"""
    CONCEPT = "concept"           # Theories, principles, business context
    IMPLEMENTATION = "implementation"  # Concrete code, functions, jobs
    WORKFLOW = "workflow"         # ETL pipelines, end-to-end processes
    RESOURCE = "resource"         # Data sources, datasets, event topics
    UNKNOWN = "unknown"           # Fallback for unclassified pages


class EdgeRelationship(Enum):
    """
    Link relationships between wiki pages - aligned with DE Wiki guideline SMW properties.
    
    Structural relationships:
    - implements: source provides practical realization of target (Concept → Implementation)
    - extends: source specializes target (Concept → Concept, Implementation → Implementation)
    
    Sequential relationships:
    - precedes: source comes before target (Workflow → Workflow/Implementation/Concept)
    
    Conceptual relationships:
    - related_to: symmetric thematic association (same type ↔ same type)
    
    Data flow relationships:
    - consumes: source reads/uses target resource (Workflow/Implementation → Resource)
    - produces: source writes/emits target resource (Workflow/Implementation → Resource)
    """
    # Structural
    IMPLEMENTS = "implements"           # Concept → Implementation
    IMPLEMENTED_BY = "implemented_by"   # Implementation → Concept (inverse)
    EXTENDS = "extends"                 # Specialization (same type)
    EXTENDED_BY = "extended_by"         # Inverse of extends
    
    # Sequential
    PRECEDES = "precedes"               # Sequential ordering
    FOLLOWED_BY = "followed_by"         # Inverse of precedes
    
    # Conceptual
    RELATED_TO = "related_to"           # Symmetric thematic link
    
    # Data Flow
    CONSUMES = "consumes"               # Source reads target Resource
    CONSUMED_BY = "consumed_by"         # Resource read by source (inverse)
    PRODUCES = "produces"               # Source writes target Resource
    PRODUCED_BY = "produced_by"         # Resource written by source (inverse)
    
    # Fallback
    UNKNOWN = "links_to"                # Generic link


@dataclass
class WikiNode:
    """Represents a wiki page as a graph node"""
    id: str  # Page title as ID
    type: NodeType
    name: str
    sections: Dict[str, str]  # Section name -> content
    categories: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'sections': self.sections,
            'categories': self.categories,
            'metadata': self.metadata
        }


@dataclass
class WikiEdge:
    """Represents a link between wiki pages as a graph edge"""
    source_id: str
    target_id: str
    relationship: EdgeRelationship
    context: str
    section: str
    link_text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship': self.relationship.value,
            'context': self.context,
            'section': self.section,
            'link_text': self.link_text
        }


class WikiParser:
    """Parser for extracting graph structure from wikitext - aligned with DE Wiki guideline"""
    
    # Patterns for identifying page types from title (primary method per guideline)
    TYPE_PATTERNS = {
        NodeType.CONCEPT: [r'^Concept:'],
        NodeType.IMPLEMENTATION: [r'^Implementation:'],
        NodeType.WORKFLOW: [r'^Workflow:'],
        NodeType.RESOURCE: [r'^Resource:']
    }
    
    # Map SMW property names to EdgeRelationship enum (following guideline)
    SMW_PROPERTY_MAP = {
        # Structural
        'implements': EdgeRelationship.IMPLEMENTS,
        'implemented_by': EdgeRelationship.IMPLEMENTED_BY,
        'extends': EdgeRelationship.EXTENDS,
        'extended_by': EdgeRelationship.EXTENDED_BY,
        
        # Sequential
        'precedes': EdgeRelationship.PRECEDES,
        'followed_by': EdgeRelationship.FOLLOWED_BY,
        
        # Conceptual
        'related_to': EdgeRelationship.RELATED_TO,
        
        # Data Flow
        'consumes': EdgeRelationship.CONSUMES,
        'consumed_by': EdgeRelationship.CONSUMED_BY,
        'produces': EdgeRelationship.PRODUCES,
        'produced_by': EdgeRelationship.PRODUCED_BY
    }
    
    # Fallback patterns for context-based relationship detection (for regular links)
    RELATIONSHIP_PATTERNS = {
        EdgeRelationship.IMPLEMENTS: [
            r'implements?', r'realizes?', r'implementation\s+of'
        ],
        EdgeRelationship.EXTENDS: [
            r'extends?', r'builds?\s+upon', r'enhances?', r'specializes?'
        ],
        EdgeRelationship.PRECEDES: [
            r'next\s+step', r'continues?\s+with', r'precedes?', r'then', r'followed\s+by'
        ],
        EdgeRelationship.RELATED_TO: [
            r'see\s+also', r'related', r'similar'
        ],
        EdgeRelationship.CONSUMES: [
            r'consumes?', r'reads?', r'uses?\s+(data|resource)', r'inputs?'
        ],
        EdgeRelationship.PRODUCES: [
            r'produces?', r'writes?', r'generates?', r'outputs?', r'emits?'
        ]
    }
    
    def parse(self, page_title: str, wikitext: str) -> Tuple[WikiNode, List[WikiEdge]]:
        """
        Parse a wiki page to extract node and edges.
        
        Args:
            page_title: Title of the wiki page
            wikitext: Raw wikitext content
            
        Returns:
            Tuple of (WikiNode, List[WikiEdge])
        """
        # Extract node information
        node = self._extract_node(page_title, wikitext)
        
        # Extract edges (links)
        edges = self._extract_edges(page_title, wikitext)
        
        return node, edges
    
    def _extract_node(self, page_title: str, wikitext: str) -> WikiNode:
        """Extract node information from wikitext"""
        # Determine page type
        page_type = self._identify_page_type(page_title, wikitext)
        
        # Extract sections
        sections = self._extract_sections(wikitext)
        
        # Extract categories
        categories = self._extract_categories(wikitext)
        
        # Extract metadata
        metadata = self._extract_metadata(wikitext, sections)
        
        return WikiNode(
            id=page_title,
            type=page_type,
            name=self._clean_page_name(page_title),
            sections=sections,
            categories=categories,
            metadata=metadata
        )
    
    def _identify_page_type(self, title: str, wikitext: str) -> NodeType:
        """
        Identify the type of wiki page based on title prefix (per DE Wiki guideline).
        
        Primary method: Title must start with one of the 4 prefixes:
        - Concept:
        - Implementation:
        - Workflow:
        - Resource:
        
        Fallback: Content-based heuristics if no prefix found.
        """
        # Check title patterns first (primary method per guideline)
        for node_type, patterns in self.TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    return node_type
        
        # Fallback: Check content structure for pages without proper prefixes
        text_lower = wikitext.lower()
        
        if '<syntaxhighlight' in text_lower or '<source' in text_lower:
            return NodeType.IMPLEMENTATION
        elif 'workflow steps' in text_lower or '=== step' in text_lower:
            return NodeType.WORKFLOW
        elif 'schema' in text_lower and ('location & access' in text_lower or 'where used' in text_lower):
            return NodeType.RESOURCE
        elif 'theoretical' in text_lower or 'principle' in text_lower:
            return NodeType.CONCEPT
        
        return NodeType.UNKNOWN
    
    def _extract_sections(self, wikitext: str) -> Dict[str, str]:
        """Extract all sections from wikitext"""
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        # Split by lines and process
        lines = wikitext.split('\n')
        section_pattern = r'^(=+)\s*([^=]+)\s*\1'
        
        for line in lines:
            match = re.match(section_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Start new section
                current_section = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_categories(self, wikitext: str) -> List[str]:
        """Extract categories from wikitext"""
        category_pattern = r'\[\[Category:([^\]]+)\]\]'
        return re.findall(category_pattern, wikitext)
    
    def _extract_metadata(self, wikitext: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract additional metadata from the page"""
        metadata = {}
        
        # Count code blocks
        code_blocks = len(re.findall(r'<syntaxhighlight|<source|<pre', wikitext, re.IGNORECASE))
        metadata['code_blocks'] = code_blocks
        
        # Count tables
        metadata['tables'] = len(re.findall(r'\{\|', wikitext))
        
        # Check for specific section types
        metadata['has_examples'] = any('example' in s.lower() for s in sections.keys())
        metadata['has_prerequisites'] = 'Prerequisites' in sections
        metadata['has_references'] = 'References' in sections or 'See Also' in sections
        
        return metadata
    
    def _extract_edges(self, page_title: str, wikitext: str) -> List[WikiEdge]:
        """
        Extract all edges (links) from the page.
        
        Supports two link types per DE Wiki guideline:
        1. Semantic properties: [[property::Target]] - explicit relationship
        2. Regular links: [[Target]] or [[Target|Display]] - inferred relationship
        """
        edges = []
        
        # Get sections for context
        sections = self._extract_sections(wikitext)
        
        # Pattern for semantic properties: [[property::Target]]
        # Captures property name and target
        semantic_pattern = r'\[\[([a-z_]+)::([^\]]+)\]\]'
        
        # Pattern for regular wiki links: [[Target|Display]] or [[Target]]
        # Must not match semantic properties (no :: in first group)
        regular_link_pattern = r'\[\[([^:\|\]]+:[^\|\]]+|[^:\|\]]+)(?:\|([^\]]+))?\]\]'
        
        # Process each section
        for section_name, section_content in sections.items():
            # First, extract semantic property links (explicit relationships)
            for match in re.finditer(semantic_pattern, section_content):
                property_name = match.group(1).strip()
                target = match.group(2).strip()
                
                # Skip non-relationship properties (owner, status, etc.)
                if property_name not in self.SMW_PROPERTY_MAP:
                    continue
                
                # Skip category and file links
                if target.startswith('Category:') or target.startswith('File:'):
                    continue
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(section_content), match.end() + 50)
                context = section_content[start:end].replace('\n', ' ').strip()
                
                # Get relationship from property name
                relationship = self.SMW_PROPERTY_MAP[property_name]
                
                # Create edge with semantic property
                edge = WikiEdge(
                    source_id=page_title,
                    target_id=target,
                    relationship=relationship,
                    context=context,
                    section=section_name,
                    link_text=f"{property_name}::{target}"
                )
                edges.append(edge)
            
            # Second, extract regular links (inferred relationships)
            for match in re.finditer(regular_link_pattern, section_content):
                target = match.group(1).strip()
                
                # Skip category and file links
                if target.startswith('Category:') or target.startswith('File:'):
                    continue
                
                # Skip if this is actually a semantic property (double check)
                if '::' in target:
                    continue
                
                link_text = match.group(2).strip() if match.group(2) else target
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(section_content), match.end() + 50)
                context = section_content[start:end].replace('\n', ' ').strip()
                
                # Identify relationship type from context
                relationship = self._identify_relationship(
                    target, link_text, context, section_name
                )
                
                # Create edge
                edge = WikiEdge(
                    source_id=page_title,
                    target_id=target,
                    relationship=relationship,
                    context=context,
                    section=section_name,
                    link_text=link_text
                )
                edges.append(edge)
        
        return edges
    
    def _identify_relationship(self, target: str, link_text: str, 
                              context: str, section: str) -> EdgeRelationship:
        """
        Identify the relationship type based on link text, context, and section.
        
        This is a fallback for regular links. Semantic properties [[property::Target]]
        have explicit relationships and don't use this method.
        """
        combined_text = f"{section} {link_text} {context}".lower()
        
        # Check section-based hints first (per guideline structure)
        section_lower = section.lower()
        if 'see also' in section_lower or 'related' in section_lower:
            return EdgeRelationship.RELATED_TO
        elif 'implementation' in section_lower and 'Concept:' in target:
            return EdgeRelationship.IMPLEMENTED_BY
        elif 'resource' in section_lower or 'input' in section_lower:
            if 'Resource:' in target:
                return EdgeRelationship.CONSUMES
        elif 'output' in section_lower:
            if 'Resource:' in target:
                return EdgeRelationship.PRODUCES
        
        # Check relationship patterns in context
        for relationship, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return relationship
        
        # Default: use UNKNOWN for generic links
        return EdgeRelationship.UNKNOWN
    
    def _clean_page_name(self, title: str) -> str:
        """Clean page name by removing type prefixes (per DE Wiki guideline)"""
        # Only these 4 prefixes are defined in the guideline
        prefixes = ['Concept:', 'Implementation:', 'Workflow:', 'Resource:']
        
        clean_name = title
        for prefix in prefixes:
            if title.startswith(prefix):
                clean_name = title[len(prefix):]
                break
        
        return clean_name.strip()