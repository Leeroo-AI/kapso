#!/usr/bin/env python3
"""
Standalone test for the get_page_structure MCP tool.

Tests that the tool correctly returns sections_definition.md content
for each page type.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.gated_mcp.gates.base import GateConfig
from src.knowledge.gated_mcp.gates.kg_gate import KGGate


async def test_get_page_structure():
    """Test the get_page_structure handler directly via KGGate."""
    
    print("=" * 60)
    print("Testing get_page_structure MCP tool (via KGGate)")
    print("=" * 60)
    
    # Create KGGate instance
    config = GateConfig(enabled=True, params={})
    gate = KGGate(config)
    
    # Test all valid page types
    valid_types = ["principle", "implementation", "environment", "heuristic", "workflow"]
    
    all_passed = True
    for page_type in valid_types:
        print(f"\n--- Testing page_type: {page_type} ---")
        
        try:
            result = await gate.handle_call("get_page_structure", {"page_type": page_type})
            
            if result:
                text_content = result[0].text if result else "No result"
                
                # Check if it's an actual error (starts with error message patterns)
                is_error = (
                    text_content.startswith("Invalid page type:") or
                    text_content.startswith("Sections definition not found") or
                    text_content.startswith("Error:")
                )
                
                if is_error:
                    print(f"  ERROR: {text_content[:200]}")
                    all_passed = False
                else:
                    # Show first 300 chars of successful result
                    preview = text_content[:300].replace('\n', '\n  ')
                    print(f"  SUCCESS: {len(text_content)} chars")
                    print(f"  Preview:\n  {preview}...")
            else:
                print(f"  ERROR: Empty result")
                all_passed = False
                
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            all_passed = False
    
    # Test invalid page type
    print(f"\n--- Testing invalid page_type: 'invalid' ---")
    try:
        result = await gate.handle_call("get_page_structure", {"page_type": "invalid"})
        text_content = result[0].text if result else "No result"
        if text_content.startswith("Invalid page type:"):
            print(f"  EXPECTED ERROR (correct): {text_content[:100]}")
        else:
            print(f"  UNEXPECTED: {text_content[:200]}")
            all_passed = False
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        all_passed = False
    
    # Test missing page_type
    print(f"\n--- Testing missing page_type ---")
    try:
        result = await gate.handle_call("get_page_structure", {})
        text_content = result[0].text if result else "No result"
        if text_content.startswith("Invalid page type:"):
            print(f"  EXPECTED ERROR (correct): {text_content[:100]}")
        else:
            print(f"  UNEXPECTED: {text_content[:200]}")
            all_passed = False
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)
    
    return all_passed


def check_wiki_structure_paths():
    """Check if the wiki_structure directory and files exist."""
    
    print("\n" + "=" * 60)
    print("Checking wiki_structure paths")
    print("=" * 60)
    
    # Path relative to kg_gate.py
    kg_gate_path = Path(__file__).parent.parent / "src" / "knowledge" / "gated_mcp" / "gates" / "kg_gate.py"
    wiki_structure_dir = kg_gate_path.parent.parent.parent / "wiki_structure"
    
    print(f"\nkg_gate.py location: {kg_gate_path}")
    print(f"  Exists: {kg_gate_path.exists()}")
    
    print(f"\nwiki_structure_dir (computed): {wiki_structure_dir}")
    print(f"  Exists: {wiki_structure_dir.exists()}")
    
    if wiki_structure_dir.exists():
        print(f"\nContents of wiki_structure_dir:")
        for item in sorted(wiki_structure_dir.iterdir()):
            print(f"  - {item.name}/ " if item.is_dir() else f"  - {item.name}")
            if item.is_dir():
                for subitem in sorted(item.iterdir()):
                    print(f"      - {subitem.name}")
    
    # Check each page type
    page_types = ["principle", "implementation", "environment", "heuristic", "workflow"]
    print(f"\nChecking sections_definition.md for each page type:")
    for page_type in page_types:
        sections_file = wiki_structure_dir / f"{page_type}_page" / "sections_definition.md"
        exists = sections_file.exists()
        print(f"  {page_type}: {sections_file}")
        print(f"    Exists: {exists}")
        if exists:
            size = sections_file.stat().st_size
            print(f"    Size: {size} bytes")


if __name__ == "__main__":
    # First check paths
    check_wiki_structure_paths()
    
    # Then run async test
    passed = asyncio.run(test_get_page_structure())
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)
