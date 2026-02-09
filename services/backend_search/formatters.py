"""
Result formatters for wiki search.

Formats search results as markdown for MCP tool responses.
"""

from typing import Any


def format_idea_results(query: str, result: Any) -> str:
    """
    Format idea search results as markdown.

    Args:
        query: Original search query
        result: Search result object with results list

    Returns:
        Markdown formatted string
    """
    if result.is_empty:
        return (
            f'# Idea Search: "{query}"\n\n'
            "No results found. Try wiki_code_search for implementation and code content."
        )

    parts = [
        f'# Idea Search: "{query}"\n',
        f"Found **{result.total_found}** results:\n",
    ]

    for i, item in enumerate(result.results, 1):
        parts.append(f"\n---\n## [{i}] {item.page_title}\n")
        parts.append(f"**Type:** {item.page_type} | **Score:** {item.score:.2f}\n")
        if item.domains:
            parts.append(f"**Domains:** {', '.join(item.domains)}\n")
        parts.append(f"\n### Overview\n{item.overview}\n")
        if item.content:
            # Ideas get shorter previews
            max_len = 800
            preview = item.content[:max_len] + ("..." if len(item.content) > max_len else "")
            parts.append(f"\n### Content\n{preview}\n")

    return "".join(parts)


def format_code_results(query: str, result: Any) -> str:
    """
    Format code search results as markdown.

    Args:
        query: Original search query
        result: Search result object with results list

    Returns:
        Markdown formatted string
    """
    if result.is_empty:
        return (
            f'# Code Search: "{query}"\n\n'
            "No results found. Try wiki_idea_search for concepts and principles."
        )

    parts = [
        f'# Code Search: "{query}"\n',
        f"Found **{result.total_found}** results:\n",
    ]

    for i, item in enumerate(result.results, 1):
        parts.append(f"\n---\n## [{i}] {item.page_title}\n")
        parts.append(f"**Type:** {item.page_type} | **Score:** {item.score:.2f}\n")
        if item.domains:
            parts.append(f"**Domains:** {', '.join(item.domains)}\n")
        parts.append(f"\n### Overview\n{item.overview}\n")
        if item.content:
            # Code gets longer previews to show full examples
            max_len = 1000
            preview = item.content[:max_len] + ("..." if len(item.content) > max_len else "")
            parts.append(f"\n### Content\n{preview}\n")

    return "".join(parts)
