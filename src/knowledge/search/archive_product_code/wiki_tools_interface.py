"""
Wiki tools for CodeAct agent.
Provides tools for editing and searching wiki pages.
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# Wiki Edit Tool
WikiEditTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='wiki_edit',
        description="""Edit a wiki page.

This tool allows you to perform various edit actions on wiki pages:
- create: Create a new wiki page
- edit: Replace entire content of an existing page
- append: Add content to the end of a page
- prepend: Add content to the beginning of a page
- delete: Delete a wiki page
- move: Rename/move a wiki page

The tool connects to the configured wiki server.""",
        parameters={
            'type': 'object',
            'properties': {
                'wiki_url': {
                    'type': 'string',
                    'description': 'Wiki base URL with https:// prefix (e.g., "https://acme.leeroo.com" for domain acme.leeroo.com)',
                },
                'action': {
                    'type': 'string',
                    'description': 'Action to perform',
                    'enum': ['create', 'edit', 'append', 'prepend', 'delete', 'move'],
                },
                'title': {
                    'type': 'string',
                    'description': 'Title of the wiki page to edit. It must be WikiText format, never use Markdown format.',
                },
                'text': {
                    'type': 'string',
                    'description': 'Content for the page (required for create/edit/append/prepend actions). It must be WikiText format, never use Markdown format.',
                },
                'new_title': {
                    'type': 'string',
                    'description': 'New title for the page (required for move action). It must be WikiText format, never use Markdown format.',
                },
                'reason': {
                    'type': 'string',
                    'description': 'Edit summary or reason for the change',
                },
            },
            'required': ['wiki_url', 'action', 'title'],
        },
    ),
)

# Wiki Get Tool
WikiGetTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='wiki_get',
        description="""Retrieve the full content of a specific wiki page.

Use this tool after searching to get the complete content of a page you're interested in.
Requires the wiki URL and page title.""",
        parameters={
            'type': 'object',
            'properties': {
                'wiki_url': {
                    'type': 'string',
                    'description': 'Wiki base URL with https:// prefix (e.g., "https://acme.leeroo.com" for domain acme.leeroo.com). Use the same domain from wiki_search results but add https:// prefix.',
                },
                'title': {
                    'type': 'string',
                    'description': 'Exact page title to retrieve',
                },
            },
            'required': ['wiki_url', 'title'],
        },
    ),
)

# Wiki Search Tool
WikiSearchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='wiki_search',
        description="""Search wiki content for documentation, solutions, or related information.

The search method is configured in system settings and can be:
- keyword: Traditional keyword-based search
- semantic: AI-powered semantic search for conceptual matches
- hybrid: Combines keyword and semantic search for best results

Search type, result limit, and semantic weight are configured in system settings.
The tool searches within the configured wiki domain and returns relevant page information.""",
        parameters={
            'type': 'object',
            'properties': {
                'wiki_domain': {
                    'type': 'string',
                    'description': 'Wiki domain to search within (e.g., "wiki.example.com")',
                },
                'query': {
                    'type': 'string',
                    'description': 'Search query string',
                },
                'page_type': {
                    'type': 'string',
                    'description': 'Optional filter by page type. Valid values: "concept" (conceptual/theoretical pages), "implementation" (code/technical implementation), "workflow" (process/procedure guides), "resource" (reference materials/links)',
                    'enum': ['concept', 'implementation', 'workflow', 'resource'],
                },
            },
            'required': ['wiki_domain', 'query'],
        },
    ),
)
