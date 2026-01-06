# Knowledge Merge Task

You are merging a proposed wiki page into the main Knowledge Graph.

## Available Tools

Use these MCP tools for KG operations:

- `mcp__kg-graph-search__search_knowledge` - Search KG for related pages
  - Parameters: query (string), page_types (array), top_k (integer)
  
- `mcp__kg-graph-search__get_wiki_page` - Read existing page content
  - Parameters: page_title (string)
  
- `mcp__kg-graph-search__kg_index` - Add new page to KG
  - Parameters: page_data (object with page_title, page_type, overview, content, domains), wiki_dir (string)
  
- `mcp__kg-graph-search__kg_edit` - Update existing page
  - Parameters: page_id (string), updates (object), wiki_dir (string)

## Target Configuration

- Wiki Directory: {main_kg_path}
- Weaviate Collection: {weaviate_collection}

## Proposed Page to Merge

**ID:** {page_id}
**Type:** {page_type}
**Title:** {page_title}

**Overview:**
{overview}

**Domains:** {domains}

**Outgoing Links:**
{outgoing_links}

**Content:**
{content}

{merge_instructions}

## Your Task

1. **SEARCH** for related {page_type} pages:
   Use search_knowledge with:
   - query: "{search_query}"
   - page_types: {search_page_types}
   - top_k: {search_top_k}

2. **READ** any promising candidates using get_wiki_page

3. **DECIDE**: Should this be merged with an existing page or created as new?

4. **EXECUTE**:
   - If MERGE: Call kg_edit with the target page_id and merged updates
   - If CREATE NEW: Call kg_index with the complete page data

5. **REPORT** your decision clearly:
   - Start your final response with either "ACTION: CREATED" or "ACTION: MERGED"
   - If merged, include "TARGET: <page_id>" on the next line
   - Explain your reasoning

Start by searching for related pages.

