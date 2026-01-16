# Principle: Page_Content_Extraction

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Information_Extraction]], [[domain::NLP]], [[domain::Summarization]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Query-focused content extraction that processes webpage responses to identify and summarize information relevant to the user's question.

=== Description ===

Page Content Extraction transforms verbose webpage content into concise, actionable information. The process:

1. **Clean content** - Remove boilerplate, ads, navigation
2. **Identify relevance** - Match content to user query
3. **Extract key points** - Pull out relevant facts, dates, entities
4. **Summarize** - Compress to fit agent context limits
5. **Structure output** - Format for agent memory

This is distinct from simple truncation - it actively selects the most relevant portions.

=== Usage ===

Use Page Content Extraction when:
- Webpage content exceeds context limits
- Need to focus on query-relevant information
- Building agent memory from visited pages
- Converting HTML to structured knowledge

== Theoretical Basis ==

Extraction follows query-focused summarization:

'''Extraction Pattern:'''
<syntaxhighlight lang="python">
def process_response(content: str, query: str, max_length: int = 2000) -> str:
    """
    Extract query-relevant content from webpage.

    Args:
        content: Raw webpage content
        query: User's question for context
        max_length: Maximum output length

    Returns:
        Extracted and summarized content
    """
    # Step 1: Clean the content
    clean_content = remove_boilerplate(content)

    # Step 2: Split into sentences/paragraphs
    segments = segment_text(clean_content)

    # Step 3: Score relevance to query
    scored = [(seg, relevance_score(seg, query)) for seg in segments]

    # Step 4: Select top segments
    selected = sorted(scored, key=lambda x: -x[1])
    result = ""
    for seg, score in selected:
        if len(result) + len(seg) > max_length:
            break
        result += seg + "\n"

    # Step 5: Format output
    return f"Relevant information:\n{result}"
</syntaxhighlight>

Relevance scoring methods:
- **Keyword overlap**: Count query terms in segment
- **Embedding similarity**: Cosine similarity of embeddings
- **LLM scoring**: Ask LLM to rate relevance

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Tool_Explore]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
