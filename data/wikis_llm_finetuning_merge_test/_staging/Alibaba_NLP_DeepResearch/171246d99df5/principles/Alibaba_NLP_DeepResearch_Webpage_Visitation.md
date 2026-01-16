# Principle: Webpage_Visitation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::API|Jina AI Reader|https://jina.ai/reader/]]
* [[source::Paper|WebGPT: Browser-assisted question-answering with human feedback|https://arxiv.org/abs/2112.09332]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Web_Scraping]], [[domain::Content_Extraction]], [[domain::Agent_Tools]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Webpage content retrieval and LLM-based summarization. Uses Jina AI reader service to convert webpages to markdown and extracts goal-relevant information.

=== Description ===

Webpage visitation is a critical capability for autonomous research agents. It allows the agent to retrieve the full content of webpages discovered through search and extract relevant information based on a specified goal.

The implementation follows a two-stage process:

1. **Content Retrieval** - Uses Jina AI reader service (r.jina.ai) to fetch and convert webpage content to clean markdown
2. **Goal-Directed Summarization** - An LLM extracts evidence and summary relevant to the user's goal

Key features:
- **Batch URL Processing** - Multiple URLs can be visited in a single tool call
- **Token Truncation** - Content is truncated to 95,000 tokens before summarization
- **Retry Logic** - Multiple retries for both content fetching and summarization
- **Structured Output** - Returns evidence and summary in JSON format
- **Timeout Protection** - 15-minute timeout for batch operations

The tool is designed to handle various failure modes gracefully, always returning structured output even when content cannot be accessed.

=== Usage ===

Use Webpage Visitation when:
- The agent needs to read full content of pages found via search
- Extracting specific information from a webpage based on a goal
- Processing multiple related URLs together

Output structure:
| Field | Description |
|-------|-------------|
| evidence | Direct quotes and data from the webpage relevant to the goal |
| summary | Synthesized paragraph with logical flow and goal relevance assessment |

== Theoretical Basis ==

Webpage visitation implements a retrieval-augmented extraction pattern:

<math>
\text{Evidence} = \text{Extract}(\text{Content}, \text{Goal})
</math>

<math>
\text{Summary} = \text{Summarize}(\text{Evidence}, \text{Goal})
</math>

The extraction uses a two-stage LLM process:
<math>
\text{Output} = \text{SummaryLLM}(\text{EXTRACTOR\_PROMPT}(\text{Truncate}(\text{Jina}(\text{URL})), \text{Goal}))
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Webpage Visitation Pattern
def visit(urls: Union[str, List[str]], goal: str) -> str:
    """Visit webpage(s) and extract goal-relevant information."""

    def visit_single(url: str, goal: str) -> str:
        # Stage 1: Fetch content via Jina AI
        for attempt in range(3):
            content = jina_reader.fetch(f"https://r.jina.ai/{url}")
            if content and not content.startswith("[visit] Failed"):
                break

        if not content:
            return format_error_response(url, goal)

        # Truncate to token limit
        content = truncate_to_tokens(content, max_tokens=95000)

        # Stage 2: LLM-based extraction
        prompt = EXTRACTOR_PROMPT.format(
            webpage_content=content,
            goal=goal
        )

        for retry in range(3):
            raw_output = summary_llm.call([{"role": "user", "content": prompt}])

            try:
                result = json.loads(raw_output)
                return format_success_response(url, goal, result)
            except json.JSONDecodeError:
                # Truncate content further and retry
                content = content[:int(0.7 * len(content))]
                continue

        return format_error_response(url, goal)

    # Handle single or batch URLs
    if isinstance(urls, str):
        return visit_single(urls, goal)
    else:
        results = []
        start_time = time.time()
        for url in urls:
            if time.time() - start_time > 900:  # 15 minute timeout
                results.append(format_timeout_response(url, goal))
            else:
                results.append(visit_single(url, goal))
        return "\n=======\n".join(results)
</syntaxhighlight>

Key visitation principles:
- **Goal-Directed Extraction**: Only content relevant to the goal is extracted
- **Progressive Truncation**: If summarization fails, content is truncated and retried
- **Structured Output**: JSON format ensures consistent parsing by the agent
- **Batch Efficiency**: Multiple URLs processed in one tool call reduce round trips

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Visit_call]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
