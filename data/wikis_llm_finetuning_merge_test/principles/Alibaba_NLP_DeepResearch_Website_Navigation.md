# Principle: Website_Navigation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|WebWalker|https://arxiv.org/abs/2501.07572]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Web_Navigation]], [[domain::Agent_Systems]], [[domain::Information_Extraction]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Autonomous website navigation agent that traverses web pages by clicking links and extracting information to answer questions about specific websites.

=== Description ===

Website Navigation enables agents to explore websites interactively:

1. **Page observation** - Extract visible content and clickable links
2. **Action selection** - LLM decides which link to click
3. **Memory accumulation** - Build understanding as pages are visited
4. **Goal tracking** - Monitor progress toward answering query
5. **Termination** - Decide when sufficient information is gathered

Unlike web search, this focuses on navigating within a single website.

=== Usage ===

Use Website Navigation when:
- Questions require exploring a specific website
- Information spans multiple linked pages
- Need to follow navigation patterns (menu, breadcrumbs)
- Building website-specific agents

Not suitable for:
- Cross-website search (use Web Search)
- Simple single-page extraction

== Theoretical Basis ==

Website navigation as graph traversal:

<math>
\pi(s_t) = \arg\max_a Q(s_t, a)
</math>

Where state s_t is current page + memory, and actions are clickable links.

'''Website Navigation Pattern:'''
<syntaxhighlight lang="python">
class WebWalker:
    """Website navigation agent."""

    def __init__(self, llm, max_rounds: int = 10):
        self.llm = llm
        self.max_rounds = max_rounds
        self.memory = []

    async def run(self, website: str, query: str) -> str:
        """
        Navigate website to answer query.

        Args:
            website: Starting URL
            query: User question

        Returns:
            Answer extracted from website
        """
        current_url = website
        current_content = await self.visit(current_url)

        for round in range(self.max_rounds):
            # Extract clickable links
            links = self.extract_links(current_content)

            # Build observation
            observation = {
                'url': current_url,
                'content': current_content,
                'links': links,
                'memory': self.memory
            }

            # LLM decides action
            action = await self.llm.decide(query, observation)

            if action['type'] == 'answer':
                return action['answer']
            elif action['type'] == 'click':
                # Navigate to clicked link
                current_url = action['url']
                current_content = await self.visit(current_url)
                # Update memory
                self.memory.append({
                    'url': current_url,
                    'summary': action.get('summary', '')
                })

        return "Max rounds reached without answer"
</syntaxhighlight>

Navigation strategies:
- **Breadth-first**: Explore menu items systematically
- **Goal-directed**: Follow links most relevant to query
- **Backtracking**: Return to previous pages if stuck

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebWalker_Agent]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Browser_Agent]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
