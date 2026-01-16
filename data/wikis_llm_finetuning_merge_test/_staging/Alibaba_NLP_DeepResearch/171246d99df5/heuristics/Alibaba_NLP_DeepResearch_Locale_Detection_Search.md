# Heuristic: Locale_Detection_Search

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Experience|Internal|Code analysis of tool_search.py]]
|-
! Domains
| [[domain::Web_Search]], [[domain::Internationalization]], [[domain::LLM_Agents]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Automatic locale detection for web search queries based on presence of Chinese characters, switching between US/English and China/Chinese search configurations.

=== Description ===
The Search tool automatically detects the language of user queries by checking for Chinese characters in the Unicode range `\u4E00-\u9FFF`. When Chinese characters are detected, the search is configured for Chinese locale (China location, Chinese language). Otherwise, it defaults to US/English locale. This improves search relevance for international users without requiring explicit locale configuration.

=== Usage ===
Use this heuristic implicitly when making search queries. The detection is automatic and requires no user intervention. Useful for agents serving multilingual user bases or processing datasets with mixed language content.

== The Insight (Rule of Thumb) ==
* **Action:** Before each search, check query text for Chinese characters using Unicode range detection.
* **Detection Range:** `\u4E00` to `\u9FFF` (CJK Unified Ideographs block)
* **Chinese Config:** `location="China"`, `gl="cn"`, `hl="zh-cn"`
* **English Config:** `location="United States"`, `gl="us"`, `hl="en"`
* **Trade-off:** Simple binary detection; doesn't handle other Asian languages or mixed queries well.

== Reasoning ==
Search engine results vary significantly by locale:

1. **Relevance:** Chinese queries return better results from Chinese search indices
2. **Content availability:** Some content is only indexed in regional search
3. **Ranking differences:** Local content is prioritized in localized searches
4. **No explicit config needed:** Users don't need to specify their locale

The Unicode-based detection is fast (single pass through string) and reliable for Chinese text.

== Code Evidence ==

Chinese character detection and locale switching from `tool_search.py:39-56`:
<syntaxhighlight lang="python">
def google_search_with_serp(self, query: str):
    def contains_chinese_basic(text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)
    conn = http.client.HTTPSConnection("google.serper.dev")
    if contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        })

    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        })
</syntaxhighlight>

Similar detection pattern in tool_python.py for code comments:
<syntaxhighlight lang="python">
CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')

def has_chinese_chars(data: Any) -> bool:
    return bool(CHINESE_CHAR_RE.search(text))
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_Search_call]]
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_WebSearch_call]]
* [[used_by::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
