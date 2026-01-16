# Heuristic: Exponential_Backoff_Retry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Experience|Internal|Code analysis of react_agent.py, rag_system.py]]
|-
! Domains
| [[domain::Error_Handling]], [[domain::API_Integration]], [[domain::Reliability]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Retry strategy using exponential backoff with jitter for handling transient API failures in LLM calls and external service requests.

=== Description ===
DeepResearch makes many external API calls to LLM services, search engines, and webpage readers. These calls can fail due to rate limits, network issues, or service overload. The codebase implements exponential backoff retry patterns with random jitter to handle transient failures gracefully while avoiding thundering herd problems.

=== Usage ===
Use this heuristic when making API calls that may experience transient failures. Implement exponential backoff to prevent overwhelming services during outages and to comply with rate limits.

== The Insight (Rule of Thumb) ==
* **Action:** On API failure, wait `base_sleep_time * (2 ** attempt) + random.uniform(0, 1)` seconds before retry.
* **Base Sleep:** 1 second (configurable)
* **Max Sleep:** 30 seconds (cap to prevent excessive waits)
* **Max Retries:** 10 attempts for LLM calls, 3 for webpage parsing
* **Jitter:** Random 0-1 second added to prevent synchronized retries

== Reasoning ==
Exponential backoff is industry standard for API reliability:

1. **Rate limit compliance:** Increasing delays naturally comply with rate limits
2. **Service recovery:** Gives overloaded services time to recover
3. **Thundering herd prevention:** Jitter prevents all clients retrying simultaneously
4. **Resource efficiency:** Fewer wasted requests during outages
5. **User experience:** Eventually succeeds without manual intervention

The 30-second cap ensures users don't wait indefinitely for a single request.

== Code Evidence ==

Exponential backoff in LLM calls from `react_agent.py:101-106`:
<syntaxhighlight lang="python">
if attempt < max_tries - 1:
    sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
    sleep_time = min(sleep_time, 30)

    print(f"Retrying in {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)
else:
    print("Error: All retry attempts have been exhausted. The call has failed.")
</syntaxhighlight>

Tenacity-based retry with exponential wait from `rag_system.py:38`:
<syntaxhighlight lang="python">
@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=4, max=60))
async def call_model(self, query_text, sys_prompt="You are a helpful assistant."):
</syntaxhighlight>

Webpage parsing retry from `tool_visit.py:226-234`:
<syntaxhighlight lang="python">
parse_retry_times = 0
while parse_retry_times < 3:
    try:
        raw = json.loads(raw)
        break
    except:
        raw = summary_page_func(messages, max_retries=max_retries)
        parse_retry_times += 1

if parse_retry_times >= 3:
    # Handle failure case
</syntaxhighlight>

Generic retry mechanism from `llm/base.py:544-572`:
<syntaxhighlight lang="python">
def retry_model_service(
    func,
    max_retries: int = 10,
    exponential_base: float = 2.0,
):
    """Retry with exponential backoff"""
    if max_retries <= 0:  # no retry
        return func()

    # ... retry logic with exponential delay
    delay = min(delay * exponential_base, max_delay) * jitter
</syntaxhighlight>

Evaluation retry with exponential backoff from `WebWalker/src/evaluate.py:49-63`:
<syntaxhighlight lang="python">
def retry_with_backoff(self, func, max_retries=3):
    """Handles evaluation retries with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))  # Exponential backoff
            else:
                raise e  # Raise the exception if the last retry fails
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_Visit_call]]
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_Call_Llm_Judge]]
* [[used_by::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
