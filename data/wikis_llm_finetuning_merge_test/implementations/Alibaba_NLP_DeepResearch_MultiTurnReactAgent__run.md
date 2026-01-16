# Implementation: MultiTurnReactAgent__run

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::NLP]], [[domain::Autonomous_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Main execution method for the MultiTurnReactAgent that implements the ReAct loop for autonomous web research.

=== Description ===

The `_run` method is the core execution engine of the MultiTurnReactAgent. It orchestrates the entire research process from question input to final answer extraction.

Key responsibilities:
1. **Question Extraction** - Parses the input data structure to extract the user question
2. **Message Initialization** - Sets up system prompt with current date and user query
3. **Loop Management** - Iterates up to MAX_LLM_CALL_PER_RUN (default 100) times
4. **Tool Dispatch** - Parses tool calls and routes to appropriate handlers
5. **Termination Detection** - Monitors for answer tags, time limits, and token limits
6. **Result Packaging** - Returns structured dict with question, answer, messages, prediction, and termination reason

=== Usage ===

Use `MultiTurnReactAgent._run()` when:
- Executing a research query through the agent
- Running evaluation benchmarks on question-answer datasets
- Processing batched research requests

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/react_agent.py
* '''Lines:''' 120-226

=== Signature ===
<syntaxhighlight lang="python">
def _run(self, data: Dict, model: str, **kwargs) -> Dict:
    """
    Execute the ReAct loop for autonomous research.

    Args:
        data: Dict containing:
            - item.question: str - The research question
            - item.answer: str - Ground truth answer (for evaluation)
            - item.messages: Optional alternative message format
            - planning_port: int - Port for vLLM server
        model: str - Model name for the vLLM server

    Returns:
        Dict with keys:
            - question: str - Original question
            - answer: str - Ground truth answer
            - messages: List[Dict] - Full conversation history
            - prediction: str - Agent's predicted answer
            - termination: str - Reason for loop termination:
                - "answer" - Normal completion
                - "exceed available llm calls" - Hit call limit
                - "answer not found" - Loop ended without answer
                - "generate an answer as token limit reached" - Forced answer
                - "format error: generate an answer as token limit reached"
                - "No answer found after 2h30mins" - Time limit
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| data || Dict || Yes || Input data containing question, answer, and server port
|-
| data["item"]["question"] || str || Yes* || The research question (*or messages format)
|-
| data["item"]["answer"] || str || Yes || Ground truth for evaluation
|-
| data["planning_port"] || int || Yes || Port number for vLLM inference server
|-
| model || str || Yes || Model identifier for the vLLM server
|-
| **kwargs || dict || No || Additional arguments (unused)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || Dict || Structured result containing all execution details
|-
| result["question"] || str || Original input question
|-
| result["answer"] || str || Ground truth answer (passthrough)
|-
| result["messages"] || List[Dict] || Complete conversation history
|-
| result["prediction"] || str || Agent's extracted answer
|-
| result["termination"] || str || Termination reason code
|}

== Usage Examples ==

=== Basic Execution ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

# Initialize agent
llm_config = {
    "model": "/models/Qwen2.5-72B-Instruct",
    "generate_cfg": {"temperature": 0.6, "top_p": 0.95}
}
agent = MultiTurnReactAgent(llm=llm_config)

# Prepare input data
data = {
    "item": {
        "question": "What was the population of Tokyo in 2023?",
        "answer": "approximately 14 million"
    },
    "planning_port": 8000
}

# Execute research
result = agent._run(data, model="qwen-72b")

print(f"Question: {result['question']}")
print(f"Prediction: {result['prediction']}")
print(f"Termination: {result['termination']}")
print(f"Rounds: {len([m for m in result['messages'] if m['role'] == 'assistant'])}")
</syntaxhighlight>

=== Batch Evaluation ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent
import json

# Initialize agent
agent = MultiTurnReactAgent(llm=llm_config)

# Load evaluation dataset
with open("eval_data/questions.jsonl") as f:
    questions = [json.loads(line) for line in f]

results = []
for item in questions:
    data = {
        "item": {
            "question": item["question"],
            "answer": item["answer"]
        },
        "planning_port": 8000
    }

    result = agent._run(data, model="qwen-72b")
    results.append(result)

    # Check termination status
    if result["termination"] == "answer":
        print(f"SUCCESS: {item['question'][:50]}...")
    else:
        print(f"FAILED ({result['termination']}): {item['question'][:50]}...")

# Save results
with open("results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
</syntaxhighlight>

=== Analyzing Conversation History ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

agent = MultiTurnReactAgent(llm=llm_config)
result = agent._run(data, model="qwen-72b")

# Analyze tool usage
tool_calls = []
for msg in result["messages"]:
    if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
        # Extract tool name
        content = msg["content"]
        if '"name"' in content:
            tool_name = content.split('"name"')[1].split('"')[1]
            tool_calls.append(tool_name)

print(f"Tools used: {tool_calls}")
print(f"Search calls: {tool_calls.count('search')}")
print(f"Visit calls: {tool_calls.count('visit')}")
print(f"Python calls: {tool_calls.count('PythonInterpreter')}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__init__]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_Search_call]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_Visit_call]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_PythonInterpreter_call]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Token_Limit_Management]]
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry]]
