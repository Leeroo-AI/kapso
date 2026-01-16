# Implementation: Prompt_Construction_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|agent_eval.py|WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py]]
|-
! Domains
| [[domain::Prompt_Engineering]], [[domain::Multimodal]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Pattern documentation for the prompt template system used in OmniSearch multimodal agent. Defines placeholders, tool schemas, and response format.

=== Description ===

The Prompt Construction Pattern documents the template-based approach used in agent_eval.py for constructing multimodal agent prompts. The `prompt_ins` template (lines 30-128) defines the system prompt with placeholders that are filled at inference time.

Key pattern elements:
- `{Question}` placeholder for the user's question
- `{Image_url}` placeholder for the processed image data URL
- Tool definitions in JSON schema format
- Response tag specifications (`<think>`, `<tool_call>`, `<answer>`)

This is a '''Pattern Document''' - it describes the interface and structure rather than a single callable function.

=== Usage ===

Reference this pattern when:
- Understanding the agent's prompt structure
- Adding new tools to the multimodal agent
- Customizing response format requirements
- Debugging agent behavior related to prompting

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py
* '''Lines:''' 30-128

=== Interface ===

'''Template Placeholders:'''
{| class="wikitable"
|-
! Placeholder !! Type !! Description
|-
| {Question} || str || User's question about the image
|-
| {Image_url} || str || Base64 data URL of the processed image
|}

'''Response Tags:'''
{| class="wikitable"
|-
! Tag !! Purpose !! Content
|-
| `<think>...</think>` || Reasoning || Chain-of-thought reasoning (optional)
|-
| `<tool_call>...</tool_call>` || Tool Invocation || JSON with name and arguments
|-
| `<answer>...</answer>` || Final Response || The answer to the user's question
|}

=== Template Structure ===
<syntaxhighlight lang="text">
prompt_ins = """
You are a multimodal assistant that can analyze images and search the web.

## Tools

You have access to the following tools:

### VLSearchImage
Search for visually similar images using reverse image search.
Parameters:
- image_urls: List[str] - URLs of images to search

### web_search
Search the web for information.
Parameters:
- queries: List[str] - Search queries to execute

### visit
Visit web pages and extract information.
Parameters:
- urls: List[str] - URLs to visit
- goal: str - What information to extract

### code
Execute Python code in a sandbox.
Parameters:
- code: str - Python code to execute

## Response Format

Think step by step using <think> tags.
Call tools using <tool_call> tags with JSON format.
Provide your final answer in <answer> tags.

## Question

{Question}

## Image

{Image_url}
"""
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Template Variables) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Question || str || Yes || The question to answer about the image
|-
| Image_url || str || Yes || Base64-encoded data URL (data:image/jpeg;base64,...)
|}

=== Outputs (Expected Response Structure) ===
{| class="wikitable"
|-
! Component !! Format !! Example
|-
| Thinking || `<think>reasoning</think>` || `<think>I need to identify this landmark...</think>`
|-
| Tool Call || `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` || `<tool_call>{"name": "VLSearchImage", "arguments": {"image_urls": ["..."]}}</tool_call>`
|-
| Answer || `<answer>response</answer>` || `<answer>This is the Eiffel Tower in Paris.</answer>`
|}

== Usage Examples ==

=== Basic Prompt Instantiation ===
<syntaxhighlight lang="python">
# Template defined in agent_eval.py
prompt_ins = """..."""  # Full template

# Sample data
sample = {
    "file_path": "/data/images/landmark.jpg",
    "prompt": "What is this building and when was it constructed?"
}

# Process image and get data URL
img, data_url = agent.process_image(sample["file_path"])

# Instantiate prompt
final_prompt = prompt_ins.format(
    Question=sample["prompt"],
    Image_url=data_url
)
</syntaxhighlight>

=== Tool Call JSON Format ===
<syntaxhighlight lang="python">
# VLSearchImage tool call
vl_search_call = {
    "name": "VLSearchImage",
    "arguments": {
        "image_urls": ["https://example.com/uploaded_image.jpg"]
    }
}

# web_search tool call
search_call = {
    "name": "web_search",
    "arguments": {
        "queries": ["Eiffel Tower history", "Eiffel Tower construction date"]
    }
}

# visit tool call
visit_call = {
    "name": "visit",
    "arguments": {
        "urls": ["https://en.wikipedia.org/wiki/Eiffel_Tower"],
        "goal": "When was the Eiffel Tower constructed?"
    }
}

# code tool call
code_call = {
    "name": "code",
    "arguments": {
        "code": "print(1889 - 1887)  # Construction duration"
    }
}
</syntaxhighlight>

=== Expected Agent Response Flow ===
<syntaxhighlight lang="text">
# Turn 1: Agent thinks and calls VLSearchImage
<think>
I see a tall iron tower in the image. Let me search for visually similar images.
</think>
<tool_call>
{"name": "VLSearchImage", "arguments": {"image_urls": ["data:image/jpeg;base64,..."]}}
</tool_call>

# Turn 2: After receiving search results, agent follows up
<think>
The reverse image search shows this is the Eiffel Tower. Let me search for construction details.
</think>
<tool_call>
{"name": "web_search", "arguments": {"queries": ["Eiffel Tower construction date"]}}
</tool_call>

# Turn 3: Agent provides final answer
<think>
Based on the search results, I now have the information needed.
</think>
<answer>
This is the Eiffel Tower in Paris, France. It was constructed between 1887 and 1889
for the 1889 World's Fair (Exposition Universelle).
</answer>
</syntaxhighlight>

=== Customizing for Different Benchmarks ===
<syntaxhighlight lang="python">
# HLE benchmark - complex reasoning questions
hle_template = prompt_ins + """
Note: This is a challenging question requiring careful reasoning.
Take your time to gather all necessary information before answering.
"""

# MMSearch benchmark - visual search focused
mmsearch_template = prompt_ins + """
Focus on using visual search to identify entities in the image.
Verify information through web search before providing your answer.
"""

# SimpleVQA - direct answers
simplevqa_template = prompt_ins + """
Provide a concise, direct answer based on the image content.
"""
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Prompt_Construction]]
