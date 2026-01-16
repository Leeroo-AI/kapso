# Principle: Prompt_Construction

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

Template-based prompt construction for multimodal agents. Defines tool schemas in JSON format.

=== Description ===

Prompt Construction is the systematic process of building effective prompts for multimodal vision-language agents. This involves constructing system prompts that define the agent's capabilities, tool schemas, and behavioral guidelines.

Key components of prompt construction:

1. **System Prompt** - Defines agent identity, capabilities, and response format
2. **Tool Schemas** - JSON definitions of available tools with parameters
3. **Template Variables** - Placeholders for dynamic content ({Question}, {Image_url})
4. **Response Format** - Expected tags for thinking, tool calls, and answers

The prompt template ensures:
- Consistent agent behavior across samples
- Proper tool invocation format
- Clear reasoning structure
- Reproducible outputs

=== Usage ===

Use prompt construction patterns when:
- Configuring multimodal agent behavior
- Adding new tools to the agent's toolkit
- Customizing response formats
- Adapting agents for specific benchmarks

Prompt templates are defined at initialization and instantiated per-sample.

== Theoretical Basis ==

The prompt construction pattern:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Template-based prompt construction
prompt_ins = """You are a helpful assistant that can answer questions about images using tools.

## Available Tools

{tools_json}

## Response Format

Use these tags in your responses:
- <think>Your reasoning process</think>
- <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
- <answer>Your final answer</answer>

## Current Task

Question: {Question}
Image: {Image_url}
"""

# Tool schema definition
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "VLSearchImage",
            "description": "Search for visually similar images",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs of images to search"
                    }
                },
                "required": ["image_urls"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search queries"
                    }
                },
                "required": ["queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit URLs and extract information",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "goal": {"type": "string"}
                },
                "required": ["urls", "goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    }
]

# Instantiate prompt for sample
def construct_prompt(sample: Dict, tools: List[Dict]) -> str:
    return prompt_ins.format(
        Question=sample["prompt"],
        Image_url=sample["image_url"],
        tools_json=json.dumps(tools, indent=2)
    )
</syntaxhighlight>

The JSON tool schema follows OpenAI's function calling format, enabling:
- Clear parameter definitions with types
- Required vs optional parameter distinction
- Nested object and array support
- Description fields for LLM understanding

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Prompt_Construction_Pattern]]
