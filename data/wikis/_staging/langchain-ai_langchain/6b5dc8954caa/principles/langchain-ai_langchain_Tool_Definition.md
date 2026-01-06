{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Toolformer|https://arxiv.org/abs/2302.04761]]
* [[source::Doc|OpenAI Function Calling|https://platform.openai.com/docs/guides/function-calling]]
* [[source::Doc|LangChain Tools|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Tool_Calling]], [[domain::Agents]], [[domain::Function_Calling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Methodology for defining callable functions that extend language model capabilities beyond text generation.

=== Description ===

Tool Definition is the process of specifying external functions that a language model can invoke to perform actions or retrieve information. This transforms LLMs from pure text generators into agents that can interact with external systems, APIs, databases, and computational tools.

A well-defined tool consists of three essential components:
* **Name:** Unique identifier the model uses to select the tool
* **Description:** Natural language explanation helping the model understand when to use it
* **Schema:** Structured specification of expected inputs (types, constraints, descriptions)

This principle addresses the fundamental limitation that LLMs cannot directly execute code or access external systems. By defining tools with clear schemas, models can generate structured tool calls that are then executed by the runtime.

=== Usage ===

Use Tool Definition when:
* Building agents that need to interact with external APIs
* Creating systems that combine LLM reasoning with computational capabilities
* Implementing retrieval-augmented generation (RAG) with custom retrievers
* Developing conversational interfaces that can take real-world actions

Key design considerations:
* Clear, unambiguous tool names and descriptions
* Strongly-typed input schemas with validation
* Appropriate granularity (not too broad, not too narrow)

== Theoretical Basis ==

Tool Definition implements the **Function Calling** paradigm introduced in modern LLMs. The theoretical foundation involves:

'''1. Schema-Based Tool Specification'''

Tools are defined as JSON Schema objects that describe their interface:

<syntaxhighlight lang="json">
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and state, e.g., 'San Francisco, CA'"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit"
      }
    },
    "required": ["location"]
  }
}
</syntaxhighlight>

'''2. Tool Selection Process'''

The model learns to:
1. Parse user intent from natural language
2. Match intent to available tools
3. Extract parameters from context
4. Generate structured tool call

<syntaxhighlight lang="python">
# Pseudo-code for tool selection
def select_tool(user_message: str, available_tools: list[Tool]) -> ToolCall | None:
    # Model evaluates each tool's relevance
    for tool in available_tools:
        if tool.description matches user_intent(user_message):
            params = extract_parameters(user_message, tool.schema)
            return ToolCall(name=tool.name, arguments=params)
    return None  # No tool needed
</syntaxhighlight>

'''3. Execution and Response Integration'''

After tool execution, results are fed back to the model:

<syntaxhighlight lang="python">
# Pseudo-code for tool execution flow
def agent_loop(messages, tools):
    while True:
        response = model.generate(messages, tools=tools)

        if response.has_tool_calls:
            for call in response.tool_calls:
                result = execute_tool(call.name, call.arguments)
                messages.append(ToolMessage(result, tool_call_id=call.id))
        else:
            return response.content  # Final answer
</syntaxhighlight>

'''4. Type Safety and Validation'''

Pydantic models provide runtime validation:

<syntaxhighlight lang="python">
from pydantic import BaseModel, Field, validator

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=100)

    @validator('query')
    def sanitize_query(cls, v):
        return v.strip()
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_BaseTool_and_StructuredTool]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 2)
