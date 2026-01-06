{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|Structured Outputs|https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#guided-generation]]
|-
! Domains
| [[domain::LLM_Inference]], [[domain::Structured_Output]], [[domain::JSON_Generation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for generating constrained outputs using JSON schemas, regular expressions, grammars, or choice constraints.

=== Description ===
This workflow demonstrates vLLM's structured output generation capabilities, which constrain model outputs to match specific formats. Using `StructuredOutputsParams`, you can guarantee outputs match JSON schemas (via Pydantic models), regex patterns, EBNF grammars, or a fixed set of choices. This eliminates parsing errors and ensures reliable integration with downstream systems that expect structured data.

=== Usage ===
Execute this workflow when you need guaranteed structured output formats, such as JSON API responses, classification into predefined categories, formatted data extraction, or SQL generation. Structured outputs are essential for building reliable LLM-powered applications where output parsing must not fail.

== Execution Steps ==

=== Step 1: Constraint Definition ===
[[step::Principle:vllm-project_vllm_Constraint_Definition]]

Define the output constraint using one of the supported constraint types. Options include JSON Schema (from Pydantic models), regex patterns, choice lists, or EBNF grammars. Only one constraint type can be active per request.

'''Constraint types:'''
* `json` - JSON Schema dict or Pydantic model schema for structured data
* `regex` - Regular expression pattern for string format constraints
* `choice` - List of allowed string outputs for classification
* `grammar` - EBNF grammar for complex structural constraints
* `json_object` - Simpler JSON output mode without schema

=== Step 2: StructuredOutputsParams Configuration ===
[[step::Principle:vllm-project_vllm_StructuredOutputsParams_Configuration]]

Create a `StructuredOutputsParams` object with the chosen constraint. Additional options control whitespace handling, fallback behavior, and backend selection. The params object is then passed to `SamplingParams`.

'''Configuration options:'''
* Constraint field (`json`, `regex`, `choice`, `grammar`)
* `disable_fallback` - Disable fallback to unconstrained generation
* `disable_any_whitespace` - Stricter whitespace control
* `whitespace_pattern` - Custom whitespace regex
* Backend automatically selected based on constraint type

=== Step 3: SamplingParams Integration ===
[[step::Principle:vllm-project_vllm_Structured_SamplingParams]]

Integrate the structured outputs constraint into `SamplingParams` via the `structured_outputs` field. Other sampling parameters like temperature and max_tokens work alongside constraints. The constraint processor is applied during token sampling.

'''Integration pattern:'''
* Create `StructuredOutputsParams` with constraint
* Pass to `SamplingParams(structured_outputs=...)`
* All standard sampling params remain available
* Constraint applied at each token selection step

=== Step 4: Constrained Generation ===
[[step::Principle:vllm-project_vllm_Constrained_Generation]]

Execute generation with the constrained sampling params. The engine applies logit masking at each step to enforce the constraint, only allowing tokens that maintain validity. Generation terminates when a complete valid output is produced.

'''Generation behavior:'''
* Logit processor masks invalid token choices
* Only tokens preserving constraint validity are sampled
* Generation may terminate at constraint completion
* Output guaranteed to match constraint

=== Step 5: Structured Output Parsing ===
[[step::Principle:vllm-project_vllm_Structured_Output_Parsing]]

Parse the generated output according to its constraint type. JSON outputs can be directly deserialized; regex outputs match the specified pattern; choice outputs are one of the allowed values. Post-processing is simplified by guaranteed format compliance.

'''Parsing patterns:'''
* JSON: `json.loads(output.text)` or Pydantic model parsing
* Regex: Output guaranteed to match, direct string use
* Choice: Output is exactly one of the choices
* Grammar: Output follows grammar structure

== Execution Diagram ==
{{#mermaid:graph TD
    A[Constraint Definition] --> B[StructuredOutputsParams Configuration]
    B --> C[SamplingParams Integration]
    C --> D[Constrained Generation]
    D --> E[Structured Output Parsing]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_Constraint_Definition]]
* [[step::Principle:vllm-project_vllm_StructuredOutputsParams_Configuration]]
* [[step::Principle:vllm-project_vllm_Structured_SamplingParams]]
* [[step::Principle:vllm-project_vllm_Constrained_Generation]]
* [[step::Principle:vllm-project_vllm_Structured_Output_Parsing]]
