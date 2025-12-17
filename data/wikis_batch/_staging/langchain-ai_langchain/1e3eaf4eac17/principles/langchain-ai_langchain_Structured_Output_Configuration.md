# Structured Output Configuration

**Sources:**
- `libs/langchain_v1/langchain/agents/structured_output.py:L1-443`
- LangChain Documentation: Structured Output
- OpenAI Structured Outputs API

**Domains:** LLM Output Parsing, Schema Validation, Agent Interfaces

**Last Updated:** 2025-12-17

---

## Overview

Structured Output Configuration is the principle of constraining LLM responses to conform to predefined schemas (Pydantic models, dataclasses, TypedDicts), enabling reliable parsing and type-safe integration with downstream systems. This principle bridges the gap between free-form natural language generation and structured data requirements.

## Description

The principle addresses a fundamental challenge in LLM applications: while language models excel at generating natural language, many applications require specific data structures (JSON objects, database records, API requests). Structured Output Configuration establishes mechanisms to:

1. **Define output schemas** - Specify the exact structure LLM outputs must conform to
2. **Enforce schemas** - Use either tool calling or native provider features to constrain generation
3. **Validate outputs** - Parse and validate LLM responses against the schema
4. **Handle errors** - Provide retry mechanisms when outputs fail validation
5. **Strategy selection** - Choose between tool-based and provider-native approaches

This principle is critical for building reliable agent systems where outputs drive subsequent actions - malformed data can cascade into system failures.

### Key Architectural Decisions

**Two Strategy Model**
The system provides two distinct strategies for structured output:

1. **ToolStrategy** - Uses the model's tool-calling capability, treating the schema as an artificial tool. Works with any model that supports tool calling.
2. **ProviderStrategy** - Leverages provider-native structured output features (e.g., OpenAI's JSON mode). Offers stronger guarantees but limited provider support.

**Auto-Detection with AutoStrategy**
An `AutoStrategy` wrapper enables automatic selection between Tool and Provider strategies based on model capabilities, simplifying the API for users while optimizing for the best available approach.

**Union Type Support**
When using `ToolStrategy`, Union types (e.g., `ResponseA | ResponseB`) are automatically decomposed into multiple tools, allowing the LLM to select the appropriate response structure.

**Error Handling Configuration**
Structured output can be configured with retry behavior via `handle_errors` parameter, enabling automatic recovery from validation failures by feeding errors back to the LLM.

**Integration with Agent State**
Parsed structured responses are stored in `state["structured_response"]`, making them accessible to middleware and application code while keeping them separate from the conversational message history.

## Theoretical Basis

This principle draws from several foundational concepts:

**Schema Theory**
From database systems and data modeling - schemas define data structure constraints that enable validation, type safety, and interoperability.

**Parser Combinators**
The approach mirrors parser combinator libraries where complex parsers are built from simpler ones, with structured output strategies being composable parsing strategies.

**Contract Programming**
The schema acts as a contract between the LLM (producer) and the application (consumer), with validation enforcing the contract at runtime.

**Constraint-Based Generation**
Provider strategies implement constraint-based generation where the LLM's output space is restricted to valid schema instances during generation (not just post-processing).

**Error Recovery Patterns**
The retry mechanism implements a feedback loop where validation errors guide the LLM toward producing valid outputs, similar to reinforcement learning from error signals.

## Usage

### When to Apply This Principle

Apply Structured Output Configuration when:

- Building agents that need to produce database records, API requests, or structured data
- Implementing multi-step workflows where outputs become inputs to other systems
- Creating chatbots that need to extract specific information from conversations
- Generating structured knowledge from unstructured text
- Building reliable agent interfaces with predictable output formats

### When to Use Alternative Approaches

Consider alternatives when:

- **Simple extraction**: For extracting simple values (dates, numbers, yes/no), regex or simple parsing may suffice
- **Flexible outputs**: If output structure varies significantly by context, structured output constraints may be too rigid
- **Maximum creativity**: For creative writing or open-ended generation, schema constraints inhibit the model
- **Legacy systems**: If working with older models without tool calling or structured output support

### Anti-Patterns to Avoid

1. **Overly complex schemas**: Schemas with deep nesting or many optional fields confuse LLMs
2. **Ambiguous field names**: Field names that don't clearly indicate their purpose lead to incorrect values
3. **Missing descriptions**: Pydantic field descriptions are crucial for LLMs to understand what to generate
4. **Ignoring validation errors**: Disabling error handling without understanding failure modes
5. **Schema-application mismatch**: Defining schemas that don't match actual downstream data requirements
6. **Unnecessary structure**: Using structured output when free-form text would suffice

### Best Practices

**Write Clear Field Descriptions**
Every field in your schema should have a description explaining what it represents and any constraints.

**Keep Schemas Flat**
Prefer flat structures over deep nesting. Deeply nested objects increase complexity and error rates.

**Use Appropriate Types**
Match field types to actual data (use enums for fixed choices, use Optional for truly optional fields).

**Test with Real Data**
Validate that your schema works with representative inputs before deploying.

**Handle Validation Errors**
Configure appropriate error handling strategies based on whether outputs are critical or optional.

## Related Pages

**Implementation:**
- [[implemented_by::Implementation:langchain-ai_langchain_ResponseFormat_strategies]] - Implementation of strategy classes

**Related Principles:**
- [[langchain-ai_langchain_Tool_Definition]] - Tool-based strategy builds on tool calling
- [[langchain-ai_langchain_Agent_Graph_Construction]] - Structured output integrates into agent graphs

**Used In Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Structured output configuration is Step 4

**Related Implementations:**
- [[langchain-ai_langchain_ToolStrategy]] - Tool-based structured output
- [[langchain-ai_langchain_ProviderStrategy]] - Provider-native structured output
- [[langchain-ai_langchain_OutputToolBinding]] - Internal tool binding for schemas
- [[langchain-ai_langchain_parse_with_schema]] - Schema validation logic

**Environment:**
- [[langchain-ai_langchain_Python]] - Python type system and Pydantic integration
