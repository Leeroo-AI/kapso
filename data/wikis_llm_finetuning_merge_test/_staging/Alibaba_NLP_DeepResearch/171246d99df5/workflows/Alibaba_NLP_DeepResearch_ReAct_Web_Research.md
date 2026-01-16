# Workflow: ReAct_Web_Research

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Paper|Tongyi DeepResearch|https://arxiv.org/pdf/2510.24701]]
* [[source::Blog|Tech Blog|https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Web_Agents]], [[domain::Information_Retrieval]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

End-to-end process for conducting autonomous deep web research using a ReAct (Reasoning + Acting) agent loop with integrated search, visit, and code execution tools.

=== Description ===

This workflow implements the core inference paradigm of Tongyi DeepResearch, enabling LLMs to autonomously investigate complex questions through iterative web searches and webpage visits. The agent follows a ReAct pattern where it reasons about each step, selects appropriate tools (search, visit, python interpreter, scholar search, file parsing), processes tool responses, and continues until it arrives at a comprehensive answer.

The process leverages:
* **Multi-query search**: Batched web searches via Google Serper API with locale-aware results
* **Intelligent page summarization**: Webpage content extraction via Jina AI with LLM-powered summarization
* **Sandboxed code execution**: Python interpreter for calculations and data analysis
* **Token management**: Automatic context window management with forced answer generation on limit

=== Usage ===

Execute this workflow when you need to:
* Answer complex, multi-faceted research questions requiring web information synthesis
* Investigate topics that require visiting multiple sources and cross-validating information
* Perform deep research tasks that benefit from iterative search refinement
* Process questions that may require code execution for calculations or data analysis

The input is a question (with optional file attachments), and the output is a structured answer with supporting evidence gathered from web sources.

== Execution Steps ==

=== Step 1: Environment Configuration ===

Configure the required API keys and model settings. This includes setting up access to the web search service (Serper API), webpage reader service (Jina AI), the LLM inference endpoint (vLLM or OpenRouter), and optional services like the Python sandbox.

'''Key considerations:'''
* Ensure SERPER_KEY_ID is set for web search functionality
* Configure JINA_API_KEYS for webpage content extraction
* Set MODEL_PATH to point to the Tongyi-DeepResearch model weights
* Configure API_KEY and API_BASE for the summarization LLM

=== Step 2: Data Preparation ===

Prepare the evaluation data in JSONL or JSON format. Each item must contain a question field, and optionally an answer field for evaluation purposes. If the question references uploaded files, place them in the designated file corpus directory.

'''Data format requirements:'''
* JSONL format: One JSON object per line with "question" and "answer" keys
* JSON format: Array of objects with "question" and "answer" keys
* File references: Prepend filename to question field (e.g., "(Uploaded 1 file: ['report.pdf'])")

=== Step 3: Agent Initialization ===

Initialize the MultiTurnReactAgent with the configured LLM settings and tool list. The agent sets up the system prompt with current date context and prepares the tool registry containing Search, Visit, PythonInterpreter, Scholar, and FileParser tools.

'''What happens:'''
* System prompt is constructed with tool definitions in XML format
* Tool instances are created and registered in the tool map
* LLM generation configuration is loaded (temperature, top_p, presence_penalty)

=== Step 4: ReAct Loop Execution ===

Execute the main reasoning loop where the agent iteratively thinks, selects tools, and processes responses. Each iteration involves calling the LLM server, parsing tool calls from the response, executing the selected tool, and appending results to the conversation.

'''Loop mechanics:'''
* Agent generates response with <think> reasoning and optional <tool_call>
* Tool calls are parsed (JSON format for most tools, special format for Python)
* Tool execution results are wrapped in <tool_response> tags
* Loop continues until <answer> tag is produced or limits are reached

=== Step 5: Tool Execution ===

Execute the selected tool based on the agent's decision. Each tool has specific behavior:

'''Tool behaviors:'''
* **search**: Sends queries to Google Serper API, returns top 10 results with snippets
* **visit**: Fetches webpage via Jina AI, uses LLM to extract goal-relevant information
* **PythonInterpreter**: Executes code in sandboxed environment, returns stdout
* **google_scholar**: Searches academic publications via Google Scholar
* **parse_file**: Extracts content from uploaded documents (PDF, DOCX, etc.)

=== Step 6: Context Management ===

Monitor and manage the conversation context length. If the token count exceeds the maximum limit (110K tokens), force the agent to stop tool calls and generate a final answer based on accumulated information.

'''Safeguards:'''
* Token counting using model tokenizer
* Forced answer prompt when context limit approached
* Time limit enforcement (150 minutes maximum)
* Maximum LLM call limit (configurable, default 100)

=== Step 7: Answer Extraction ===

Extract the final answer from the agent's response. The answer must be enclosed in <answer></answer> tags. Record the termination reason (answer found, token limit, time limit, or call limit exceeded).

'''Output structure:'''
* question: Original input question
* answer: Ground truth (if provided)
* prediction: Extracted answer from agent
* messages: Full conversation history
* termination: Reason for stopping

== Execution Diagram ==

{{#mermaid:graph TD
    A[Environment Configuration] --> B[Data Preparation]
    B --> C[Agent Initialization]
    C --> D[ReAct Loop Execution]
    D --> E{Tool Call?}
    E -->|Yes| F[Tool Execution]
    F --> G[Context Management]
    G --> H{Limits OK?}
    H -->|Yes| D
    H -->|No| I[Force Answer]
    E -->|No - Answer| J[Answer Extraction]
    I --> J
    J --> K[Output Result]
}}

== GitHub URL ==

[[github_url::https://github.com/leeroo-coder/workflow-alibaba-nlp-deepresearch-react-web-research]]
