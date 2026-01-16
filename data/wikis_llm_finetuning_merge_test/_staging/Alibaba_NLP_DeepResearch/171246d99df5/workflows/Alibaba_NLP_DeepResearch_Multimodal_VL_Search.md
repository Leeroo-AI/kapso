# Workflow: Multimodal_VL_Search

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Paper|WebWatcher|https://arxiv.org/pdf/2508.05748]]
* [[source::Doc|Qwen Agent|https://github.com/QwenLM/Qwen-Agent]]
|-
! Domains
| [[domain::LLMs]], [[domain::Vision_Language]], [[domain::Web_Agents]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

End-to-end process for answering visual questions through autonomous multimodal web research using vision-language models with reverse image search and text search capabilities.

=== Description ===

This workflow implements the WebWatcher agent paradigm for handling multimodal queries that combine images and text. The agent can perform reverse image searches to identify visual content, conduct text-based web searches for additional context, visit webpages to gather information, and execute Python code for analysis. The process enables answering complex visual questions that require both image understanding and web information retrieval.

Key capabilities:
* **Reverse image search**: Google Lens integration via SerpAPI for image-based queries
* **Vision-language understanding**: Multi-turn reasoning with image context preservation
* **Tool orchestration**: Coordinated use of VLSearchImage, web_search, visit, and code_interpreter
* **Sub-question decomposition**: Breaking complex visual queries into manageable research steps

=== Usage ===

Execute this workflow when you need to:
* Identify objects, landmarks, or entities in images and retrieve detailed information
* Answer questions that require understanding both visual content and web context
* Investigate image provenance or find related visual content online
* Perform visual QA tasks requiring external knowledge augmentation

The input is a question paired with one or more images, and the output is a comprehensive answer derived from both visual analysis and web research.

== Execution Steps ==

=== Step 1: Image Processing ===

Process and prepare the input image(s) for the vision-language model. Images are resized and formatted according to the model's pixel requirements, with careful handling of aspect ratios and resolution constraints.

'''Key considerations:'''
* Maximum pixels: 1024 x 28 x 28 for optimal processing
* Minimum pixels: 256 x 28 x 28 to ensure adequate detail
* Support for multiple sub-images within a single input
* Image URL generation for reverse search compatibility

=== Step 2: Prompt Construction ===

Construct the system prompt with tool definitions and the user prompt combining the question with image references. The prompt instructs the agent to decompose complex visual questions, describe image content in detail, and use tools iteratively for validation.

'''Prompt structure:'''
* System prompt with agent principles (decomposition, description, turn limits)
* Tool definitions for web_search, VLSearchImage, visit, and code_interpreter
* Format specification with think/tool_call/tool_response/answer tags
* Input question and image URL placeholders

=== Step 3: Multi_turn Agent Loop ===

Execute the main agent loop using the Qwen vision-language model. Each turn involves generating reasoning, selecting tools, and processing responses until an answer is reached or the maximum step limit (12 turns) is hit.

'''Loop mechanics:'''
* Agent receives conversation history with image context
* Generates <think> reasoning about next action
* Produces <tool_call> with selected tool and parameters
* Tool response appended to conversation
* Continues until <answer> tag generated

=== Step 4: Reverse Image Search ===

When the agent invokes VLSearchImage, perform a Google reverse image search using the SerpAPI integration. Upload the image to cloud storage if needed, then retrieve matching images and their associated metadata.

'''Search process:'''
* Image URL passed to Google Reverse Image API
* Top results returned with snippets, source URLs, and thumbnails
* Results cached to avoid redundant API calls
* Incremental search strategy for efficient result gathering

=== Step 5: Text Web Search ===

Execute text-based web searches when the agent needs to supplement visual findings with textual information. Multiple queries can be processed in parallel to gather comprehensive context.

'''Search capabilities:'''
* Multi-query support for parallel information gathering
* Locale-aware search (Chinese/English detection)
* Result snippets with source attribution
* Integration with webpage visit for deeper exploration

=== Step 6: Webpage Visitation ===

Visit specific webpages identified through search results to extract detailed information relevant to the user's goal. Content is fetched and summarized using an LLM to extract evidence and key findings.

'''Extraction workflow:'''
* URL fetched via Jina AI reader service
* Content truncated to fit context limits (95K tokens max)
* LLM extracts rationale, evidence, and summary
* Structured output returned to agent

=== Step 7: Code Execution ===

Execute Python code when the agent needs to perform calculations, data analysis, or content processing. Code runs in a sandboxed environment with stdout captured for result communication.

'''Sandbox features:'''
* Remote execution via SandboxFusion endpoint
* HTTP-based code submission and result retrieval
* Support for common data analysis libraries
* Output capture and error handling

=== Step 8: Answer Generation ===

Synthesize all gathered information into a final answer. The agent reviews visual analysis, search results, webpage content, and any computation outputs to formulate a comprehensive response.

'''Answer requirements:'''
* Enclosed in <answer></answer> tags
* Should be provided within 10 turns
* Incorporates cross-validated information from multiple sources
* Addresses the original visual question directly

== Execution Diagram ==

{{#mermaid:graph TD
    A[Image Processing] --> B[Prompt Construction]
    B --> C[Multi-turn Agent Loop]
    C --> D{Tool Selection}
    D -->|VLSearchImage| E[Reverse Image Search]
    D -->|web_search| F[Text Web Search]
    D -->|visit| G[Webpage Visitation]
    D -->|code_interpreter| H[Code Execution]
    D -->|answer| I[Answer Generation]
    E --> C
    F --> C
    G --> C
    H --> C
    I --> J[Output Result]
}}

== GitHub URL ==

[[github_url::https://github.com/leeroo-coder/workflow-alibaba-nlp-deepresearch-multimodal-vl-search]]
