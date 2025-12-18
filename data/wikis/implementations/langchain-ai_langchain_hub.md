{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Hub|https://smith.langchain.com/hub]]
|-
! Domains
| [[domain::Prompts]], [[domain::Sharing]], [[domain::Registry]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Module providing `push` and `pull` functions for interacting with the LangChain Hub to share and reuse prompts.

=== Description ===

The `hub` module provides an interface to the LangChain Hub (hosted at smith.langchain.com/hub), a registry for sharing and discovering LangChain prompts and chains. The `pull` function retrieves prompts by name, and `push` publishes prompts to the hub. The module supports both the modern LangSmith client and legacy langchainhub package.

=== Usage ===

Use `hub.pull` to load community or your own prompts from the hub, and `hub.push` to share your prompts with others. This enables prompt versioning, sharing, and reuse across projects.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/hub.py libs/langchain/langchain_classic/hub.py]
* '''Lines:''' 1-153

=== Signature ===
<syntaxhighlight lang="python">
def pull(
    owner_repo_commit: str,
    *,
    include_model: bool | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Pull an object from the hub.

    Args:
        owner_repo_commit: Name in format "owner/prompt_name:commit" or "owner/prompt_name".
        include_model: Whether to include model configuration.
        api_url: Custom API URL (optional).
        api_key: API key for authentication.

    Returns:
        The pulled LangChain object (usually a prompt template).
    """


def push(
    repo_full_name: str,
    object: Any,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    parent_commit_hash: str | None = None,
    new_repo_is_public: bool = False,
    new_repo_description: str | None = None,
    readme: str | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Push an object to the hub.

    Args:
        repo_full_name: Name in format "owner/prompt_name" or "prompt_name".
        object: LangChain object to push.
        new_repo_is_public: Whether the prompt should be public.
        new_repo_description: Description for the prompt.
        readme: README content.
        tags: Tags to associate with the prompt.

    Returns:
        URL where the object can be viewed.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Or directly
from langchain_classic.hub import pull, push
</syntaxhighlight>

== I/O Contract ==

=== pull Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| owner_repo_commit || str || Yes || Hub path like "langchain-ai/rag-prompt"
|-
| include_model || bool || No || Include model config in pulled prompt
|-
| api_key || str || No || LangSmith API key
|}

=== push Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| repo_full_name || str || Yes || Target path like "my-username/my-prompt"
|-
| object || Any || Yes || LangChain object to push
|-
| new_repo_is_public || bool || No || Make prompt public (default: False)
|}

== Usage Examples ==

=== Pulling Prompts ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Pull a community prompt
rag_prompt = hub.pull("langchain-ai/rag-prompt")
print(rag_prompt.template)

# Pull a specific version
prompt_v2 = hub.pull("langchain-ai/rag-prompt:abc123")

# Pull your own prompt
my_prompt = hub.pull("my-username/my-custom-prompt")

# Use the pulled prompt
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
chain = rag_prompt | llm
</syntaxhighlight>

=== Pushing Prompts ===
<syntaxhighlight lang="python">
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate

# Create a prompt
my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "{question}"),
])

# Push to hub (requires LANGSMITH_API_KEY)
url = hub.push(
    "my-username/domain-qa-prompt",
    my_prompt,
    new_repo_is_public=True,
    new_repo_description="A QA prompt for domain-specific questions",
    tags=["qa", "domain-specific"],
)
print(f"Prompt available at: {url}")
</syntaxhighlight>

=== With API Key ===
<syntaxhighlight lang="python">
import os

# Set API key via environment variable
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

# Or pass directly
prompt = hub.pull(
    "langchain-ai/rag-prompt",
    api_key="your-api-key",
)
</syntaxhighlight>

=== Browse Hub Prompts ===
<syntaxhighlight lang="python">
# Popular prompts to try:
prompts = [
    "langchain-ai/rag-prompt",           # Basic RAG
    "langchain-ai/chat-langchain-rephrase",  # Query reformulation
    "langchain-ai/retrieval-qa-chat",    # Conversational RAG
]

for prompt_name in prompts:
    prompt = hub.pull(prompt_name)
    print(f"{prompt_name}: {type(prompt).__name__}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

