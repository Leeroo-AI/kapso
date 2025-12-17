= LangChain Hub Interface Implementation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Source File || <code>/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/hub.py</code>
|-
| Module Name || <code>hub</code>
|-
| Package || <code>langchain_classic</code>
|-
| Lines of Code || 153
|-
| Status || Approved
|}

== Overview ==
The <code>hub</code> module provides a Python interface for interacting with the [https://smith.langchain.com/hub LangChain Hub], a centralized repository for sharing and managing prompts, chains, and other LangChain objects. The module abstracts away the underlying client implementation details, supporting both the modern LangSmith client and the legacy <code>langchainhub</code> client.

Key capabilities include:
* '''Pushing''' LangChain objects (prompts, chains) to the hub for sharing and version control
* '''Pulling''' LangChain objects from the hub by name/commit hash
* '''Automatic client selection''' based on available dependencies
* '''Metadata enrichment''' for pulled prompts (owner, repo, commit hash)

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/hub.py
Lines: 1-153
</syntaxhighlight>

=== Module Functions ===
<syntaxhighlight lang="python">
def _get_client(
    api_key: str | None = None,
    api_url: str | None = None,
) -> Any:
    """Get a client for interacting with the LangChain Hub."""

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
    """Push an object to the hub and returns the URL it can be viewed at."""

def pull(
    owner_repo_commit: str,
    *,
    include_model: bool | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Pull an object from the hub and returns it as a LangChain object."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Or import specific functions
from langchain_classic.hub import push, pull
</syntaxhighlight>

== I/O Contract ==

=== <code>_get_client</code> Function ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>api_key</code> || <code>str &#124; None</code> || No || <code>None</code> || API key to authenticate with the LangChain Hub API
|-
| <code>api_url</code> || <code>str &#124; None</code> || No || <code>None</code> || URL of the LangChain Hub API
|}

'''Returns:''' Client instance for interacting with the hub (LangSmith or LangChainHub client)

'''Raises:''' <code>ImportError</code> if neither <code>langsmith</code> nor <code>langchainhub</code> can be imported

=== <code>push</code> Function ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>repo_full_name</code> || <code>str</code> || Yes || N/A || Full name in format <code>owner/prompt_name</code> or just <code>prompt_name</code>
|-
| <code>object</code> || <code>Any</code> || Yes || N/A || The LangChain object to serialize and push to the hub
|-
| <code>api_url</code> || <code>str &#124; None</code> || No || <code>None</code> || URL of the LangChain Hub API
|-
| <code>api_key</code> || <code>str &#124; None</code> || No || <code>None</code> || API key to authenticate with the hub
|-
| <code>parent_commit_hash</code> || <code>str &#124; None</code> || No || <code>None</code> || Commit hash of the parent commit (defaults to latest)
|-
| <code>new_repo_is_public</code> || <code>bool</code> || No || <code>False</code> || Whether the prompt should be public
|-
| <code>new_repo_description</code> || <code>str &#124; None</code> || No || <code>None</code> || Description of the prompt
|-
| <code>readme</code> || <code>str &#124; None</code> || No || <code>None</code> || README content for the repository
|-
| <code>tags</code> || <code>Sequence[str] &#124; None</code> || No || <code>None</code> || Tags to associate with the prompt
|}

'''Returns:''' <code>str</code> - URL where the pushed object can be viewed in a browser

=== <code>pull</code> Function ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>owner_repo_commit</code> || <code>str</code> || Yes || N/A || Format: <code>owner/prompt_name:commit_hash</code>, <code>owner/prompt_name</code>, or just <code>prompt_name</code>
|-
| <code>include_model</code> || <code>bool &#124; None</code> || No || <code>None</code> || Whether to include model configuration in the pulled prompt
|-
| <code>api_url</code> || <code>str &#124; None</code> || No || <code>None</code> || URL of the LangChain Hub API
|-
| <code>api_key</code> || <code>str &#124; None</code> || No || <code>None</code> || API key to authenticate with the hub
|}

'''Returns:''' <code>Any</code> - The pulled LangChain object (typically a prompt or chain)

== Implementation Details ==

=== Client Selection Logic ===
The <code>_get_client</code> function implements a fallback strategy:

1. '''Try LangSmith first''': Attempts to import <code>langsmith.Client</code>
   * Checks if the client has <code>push_prompt</code> and <code>pull_prompt</code> methods
   * If yes, returns the LangSmith client
   * If no, falls back to LangChainHub client
2. '''Fallback to LangChainHub''': If LangSmith import fails, tries <code>langchainhub.Client</code>
3. '''Raise ImportError''': If both imports fail, raises an error with installation instructions

=== Push Operation ===
The <code>push</code> function:

1. Gets the appropriate client via <code>_get_client</code>
2. Checks if client has <code>push_prompt</code> method (LangSmith)
   * If yes: Calls <code>client.push_prompt()</code> with all parameters
3. Otherwise (LangChainHub client):
   * Serializes the object to JSON using <code>dumps()</code>
   * Calls <code>client.push()</code> with serialized manifest

=== Pull Operation ===
The <code>pull</code> function implements version-specific logic:

1. Gets the appropriate client via <code>_get_client</code>
2. '''LangSmith client path''': If <code>client.pull_prompt</code> exists, uses it directly
3. '''LangChainHub >= 0.1.15 path''': If <code>client.pull_repo</code> exists:
   * Calls <code>pull_repo</code> to get repository metadata
   * Deserializes the manifest using <code>loads()</code>
   * For <code>BasePromptTemplate</code> objects, enriches metadata with:
     * <code>lc_hub_owner</code>: Repository owner
     * <code>lc_hub_repo</code>: Repository name
     * <code>lc_hub_commit_hash</code>: Commit hash
4. '''LangChainHub < 0.1.15 path''': Falls back to simple <code>client.pull()</code> and deserializes

=== Serialization/Deserialization ===
* Uses <code>langchain_core.load.dump.dumps</code> for serialization
* Uses <code>langchain_core.load.load.loads</code> for deserialization
* Handles JSON conversion for compatibility with different client versions

== Usage Examples ==

=== Pulling a Prompt from the Hub ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Pull a prompt by owner/name
prompt = hub.pull("hwchase17/openai-functions-agent")

# Pull a specific commit
prompt = hub.pull("hwchase17/openai-functions-agent:abc123def")

# Pull your own prompt (no owner prefix needed)
prompt = hub.pull("my-custom-prompt")

# Use the pulled prompt
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
chain = prompt | llm
result = chain.invoke({"input": "What is LangChain?"})
</syntaxhighlight>

=== Pulling with Model Configuration ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Include model configuration (if available)
prompt = hub.pull("hwchase17/react", include_model=True)

# Check if model metadata was included
if hasattr(prompt, "metadata") and prompt.metadata:
    print(f"Hub owner: {prompt.metadata.get('lc_hub_owner')}")
    print(f"Hub repo: {prompt.metadata.get('lc_hub_repo')}")
    print(f"Commit hash: {prompt.metadata.get('lc_hub_commit_hash')}")
</syntaxhighlight>

=== Pushing a Prompt to the Hub ===
<syntaxhighlight lang="python">
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate

# Create a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Push to the hub (creates a private prompt by default)
url = hub.push("my-username/my-helpful-prompt", prompt)
print(f"Prompt available at: {url}")
</syntaxhighlight>

=== Pushing with Metadata ===
<syntaxhighlight lang="python">
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a blog post about {topic}"
)

# Push with full metadata
url = hub.push(
    "my-username/blog-post-prompt",
    prompt,
    new_repo_is_public=True,
    new_repo_description="A prompt for generating blog posts",
    readme="# Blog Post Prompt\n\nUse this to generate blog posts on any topic.",
    tags=["blog", "content-generation", "writing"]
)
</syntaxhighlight>

=== Versioning with Parent Commits ===
<syntaxhighlight lang="python">
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate

# First version
prompt_v1 = PromptTemplate(
    input_variables=["topic"],
    template="Write about {topic}"
)
url = hub.push("my-username/my-prompt", prompt_v1)

# Update with parent commit (versioning)
prompt_v2 = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed article about {topic}"
)

url = hub.push(
    "my-username/my-prompt",
    prompt_v2,
    parent_commit_hash="previous_commit_hash_here"
)
</syntaxhighlight>

=== Custom API Configuration ===
<syntaxhighlight lang="python">
from langchain_classic import hub

# Use custom API endpoint and key
prompt = hub.pull(
    "my-prompt",
    api_url="https://custom-hub.example.com",
    api_key="my-custom-api-key"
)

# Push to custom hub
url = hub.push(
    "my-username/my-prompt",
    prompt,
    api_url="https://custom-hub.example.com",
    api_key="my-custom-api-key"
)
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from langchain_classic import hub

try:
    prompt = hub.pull("owner/nonexistent-prompt")
except Exception as e:
    print(f"Failed to pull prompt: {e}")

try:
    url = hub.push("invalid/repo/name", prompt)
except Exception as e:
    print(f"Failed to push prompt: {e}")
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_BasePromptTemplate|BasePromptTemplate]] - Base class for prompts
* [[langchain-ai_langchain_ChatPromptTemplate|ChatPromptTemplate]] - Chat-based prompts
* [[langchain-ai_langchain_PromptTemplate|PromptTemplate]] - String-based prompts
* [[langchain-ai_langchain_Chain|Chain]] - Base chain class that can be shared via hub

== Notes ==
* '''Installation Required''': Requires either <code>langsmith</code> (recommended) or <code>langchainhub</code> package
* '''Authentication''': API key can be provided via parameter or environment variable (typically <code>LANGCHAIN_API_KEY</code>)
* '''Versioning''': The hub supports full version control with commit hashes
* '''Public vs Private''': Prompts are private by default; set <code>new_repo_is_public=True</code> to share publicly
* '''Naming Convention''': Use <code>owner/prompt-name</code> format; owner can be omitted for your own prompts
* '''Metadata Enrichment''': Pulled prompts include hub metadata in their <code>metadata</code> attribute
* '''Client Priority''': LangSmith client is preferred over legacy LangChainHub client
* '''Backward Compatibility''': Supports multiple versions of the underlying client libraries

== See Also ==
* [https://smith.langchain.com/hub LangChain Hub Website]
* [https://docs.langchain.com/docs/langsmith/hub LangChain Hub Documentation]
* [https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/hub.py Source Code on GitHub]
* [https://pypi.org/project/langsmith/ LangSmith Package]
