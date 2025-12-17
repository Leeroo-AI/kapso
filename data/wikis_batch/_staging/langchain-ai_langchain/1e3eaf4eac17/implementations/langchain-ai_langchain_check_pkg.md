{{Infobox Implementation
| name = check_pkg
| domain = Dependency Management
| sources = libs/langchain_v1/langchain/chat_models/base.py:L533-537
| last_updated = 2025-12-17
}}

== Overview ==

The <code>_check_pkg</code> function is the implementation of the Provider Package Verification principle. It verifies that a required Python package is installed and importable, raising an informative <code>ImportError</code> with installation instructions if the package is missing.

== Description ==

<code>_check_pkg</code> performs lightweight verification of package availability using Python's <code>importlib.util.find_spec()</code> function. This approach checks whether a package can be imported without actually triggering the import, making it efficient for validation purposes.

The function handles a common mismatch between Python import names and PyPI package names. Import names typically use underscores (e.g., <code>langchain_openai</code>), while PyPI package names use hyphens (e.g., <code>langchain-openai</code>). The <code>pkg_kebab</code> parameter allows callers to specify the correct PyPI name when it differs from the import name.

When a package is not found, the function raises an <code>ImportError</code> with a message that:
1. Identifies the missing package by import name
2. Provides the exact pip installation command
3. Uses the <code>-U</code> flag to ensure the latest version is installed

== Code Reference ==

<syntaxhighlight lang="python">
def _check_pkg(pkg: str, *, pkg_kebab: str | None = None) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab if pkg_kebab is not None else pkg.replace("_", "-")
        msg = f"Unable to import {pkg}. Please install with `pip install -U {pkg_kebab}`"
        raise ImportError(msg)
</syntaxhighlight>

Source: <code>libs/langchain_v1/langchain/chat_models/base.py</code> lines 533-537

=== Dependencies ===

<syntaxhighlight lang="python">
from importlib import util
</syntaxhighlight>

The function uses <code>importlib.util.find_spec()</code> from Python's standard library, which returns a module spec if the package can be imported, or <code>None</code> if it cannot be found.

== I/O Contract ==

=== Input Parameters ===

; <code>pkg</code> : <code>str</code>
: The Python package import name to verify (e.g., "langchain_openai", "langchain_anthropic")

; <code>pkg_kebab</code> : <code>str | None</code> (keyword-only)
: Optional PyPI package name if it differs from the import name. If not specified, defaults to converting underscores in <code>pkg</code> to hyphens

=== Return Value ===

; <code>None</code>
: The function returns <code>None</code> if the package is available. It only returns normally when verification succeeds.

=== Exceptions ===

; <code>ImportError</code>
: Raised when the specified package cannot be found. The error message includes the package import name and installation command.

=== Side Effects ===

The function has no side effects beyond raising an exception. It does not modify state, import packages, or perform I/O operations. The <code>find_spec()</code> call may read from the filesystem to locate package metadata, but this is a read-only operation.

== Usage Examples ==

=== Example 1: Verifying Standard Package ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _check_pkg

# Verify OpenAI package is installed
try:
    _check_pkg("langchain_openai")
    print("Package is available")
except ImportError as e:
    print(f"Package missing: {e}")
    # "Unable to import langchain_openai.
    #  Please install with `pip install -U langchain-openai`"
</syntaxhighlight>

=== Example 2: Package with Matching Import and PyPI Names ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _check_pkg

# When import name matches PyPI name (after underscore conversion)
_check_pkg("langchain_anthropic")
# If missing, suggests: pip install -U langchain-anthropic

_check_pkg("langchain_google_vertexai")
# If missing, suggests: pip install -U langchain-google-vertexai
</syntaxhighlight>

=== Example 3: Package with Different PyPI Name ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _check_pkg

# DeepSeek has non-standard naming
_check_pkg("langchain_deepseek", pkg_kebab="langchain-deepseek")
# Explicitly specifies the PyPI package name

# If the default conversion was used:
# _check_pkg("langchain_deepseek")
# Would suggest: pip install -U langchain-deepseek (same result in this case)
</syntaxhighlight>

=== Example 4: Integration in Model Initialization ===

<syntaxhighlight lang="python">
# From _init_chat_model_helper implementation
def initialize_openai_model(model: str, **kwargs):
    # Verify package before attempting import
    _check_pkg("langchain_openai")

    # Only import if package verification succeeds
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, **kwargs)
</syntaxhighlight>

=== Example 5: Handling Special Cases ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _check_pkg

# Nvidia package has long name
_check_pkg("langchain_nvidia_ai_endpoints")
# If missing, suggests: pip install -U langchain-nvidia-ai-endpoints

# Azure AI package
_check_pkg("langchain_azure_ai")
# If missing, suggests: pip install -U langchain-azure-ai

# Community package (legacy)
_check_pkg("langchain_community")
# If missing, suggests: pip install -U langchain-community
</syntaxhighlight>

=== Example 6: Error Handling in Application Code ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _check_pkg

def safe_initialize_model(provider: str):
    """Initialize model with graceful error handling."""
    package_map = {
        "openai": "langchain_openai",
        "anthropic": "langchain_anthropic",
        "ollama": "langchain_ollama",
    }

    pkg = package_map.get(provider)
    if not pkg:
        raise ValueError(f"Unknown provider: {provider}")

    try:
        _check_pkg(pkg)
    except ImportError as e:
        # Log error and provide guidance
        print(f"Provider {provider} requires additional installation:")
        print(f"  {e}")
        return None

    # Proceed with initialization...
    return initialize_provider(provider)
</syntaxhighlight>

== Implementation Details ==

=== Package Name Conversion ===

The default conversion strategy (<code>pkg.replace("_", "-")</code>) handles the common Python naming convention where:
* Import names use underscores: <code>langchain_openai</code>
* PyPI names use hyphens: <code>langchain-openai</code>

This conversion is applied automatically unless <code>pkg_kebab</code> is explicitly provided.

=== Using find_spec vs. Import Attempts ===

The implementation uses <code>util.find_spec()</code> rather than attempting an actual import because:

1. '''Performance''': <code>find_spec()</code> only checks if a module can be imported without executing module-level code
2. '''Safety''': Avoids side effects from module imports (e.g., logging configuration, global state)
3. '''Clarity''': The check is explicitly for availability, not for using the package
4. '''Efficiency''': Multiple checks can be performed without the overhead of full imports

=== Error Message Design ===

The error message format follows best practices:
* Clearly states the problem ("Unable to import X")
* Provides the exact command to resolve it (<code>pip install -U package</code>)
* Uses the <code>-U</code> flag to encourage getting the latest compatible version
* Uses the correct PyPI package name that pip will recognize

== Related Pages ==

=== Principles ===
* [[langchain-ai_langchain_Provider_Package_Verification|Provider Package Verification]] - Principle implemented by this function

=== Related Implementations ===
* [[langchain-ai_langchain_parse_model|parse_model]] - Determines which provider (and thus package) to verify
* [[langchain-ai_langchain_init_chat_model_helper|init_chat_model_helper]] - Uses this function to verify packages before instantiation

=== Workflows ===
* [[langchain-ai_langchain_Chat_Model_Initialization|Chat Model Initialization]] - Overall workflow utilizing package verification

[[Category:Implementations]]
[[Category:Dependency Management]]
[[Category:Error Handling]]
[[Category:LangChain]]
