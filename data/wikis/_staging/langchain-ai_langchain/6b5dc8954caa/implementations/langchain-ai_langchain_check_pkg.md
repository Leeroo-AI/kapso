{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Package_Management]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for validating provider integration package installation before model instantiation, provided by LangChain's chat model factory.

=== Description ===

`_check_pkg` is an internal validation function that verifies required provider packages are installed before attempting to import them. This provides clear, actionable error messages with installation instructions instead of cryptic import errors.

Key behaviors:
* Uses `importlib.util.find_spec` for non-importing check
* Generates `pip install` command in error message
* Handles underscore-to-hyphen conversion for pip package names

=== Usage ===

Use this function (indirectly via `init_chat_model`) to get helpful error messages when provider packages are missing. The pattern can be replicated for custom integrations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L533-537

=== Signature ===
<syntaxhighlight lang="python">
def _check_pkg(pkg: str, *, pkg_kebab: str | None = None) -> None:
    """Check if a package is installed.

    Args:
        pkg: Python package name (e.g., "langchain_openai")
        pkg_kebab: pip package name if different (e.g., "langchain-openai")

    Raises:
        ImportError: If package is not installed, with install instructions
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function, typically not imported directly
# Pattern can be replicated:
from importlib import util


def check_package(pkg: str, pip_name: str | None = None) -> None:
    """Check if package is installed."""
    if not util.find_spec(pkg):
        pip_name = pip_name or pkg.replace("_", "-")
        raise ImportError(f"Install with: pip install {pip_name}")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pkg || str || Yes || Python package name (e.g., "langchain_openai")
|-
| pkg_kebab || str | None || No || pip package name if different from pkg
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || None || Returns None if package is installed
|}

=== Errors ===
{| class="wikitable"
|-
! Error !! Condition !! Message
|-
| ImportError || Package not found || "Unable to import {pkg}. Please install with `pip install -U {pkg_kebab}`"
|}

== Usage Examples ==

=== Internal Usage Pattern ===
<syntaxhighlight lang="python">
# How _check_pkg is used internally
def _init_chat_model_helper(model: str, model_provider: str, **kwargs):
    if model_provider == "openai":
        _check_pkg("langchain_openai")  # Check before import
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, **kwargs)

    if model_provider == "anthropic":
        _check_pkg("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, **kwargs)

    # ... other providers
</syntaxhighlight>

=== Clear Error Messages ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# If langchain-openai is not installed:
try:
    model = init_chat_model("gpt-4o")
except ImportError as e:
    print(e)
    # "Unable to import langchain_openai. Please install with `pip install -U langchain-openai`"
</syntaxhighlight>

=== Custom Package Name ===
<syntaxhighlight lang="python">
# When pip name differs from import name
_check_pkg("langchain_deepseek", pkg_kebab="langchain-deepseek")
# Error: "pip install -U langchain-deepseek"

_check_pkg("langchain_nvidia_ai_endpoints")
# Error: "pip install -U langchain-nvidia-ai-endpoints"
</syntaxhighlight>

=== Replicating the Pattern ===
<syntaxhighlight lang="python">
from importlib import util


def require_package(pkg: str, pip_name: str | None = None) -> None:
    """Require a package to be installed."""
    if not util.find_spec(pkg):
        pip_name = pip_name or pkg.replace("_", "-")
        raise ImportError(
            f"This feature requires '{pip_name}'.\n"
            f"Install with: pip install {pip_name}"
        )


# Usage in custom code
def my_function_needing_numpy():
    require_package("numpy")
    import numpy as np
    # ...
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Provider_Package_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
