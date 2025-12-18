{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Integrations|https://docs.langchain.com/oss/python/integrations/providers]]
* [[source::Doc|Python importlib|https://docs.python.org/3/library/importlib.html]]
|-
! Domains
| [[domain::LLM]], [[domain::Package_Management]], [[domain::Error_Handling]], [[domain::Developer_Experience]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Validation pattern that checks for required provider packages before use and provides actionable installation instructions.

=== Description ===

Provider Package Validation ensures required integration packages are installed before attempting to use them. This pattern:
* Prevents cryptic import errors
* Provides clear `pip install` instructions
* Validates early (at initialization, not during use)
* Uses non-importing checks (`find_spec`) to avoid side effects

This is critical for LangChain's modular architecture where provider integrations are separate packages (`langchain-openai`, `langchain-anthropic`, etc.).

=== Usage ===

Apply Provider Package Validation when:
* Building plugin/integration systems
* Creating optional dependency features
* Designing modular architectures
* Improving developer experience for missing dependencies

== Theoretical Basis ==

Provider Package Validation implements **Fail-Fast** and **Helpful Error Messages** patterns.

'''1. The Validation Problem'''

<syntaxhighlight lang="python">
# Without validation - cryptic errors
def bad_init(provider):
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # ImportError if not installed
        return ChatOpenAI()
    # Error: "ModuleNotFoundError: No module named 'langchain_openai'"
    # Not helpful - what should user do?

# With validation - clear guidance
def good_init(provider):
    if provider == "openai":
        check_pkg("langchain_openai")  # Validates BEFORE import
        from langchain_openai import ChatOpenAI
        return ChatOpenAI()
    # Error: "Unable to import langchain_openai. Please install with `pip install -U langchain-openai`"
    # Actionable!
</syntaxhighlight>

'''2. Non-Importing Validation'''

<syntaxhighlight lang="python">
from importlib import util

def check_pkg(pkg: str) -> None:
    """Check package without importing it."""
    # find_spec checks if module exists without importing
    # This avoids:
    # - Side effects from __init__.py
    # - Slow imports
    # - Circular import issues
    spec = util.find_spec(pkg)
    if spec is None:
        raise ImportError(f"Package '{pkg}' not found")
</syntaxhighlight>

'''3. Package Name Translation'''

<syntaxhighlight lang="python">
# Python import names vs pip package names differ
PACKAGE_NAME_MAP = {
    # import_name -> pip_name
    "langchain_openai": "langchain-openai",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_google_vertexai": "langchain-google-vertexai",
    "langchain_nvidia_ai_endpoints": "langchain-nvidia-ai-endpoints",
}

def get_pip_name(import_name: str) -> str:
    """Convert import name to pip package name."""
    if import_name in PACKAGE_NAME_MAP:
        return PACKAGE_NAME_MAP[import_name]
    # Default: replace underscores with hyphens
    return import_name.replace("_", "-")
</syntaxhighlight>

'''4. Error Message Design'''

<syntaxhighlight lang="python">
def create_install_error(pkg: str, pip_name: str, feature: str = None) -> ImportError:
    """Create helpful ImportError with install instructions."""
    message_parts = [
        f"Unable to import '{pkg}'.",
    ]

    if feature:
        message_parts.append(f"This is required for {feature}.")

    message_parts.append(f"Please install with:\n  pip install -U {pip_name}")

    # Optional: Add documentation link
    message_parts.append(f"\nFor more info: https://docs.langchain.com/oss/python/integrations/providers")

    return ImportError("\n".join(message_parts))
</syntaxhighlight>

'''5. Validation Timing'''

<syntaxhighlight lang="python">
# Validate at these points:

# 1. Function entry (fail-fast)
def init_model(provider, model):
    validate_provider_installed(provider)  # Fails immediately
    # ... rest of initialization

# 2. Lazy validation (on first use)
class LazyModel:
    def __init__(self, provider):
        self.provider = provider
        self._model = None

    def invoke(self, *args):
        if self._model is None:
            validate_provider_installed(self.provider)  # Fails on first call
            self._model = create_model(self.provider)
        return self._model.invoke(*args)

# 3. Configuration validation (at config parse)
def load_config(config_path):
    config = parse_config(config_path)
    for provider in config.get("providers", []):
        validate_provider_installed(provider)  # Validates all providers upfront
    return config
</syntaxhighlight>

'''6. Optional Dependencies Pattern'''

<syntaxhighlight lang="python">
# For features that enhance but aren't required
def optional_feature():
    """Feature with optional dependency."""
    try:
        check_pkg("optional_package")
        import optional_package
        return optional_package.do_thing()
    except ImportError:
        # Graceful degradation
        warnings.warn("optional_package not installed, using fallback")
        return fallback_implementation()
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_check_pkg]]

=== Used By Workflows ===
* Chat_Model_Initialization_Workflow (Step 2)
