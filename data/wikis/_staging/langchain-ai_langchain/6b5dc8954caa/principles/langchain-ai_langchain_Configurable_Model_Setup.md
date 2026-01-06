{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Runnables|https://python.langchain.com/docs/expression_language/]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Runtime_Configuration]], [[domain::Design_Patterns]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Pattern for deferring model instantiation until runtime, enabling dynamic model selection and parameter override via configuration.

=== Description ===

Configurable Model Setup creates a model wrapper that delays actual model creation until invocation time. This enables:
* **Runtime model switching:** Change models without code changes
* **A/B testing:** Route different users to different models
* **Dynamic configuration:** Adjust parameters per request
* **Lazy instantiation:** Don't create model until actually needed

The configurable model acts as a proxy that reads `config["configurable"]` at invoke time and instantiates the appropriate model with those parameters.

=== Usage ===

Use Configurable Model Setup when:
* Building multi-model applications
* Implementing dynamic model routing
* Creating user-configurable interfaces
* Developing evaluation harnesses
* Building model comparison tools

Security note: Restrict `configurable_fields` in production to prevent runtime override of sensitive parameters like `api_key` or `base_url`.

== Theoretical Basis ==

Configurable Model Setup implements **Lazy Initialization** and **Proxy Pattern**.

'''1. Deferred Instantiation'''

<syntaxhighlight lang="python">
# Fixed model: instantiated immediately
fixed = init_chat_model("gpt-4o")  # Model created NOW

# Configurable model: instantiated at invoke time
configurable = init_chat_model(configurable_fields=("model",))  # No model yet

# Model created when invoked
response = configurable.invoke(
    "Hello",
    config={"configurable": {"model": "gpt-4o"}}  # NOW model is created
)
</syntaxhighlight>

'''2. Configuration Merging'''

<syntaxhighlight lang="python">
# Pseudo-code for config resolution
class ConfigurableModel:
    def _resolve_config(self, runtime_config: dict) -> dict:
        # Start with defaults
        config = dict(self.default_config)

        # Extract runtime overrides
        configurable = runtime_config.get("configurable", {})

        # Apply prefix if set
        for key, value in configurable.items():
            unprefixed = remove_prefix(key, self.config_prefix)

            # Only apply if field is configurable
            if self.configurable_fields == "any" or unprefixed in self.configurable_fields:
                config[unprefixed] = value

        return config

    def invoke(self, input, config=None):
        # Resolve final configuration
        final_config = self._resolve_config(config or {})

        # Instantiate model with resolved config
        model = init_chat_model_helper(**final_config)

        # Apply queued operations
        for op_name, args, kwargs in self.queued_operations:
            model = getattr(model, op_name)(*args, **kwargs)

        # Invoke
        return model.invoke(input)
</syntaxhighlight>

'''3. Declarative Operation Queuing'''

<syntaxhighlight lang="python">
# Operations are queued, not applied immediately
# Because actual model doesn't exist yet

class ConfigurableModel:
    def bind_tools(self, tools, **kwargs):
        # Can't call model.bind_tools() - no model yet!
        # Instead, queue the operation
        return ConfigurableModel(
            default_config=self.default_config,
            configurable_fields=self.configurable_fields,
            queued_operations=[
                *self.queued_operations,
                ("bind_tools", (tools,), kwargs)  # Queue it
            ]
        )

    def _instantiate_model(self, config):
        model = create_model(**config)

        # NOW apply queued operations
        for name, args, kwargs in self.queued_operations:
            model = getattr(model, name)(*args, **kwargs)

        return model
</syntaxhighlight>

'''4. Field Restriction for Security'''

<syntaxhighlight lang="python">
# Security: Restrict what can be configured at runtime
DANGEROUS_FIELDS = {"api_key", "base_url", "credentials"}

def safe_configurable_model(default_model: str, **kwargs):
    """Create configurable model with restricted fields."""
    safe_fields = ["model", "model_provider", "temperature", "max_tokens"]
    # Explicitly exclude dangerous fields

    return init_chat_model(
        default_model,
        configurable_fields=safe_fields,
        **kwargs
    )

# Attacker cannot override api_key at runtime:
model = safe_configurable_model("gpt-4o", api_key="sk-...")
model.invoke("Hi", config={"configurable": {
    "api_key": "malicious-key"  # IGNORED - not in safe_fields
}})
</syntaxhighlight>

'''5. Config Prefix for Multi-Model'''

<syntaxhighlight lang="python">
# Multiple configurable models in same app
primary = init_chat_model(
    "gpt-4o",
    configurable_fields="any",
    config_prefix="primary"
)

secondary = init_chat_model(
    "claude-sonnet-4-5-20250929",
    configurable_fields="any",
    config_prefix="secondary"
)

# Configure each independently
config = {
    "configurable": {
        "primary_model": "gpt-4o-mini",
        "primary_temperature": 0.5,
        "secondary_model": "claude-3-haiku",
        "secondary_temperature": 0.0,
    }
}

response1 = primary.invoke("Hello", config=config)
response2 = secondary.invoke("Hello", config=config)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_ConfigurableModel_class]]

=== Used By Workflows ===
* Chat_Model_Initialization_Workflow (Step 4)
