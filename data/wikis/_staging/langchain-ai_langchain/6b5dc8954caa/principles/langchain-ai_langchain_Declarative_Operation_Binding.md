{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Runnables|https://python.langchain.com/docs/expression_language/]]
* [[source::Doc|Builder Pattern|https://refactoring.guru/design-patterns/builder]]
|-
! Domains
| [[domain::LLM]], [[domain::Design_Patterns]], [[domain::Functional_Programming]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Pattern for queueing model operations (tool binding, structured output) that are applied when the actual model is instantiated.

=== Description ===

Declarative Operation Binding allows model configuration through method chaining even when the underlying model doesn't exist yet. Operations like `bind_tools()` and `with_structured_output()` are "declared" and stored in a queue, then applied in order when the model is finally instantiated at invocation time.

This pattern enables:
* Fluent API on configurable models
* Separation of configuration from instantiation
* Composable model modifications
* Consistent interface across fixed and configurable models

=== Usage ===

Use Declarative Operation Binding when:
* Configuring models before knowing which provider/model to use
* Building reusable model configuration "recipes"
* Creating fluent model configuration chains
* Implementing model wrappers with deferred instantiation

== Theoretical Basis ==

Declarative Operation Binding implements **Builder Pattern** with **Command Pattern** for deferred execution.

'''1. Operation Queue Structure'''

<syntaxhighlight lang="python">
# Each operation is stored as a tuple
OperationRecord = tuple[
    str,        # Method name: "bind_tools", "with_structured_output"
    tuple,      # Positional arguments
    dict        # Keyword arguments
]

# Queue is a list of operations in order
queued_operations: list[OperationRecord] = [
    ("bind_tools", ([Tool1, Tool2],), {"tool_choice": "auto"}),
    ("with_structured_output", (Schema,), {"strict": True}),
]
</syntaxhighlight>

'''2. Operation Capture via __getattr__'''

<syntaxhighlight lang="python">
# Pseudo-code for capturing declarative operations
class ConfigurableModel:
    _DECLARATIVE_METHODS = {"bind_tools", "with_structured_output"}

    def __getattr__(self, name):
        if name in self._DECLARATIVE_METHODS:
            # Return a function that queues the operation
            def queue_operation(*args, **kwargs):
                # Create new ConfigurableModel with operation added
                return ConfigurableModel(
                    default_config=self.default_config,
                    configurable_fields=self.configurable_fields,
                    queued_operations=[
                        *self.queued_operations,
                        (name, args, kwargs)  # Add to queue
                    ]
                )
            return queue_operation

        raise AttributeError(f"Unknown method: {name}")
</syntaxhighlight>

'''3. Operation Application at Instantiation'''

<syntaxhighlight lang="python">
# Pseudo-code for applying queued operations
class ConfigurableModel:
    def _instantiate_model(self, config):
        # Create base model
        model = create_model(**config)

        # Apply each queued operation in order
        for method_name, args, kwargs in self.queued_operations:
            method = getattr(model, method_name)
            model = method(*args, **kwargs)

        return model

    def invoke(self, input, config=None):
        # Resolve config and instantiate
        final_config = self._merge_config(config)
        model = self._instantiate_model(final_config)

        # Invoke the fully configured model
        return model.invoke(input)
</syntaxhighlight>

'''4. Immutable Chaining'''

<syntaxhighlight lang="python">
# Each operation returns a NEW configurable, preserving immutability
original = init_chat_model(configurable_fields=("model",))

with_tools = original.bind_tools([Tool])
# original is unchanged
# with_tools is new ConfigurableModel with operation queued

with_output = with_tools.with_structured_output(Schema)
# with_tools is unchanged
# with_output has both operations queued

# Chain in one expression (functional style)
configured = (
    init_chat_model(configurable_fields=("model",))
    .bind_tools([Tool1, Tool2])
    .with_structured_output(Schema)
)
</syntaxhighlight>

'''5. Order Preservation'''

<syntaxhighlight lang="python">
# Order matters - operations applied in queue order
# This may produce different results than reversed order

# Order 1: bind_tools first, then structured_output
model_a = (
    configurable
    .bind_tools([Tool])
    .with_structured_output(Schema)
)
# Applied as: model.bind_tools([Tool]).with_structured_output(Schema)

# Order 2: structured_output first, then bind_tools
model_b = (
    configurable
    .with_structured_output(Schema)
    .bind_tools([Tool])
)
# Applied as: model.with_structured_output(Schema).bind_tools([Tool])

# These may behave differently depending on the underlying model
</syntaxhighlight>

'''6. Consistent Interface'''

<syntaxhighlight lang="python">
# Both fixed and configurable models have same API
from langchain.chat_models import init_chat_model

# Fixed model
fixed: BaseChatModel = init_chat_model("gpt-4o")
fixed_configured = fixed.bind_tools([Tool])  # Returns modified BaseChatModel

# Configurable model
configurable: _ConfigurableModel = init_chat_model(configurable_fields=("model",))
config_configured = configurable.bind_tools([Tool])  # Returns _ConfigurableModel

# Both can be used the same way
response1 = fixed_configured.invoke("Hello")
response2 = config_configured.invoke("Hello", config={"configurable": {"model": "gpt-4o"}})
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_ConfigurableModel_declarative_methods]]

=== Used By Workflows ===
* Chat_Model_Initialization_Workflow (Step 5)
