{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::LangChain]], [[domain::Chains]], [[domain::LLM Applications]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Abstract base class for creating structured sequences of calls to components with memory, callbacks, and composability support.

=== Description ===
The Chain class is the foundational abstraction in LangChain for creating sequences of operations with language models and other components. It extends RunnableSerializable to provide a standardized interface for stateful, observable, and composable applications. The class integrates memory management for maintaining conversation state, callback systems for observability and logging, and standardized input/output validation.

Chains implement both synchronous and asynchronous execution through invoke/ainvoke methods (modern Runnable interface) and __call__/acall methods (legacy interface). They support memory integration for maintaining state across invocations, extensive callback hooks for monitoring and instrumentation, and can be serialized to/from YAML or JSON for persistence.

=== Usage ===
Use Chain as a base class when building custom sequences of operations that need state management, observability, or composition with other LangChain components. Subclasses must implement the abstract _call method and define input_keys/output_keys properties.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/base.py libs/langchain/langchain_classic/chains/base.py]
* '''Lines:''' 1-807

=== Signature ===
<syntaxhighlight lang="python">
class Chain(RunnableSerializable[dict[str, Any], dict[str, Any]], ABC):
    memory: BaseMemory | None = None
    callbacks: Callbacks = Field(default=None, exclude=True)
    verbose: bool = Field(default_factory=_get_verbosity)
    tags: list[str] | None = None
    metadata: builtins.dict[str, Any] | None = None
    callback_manager: BaseCallbackManager | None = Field(default=None, exclude=True)

    @property
    @abstractmethod
    def input_keys(self) -> list[str]: ...

    @property
    @abstractmethod
    def output_keys(self) -> list[str]: ...

    @abstractmethod
    def _call(
        self,
        inputs: builtins.dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> builtins.dict[str, Any]: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| memory || BaseMemory or None || No || Memory object for maintaining state across invocations
|-
| callbacks || Callbacks || No || Callback handlers for lifecycle events
|-
| verbose || bool || No || Whether to print intermediate logs (defaults to global setting)
|-
| tags || list[str] or None || No || Tags associated with chain calls
|-
| metadata || dict[str, Any] or None || No || Metadata passed to callback handlers
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || dict[str, Any] || Dictionary containing chain outputs (keys defined by output_keys)
|}

== Core Methods ==

=== invoke (Synchronous Execution) ===
<syntaxhighlight lang="python">
def invoke(
    self,
    input: dict[str, Any],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]
</syntaxhighlight>

Executes the chain synchronously:
1. Prepares inputs (loads from memory if configured)
2. Configures callback manager
3. Triggers on_chain_start callback
4. Validates inputs against input_keys
5. Calls _call method (implemented by subclasses)
6. Prepares outputs (saves to memory if configured)
7. Triggers on_chain_end callback
8. Returns final outputs

=== ainvoke (Asynchronous Execution) ===
<syntaxhighlight lang="python">
async def ainvoke(
    self,
    input: dict[str, Any],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]
</syntaxhighlight>

Asynchronous version of invoke with the same lifecycle but using async callbacks and _acall method.

=== _call (Abstract Implementation Method) ===
<syntaxhighlight lang="python">
@abstractmethod
def _call(
    self,
    inputs: builtins.dict[str, Any],
    run_manager: CallbackManagerForChainRun | None = None,
) -> builtins.dict[str, Any]
</syntaxhighlight>

Private method that subclasses must implement. Contains the actual chain logic. Should not be called directly by users.

=== _acall (Async Implementation Method) ===
<syntaxhighlight lang="python">
async def _acall(
    self,
    inputs: builtins.dict[str, Any],
    run_manager: AsyncCallbackManagerForChainRun | None = None,
) -> builtins.dict[str, Any]
</syntaxhighlight>

Asynchronous version of _call. Default implementation runs _call in an executor. Override for true async support.

== Memory Integration ==

=== prep_inputs ===
Prepares chain inputs by loading variables from memory:
<syntaxhighlight lang="python">
def prep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
    # Handle single string input
    if not isinstance(inputs, dict):
        _input_keys = set(self.input_keys)
        if self.memory is not None:
            _input_keys = _input_keys.difference(self.memory.memory_variables)
        inputs = {next(iter(_input_keys)): inputs}

    # Load from memory
    if self.memory is not None:
        external_context = self.memory.load_memory_variables(inputs)
        inputs = dict(inputs, **external_context)
    return inputs
</syntaxhighlight>

=== prep_outputs ===
Validates outputs and saves to memory:
<syntaxhighlight lang="python">
def prep_outputs(
    self,
    inputs: dict[str, str],
    outputs: dict[str, str],
    return_only_outputs: bool = False,
) -> dict[str, str]:
    self._validate_outputs(outputs)
    if self.memory is not None:
        self.memory.save_context(inputs, outputs)
    if return_only_outputs:
        return outputs
    return {**inputs, **outputs}
</syntaxhighlight>

== Validation ==

=== _validate_inputs ===
Checks that all required input keys are present:
<syntaxhighlight lang="python">
def _validate_inputs(self, inputs: Any) -> None:
    # Handle non-dict inputs
    if not isinstance(inputs, dict):
        _input_keys = set(self.input_keys)
        if self.memory is not None:
            _input_keys = _input_keys.difference(self.memory.memory_variables)
        if len(_input_keys) != 1:
            raise ValueError("Single string input requires exactly one non-memory input key")

    # Check for missing keys
    missing_keys = set(self.input_keys).difference(inputs)
    if missing_keys:
        raise ValueError(f"Missing some input keys: {missing_keys}")
</syntaxhighlight>

=== _validate_outputs ===
Checks that all required output keys are present:
<syntaxhighlight lang="python">
def _validate_outputs(self, outputs: dict[str, Any]) -> None:
    missing_keys = set(self.output_keys).difference(outputs)
    if missing_keys:
        raise ValueError(f"Missing some output keys: {missing_keys}")
</syntaxhighlight>

== Legacy Methods (Deprecated) ==

=== __call__ ===
Deprecated in favor of invoke (removal in 1.0):
<syntaxhighlight lang="python">
@deprecated("0.1.0", alternative="invoke", removal="1.0")
def __call__(
    self,
    inputs: dict[str, Any] | Any,
    return_only_outputs: bool = False,
    callbacks: Callbacks = None,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    include_run_info: bool = False,
) -> dict[str, Any]
</syntaxhighlight>

=== run ===
Deprecated convenience method (removal in 1.0):
<syntaxhighlight lang="python">
@deprecated("0.1.0", alternative="invoke", removal="1.0")
def run(
    self,
    *args: Any,
    callbacks: Callbacks = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    # Accepts positional or keyword arguments
    # Returns only the value from output_key (single output chains)
</syntaxhighlight>

=== apply ===
Deprecated batch method (removal in 1.0):
<syntaxhighlight lang="python">
@deprecated("0.1.0", alternative="batch", removal="1.0")
def apply(
    self,
    input_list: list[builtins.dict[str, Any]],
    callbacks: Callbacks = None,
) -> list[builtins.dict[str, str]]:
    return [self(inputs, callbacks=callbacks) for inputs in input_list]
</syntaxhighlight>

== Serialization ==

=== save ===
Saves chain to YAML or JSON file:
<syntaxhighlight lang="python">
def save(self, file_path: Path | str) -> None:
    if self.memory is not None:
        raise ValueError("Saving of memory is not yet supported.")

    chain_dict = self.model_dump()
    if "_type" not in chain_dict:
        raise NotImplementedError(f"Chain {self} does not support saving.")

    save_path = Path(file_path)
    if save_path.suffix == ".json":
        with save_path.open("w") as f:
            json.dump(chain_dict, f, indent=4)
    elif save_path.suffix in (".yaml", ".yml"):
        with save_path.open("w") as f:
            yaml.dump(chain_dict, f, default_flow_style=False)
</syntaxhighlight>

=== dict ===
Returns dictionary representation:
<syntaxhighlight lang="python">
def dict(self, **kwargs: Any) -> dict:
    _dict = super().model_dump(**kwargs)
    with contextlib.suppress(NotImplementedError):
        _dict["_type"] = self._chain_type
    return _dict
</syntaxhighlight>

== Callback Lifecycle ==

Chains trigger callbacks at key points:
1. on_chain_start: Before execution begins
2. on_chain_error: If an exception occurs
3. on_chain_end: After successful completion

Custom chains can trigger additional callbacks:
* on_text: Log intermediate text
* on_llm_start/on_llm_end: When calling LLMs
* on_tool_start/on_tool_end: When using tools

== Schema Generation ==

=== get_input_schema ===
Generates Pydantic model for inputs:
<syntaxhighlight lang="python">
def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
    return create_model("ChainInput", **dict.fromkeys(self.input_keys, (Any, None)))
</syntaxhighlight>

=== get_output_schema ===
Generates Pydantic model for outputs:
<syntaxhighlight lang="python">
def get_output_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
    return create_model("ChainOutput", **dict.fromkeys(self.output_keys, (Any, None)))
</syntaxhighlight>

== Usage Examples ==

=== Basic Custom Chain ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun

class MyChain(Chain):
    """Custom chain that reverses input text."""

    @property
    def input_keys(self) -> list[str]:
        return ["text"]

    @property
    def output_keys(self) -> list[str]:
        return ["reversed"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        text = inputs["text"]
        reversed_text = text[::-1]

        if run_manager:
            run_manager.on_text(f"Reversing: {text}\n", verbose=self.verbose)

        return {"reversed": reversed_text}

# Usage
chain = MyChain(verbose=True)
result = chain.invoke({"text": "hello"})
print(result)  # {"reversed": "olleh"}
</syntaxhighlight>

=== Chain with Memory ===
<syntaxhighlight lang="python">
from langchain_classic.chains.base import Chain
from langchain_classic.memory import ConversationBufferMemory

class ConversationChain(Chain):
    llm: Any
    memory: BaseMemory

    @property
    def input_keys(self) -> list[str]:
        return ["input"]

    @property
    def output_keys(self) -> list[str]:
        return ["response"]

    def _call(self, inputs: dict, run_manager=None) -> dict:
        # Memory loaded in prep_inputs
        # inputs now contains history
        response = self.llm.predict(inputs["input"])
        return {"response": response}

# Usage
memory = ConversationBufferMemory()
chain = ConversationChain(llm=my_llm, memory=memory)

chain.invoke({"input": "Hi, I'm Alice"})  # Memory saves this
chain.invoke({"input": "What's my name?"})  # Memory provides context
</syntaxhighlight>

=== Chain with Callbacks ===
<syntaxhighlight lang="python">
from langchain_core.callbacks import BaseCallbackHandler

class LoggingCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"Chain ended with outputs: {outputs}")

# Usage
chain = MyChain(callbacks=[LoggingCallback()], verbose=True)
result = chain.invoke({"text": "hello"})
# Prints:
# Chain started with inputs: {'text': 'hello'}
# Chain ended with outputs: {'reversed': 'olleh'}
</syntaxhighlight>

=== Async Chain ===
<syntaxhighlight lang="python">
import asyncio
from langchain_classic.chains.base import Chain

class AsyncChain(Chain):
    @property
    def input_keys(self) -> list[str]:
        return ["query"]

    @property
    def output_keys(self) -> list[str]:
        return ["result"]

    def _call(self, inputs, run_manager=None):
        return {"result": f"Sync: {inputs['query']}"}

    async def _acall(self, inputs, run_manager=None):
        await asyncio.sleep(0.1)  # Simulate async work
        return {"result": f"Async: {inputs['query']}"}

# Usage
chain = AsyncChain()
result = await chain.ainvoke({"query": "test"})
print(result)  # {"result": "Async: test"}
</syntaxhighlight>

=== Composable Chains ===
<syntaxhighlight lang="python">
# Chains are Runnables, so they compose with pipes
from langchain_core.runnables import RunnablePassthrough

chain1 = MyChain()
chain2 = AnotherChain()

# Pipe chains together
composed = chain1 | chain2

# Or use with other Runnables
composed = RunnablePassthrough() | chain1 | chain2
</syntaxhighlight>

== Related Pages ==
* [[implemented_by::Implementation:langchain-ai_langchain_LLMChain]]
* [[extends::Concept:RunnableSerializable]]
* [[uses::Implementation:langchain-ai_langchain_BaseMemory]]
* [[uses::Concept:Callback_System]]
