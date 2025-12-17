{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Security]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Controlled execution of user-provided Python code within a sandboxed environment supporting both batch and iterative processing modes.

=== Description ===

Code execution implements the core runtime environment for executing user-submitted Python code with controlled access to built-in functions, isolated global namespace, and captured output streams. The system supports two execution modes:

**All Items Mode (Batch Processing)**:
* Code receives entire dataset at once via `items` variable
* Single execution pass processes all input data
* Efficient for operations requiring full dataset visibility
* Common for aggregations, transformations, filtering

**Per Item Mode (Iterative Processing)**:
* Code executes once per input item via `item` variable
* Loop managed by runner, code processes single item
* Efficient for independent item operations
* Common for enrichment, validation, individual transformations

The execution environment provides:
* **Sandboxed Globals**: Clean namespace with only allowed built-ins and injected variables
* **Output Capture**: Custom print() implementation captures output for logging
* **Error Handling**: Exceptions caught and formatted with context
* **Result Validation**: Output must be JSON-serializable dictionaries

This controlled execution model balances flexibility (users write Python) with security (limited capabilities) and reliability (predictable behavior).

=== Usage ===

Apply this principle when building code execution systems where:
* Users submit code that must process structured data
* Both batch and streaming processing patterns are required
* Security requires limiting access to built-in functions
* Output must be captured for logging or debugging
* Results must be structured and serializable
* Execution environment must be deterministic and reproducible
* System needs to support iterative and aggregate operations

== Theoretical Basis ==

Code execution implements **controlled evaluation with namespace isolation**:

**Execution Model:**

```python
# Compile user code to bytecode
code_obj = compile(user_code, '<string>', 'exec')

# Create isolated global namespace
sandbox_globals = {
    '__builtins__': filtered_builtins,
    'items': input_data,        # or 'item': single_item
    '_print_output': output_collector
}

# Execute in isolated namespace
exec(code_obj, sandbox_globals)

# Extract results from namespace
result = sandbox_globals.get('items')  # or process return value
```

**Namespace Isolation:**

Python's `exec()` accepts explicit global and local namespaces:
```python
exec(code, globals, locals)
```

By providing custom `globals`, we control:
* Available built-in functions (filter dangerous ones)
* Injected variables (items, item, helper functions)
* Output mechanisms (custom print)

**Built-in Filtering:**

```python
ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
    'filter', 'float', 'int', 'len', 'list', 'map',
    'max', 'min', 'range', 'round', 'set', 'sorted',
    'str', 'sum', 'tuple', 'zip'
}

filtered_builtins = {
    name: getattr(builtins, name)
    for name in ALLOWED_BUILTINS
}
```

Excluded dangerous built-ins:
* `eval`, `exec`, `compile` - code injection
* `open`, `input` - I/O operations
* `__import__` - dynamic imports
* `globals`, `locals`, `vars` - namespace introspection

**Execution Modes:**

1. **All Items Mode**:
   ```python
   # User code receives:
   items = [{'value': 1}, {'value': 2}, {'value': 3}]

   # User code transforms:
   items = [{'result': item['value'] * 2} for item in items]

   # Runner extracts:
   result = items
   ```

2. **Per Item Mode**:
   ```python
   results = []
   for input_item in input_items:
       sandbox_globals = {'item': input_item}
       exec(code_obj, sandbox_globals)
       results.append(sandbox_globals.get('item'))
   ```

**Output Capture:**

```python
class OutputCollector:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

# Override print in sandbox
sandbox_globals['print'] = lambda *args: output_collector.write(' '.join(map(str, args)))
```

**Error Handling:**

```python
try:
    exec(code_obj, sandbox_globals)
except Exception as e:
    error_info = {
        'type': type(e).__name__,
        'message': str(e),
        'line': extract_line_number(e)
    }
    return TaskExecutionError(error_info)
```

**Result Validation:**

```python
# Ensure output is JSON-serializable
try:
    json.dumps(result)
except TypeError:
    raise ValueError("Result must be JSON-serializable")
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_all_items]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Subprocess_Isolation]] - Provides process boundary for execution
* [[Principle:n8n-io_n8n_Static_Security_Analysis]] - Pre-validates code before execution
* [[Principle:n8n-io_n8n_Result_Collection]] - Collects execution results
