# Principle: Sandboxed_Code_Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Program of Thoughts: A Survey of Agent Code Generation|https://arxiv.org/abs/2305.10601]]
* [[source::Paper|PAL: Program-aided Language Models|https://arxiv.org/abs/2211.10435]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Code_Execution]], [[domain::Agent_Tools]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Sandboxed Python code execution for data analysis and computation. Allows the agent to execute Python code safely in an isolated environment.

=== Description ===

Sandboxed code execution enables autonomous research agents to perform computations, data analysis, and programmatic operations that would be difficult or impossible through natural language alone. The sandbox provides a secure execution environment that isolates the agent's code from the host system.

Key features of the implementation:

1. **Sandbox Fusion Service** - Uses an external sandbox service for isolated execution
2. **Multi-Endpoint Support** - Can distribute load across multiple sandbox endpoints
3. **Retry Logic** - Up to 8 retries with random endpoint selection
4. **Timeout Protection** - Default 50-second timeout per execution
5. **Output Capture** - Returns both stdout and stderr from execution
6. **Special Formatting** - Code is provided within `<code>` XML tags for reliable extraction

The PythonInterpreter tool is registered with detailed instructions for the agent on how to format code submissions and use print() for output visibility.

=== Usage ===

Use Sandboxed Code Execution when:
- Performing mathematical calculations or data analysis
- Processing structured data (CSV, JSON, etc.)
- Implementing algorithmic solutions
- Validating or testing hypotheses computationally

Execution constraints:
| Constraint | Value |
|------------|-------|
| Timeout | 50 seconds (configurable) |
| Output format | stdout + stderr |
| Language | Python only |
| Retries | Up to 8 attempts |

== Theoretical Basis ==

Sandboxed execution implements the "Program of Thoughts" paradigm where language models generate executable programs to solve problems:

<math>
\text{Answer} = \text{Execute}(\text{LLM}(\text{Problem} \rightarrow \text{Code}))
</math>

This approach offers advantages over pure chain-of-thought reasoning:
- Exact numerical computation
- Complex data manipulation
- Reproducible operations
- Structured output generation

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Sandboxed Code Execution Pattern
def execute_code(code: str, timeout: int = 50) -> str:
    """Execute Python code in a sandboxed environment."""

    # Select endpoint (random for load balancing)
    endpoints = os.environ.get('SANDBOX_FUSION_ENDPOINT', '').split(',')

    for attempt in range(8):
        try:
            # Randomly select an endpoint
            endpoint = random.choice(endpoints)

            # Execute in sandbox
            result = sandbox_service.run(
                code=code,
                language='python',
                timeout=timeout,
                endpoint=endpoint
            )

            # Collect output
            output = []
            if result.stdout:
                output.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                output.append(f"stderr:\n{result.stderr}")

            # Check for timeout
            if result.execution_time >= timeout - 1:
                output.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")

            return '\n'.join(output) if output else 'Finished execution.'

        except TimeoutError:
            if attempt == 7:  # Last attempt
                return f"[Python Interpreter Error] TimeoutError: Execution timed out."
            continue

        except Exception as e:
            if attempt == 7:
                return f"[Python Interpreter Error]: {str(e)}"
            continue

    return "[Python Interpreter Error]: All attempts failed."
</syntaxhighlight>

Key execution principles:
- **Isolation**: Code runs in a sandboxed environment separate from the agent host
- **Output Visibility**: Agents must use print() statements to see results
- **Graceful Failure**: All execution paths return meaningful error messages
- **Load Distribution**: Random endpoint selection distributes load and handles failures

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_PythonInterpreter_call]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
