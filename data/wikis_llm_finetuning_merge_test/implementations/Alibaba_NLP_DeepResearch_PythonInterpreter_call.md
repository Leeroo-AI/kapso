# Implementation: PythonInterpreter_call

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Code_Execution]], [[domain::Agent_Tools]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

The call method of the PythonInterpreter tool that executes Python code in a sandboxed environment.

=== Description ===

The `PythonInterpreter.call()` method executes Python code using an external sandbox service (Sandbox Fusion). It provides secure, isolated code execution with:

- Random endpoint selection for load balancing
- Up to 8 retry attempts on failure
- Configurable timeout (default 50 seconds)
- Captured stdout and stderr output
- Error handling for timeouts and execution failures

The tool is designed for the agent to perform computations, data analysis, and programmatic operations.

=== Usage ===

Use `PythonInterpreter.call()` when:
- The agent needs to perform calculations
- Processing or analyzing data programmatically
- Testing algorithmic solutions
- Generating structured outputs through code

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/tool_python.py
* '''Lines:''' 72-112

=== Signature ===
<syntaxhighlight lang="python">
def call(self, params, files=None, timeout=50, **kwargs) -> str:
    """
    Execute Python code in a sandboxed environment.

    Args:
        params: str - The Python code to execute (raw code string)
        files: Optional - File attachments (unused in current implementation)
        timeout: int - Execution timeout in seconds (default: 50)
        **kwargs: Additional arguments (unused)

    Returns:
        str: Execution output containing:
            - stdout output (if any)
            - stderr output (if any)
            - Error messages (on failure)
            - "Finished execution." (if no output)

    Environment Variables:
        SANDBOX_FUSION_ENDPOINT: Comma-separated list of sandbox endpoints
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter

# Or access via TOOL_MAP
from react_agent import TOOL_MAP
python_tool = TOOL_MAP["PythonInterpreter"]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| params || str || Yes || Python code to execute (raw string)
|-
| files || Optional || No || File attachments (unused)
|-
| timeout || int || No || Execution timeout in seconds (default: 50)
|-
| **kwargs || dict || No || Additional arguments (unused)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || str || Execution output with stdout/stderr or error messages
|}

=== Output Format ===
<syntaxhighlight lang="text">
stdout:
[Standard output from the code]

stderr:
[Standard error output if any]
</syntaxhighlight>

== Usage Examples ==

=== Basic Code Execution ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter
import os

# Set sandbox endpoint
os.environ['SANDBOX_FUSION_ENDPOINT'] = 'http://localhost:8080'

python_tool = PythonInterpreter()

# Execute simple calculation
code = """
result = 2 + 2
print(f"2 + 2 = {result}")
"""

output = python_tool.call(code)
print(output)
# stdout:
# 2 + 2 = 4
</syntaxhighlight>

=== Data Analysis Example ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter

python_tool = PythonInterpreter()

code = """
import json

data = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]

avg_score = sum(d["score"] for d in data) / len(data)
top_scorer = max(data, key=lambda x: x["score"])

print(f"Average score: {avg_score:.2f}")
print(f"Top scorer: {top_scorer['name']} with {top_scorer['score']}")
"""

result = python_tool.call(code)
print(result)
</syntaxhighlight>

=== Agent Tool Call Format ===
<syntaxhighlight lang="python">
# How the agent formats PythonInterpreter tool calls
# Note: Code is placed in <code> tags, not in JSON arguments

tool_call = '''
<tool_call>
{"purpose": "Calculate compound interest", "name": "PythonInterpreter", "arguments": {"code": ""}}
<code>
principal = 1000
rate = 0.05
years = 10

amount = principal * (1 + rate) ** years
print(f"After {years} years: ${amount:.2f}")
</code>
</tool_call>
'''

# In react_agent.py, the code is extracted from <code> tags:
code_raw = tool_call.split('<code>')[1].split('</code>')[0].strip()
result = TOOL_MAP['PythonInterpreter'].call(code_raw)
</syntaxhighlight>

=== Handling Errors ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter

python_tool = PythonInterpreter()

# Code with error
code = """
x = 1 / 0
print(x)
"""

result = python_tool.call(code)
print(result)
# stderr:
# ZeroDivisionError: division by zero
</syntaxhighlight>

=== Custom Timeout ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter

python_tool = PythonInterpreter()

# Long-running computation with extended timeout
code = """
import time
time.sleep(5)
print("Completed after 5 seconds")
"""

# Set 10-second timeout
result = python_tool.call(code, timeout=10)
print(result)
</syntaxhighlight>

=== Complex Data Processing ===
<syntaxhighlight lang="python">
from tool_python import PythonInterpreter

python_tool = PythonInterpreter()

code = """
import re
from collections import Counter

text = '''
The quick brown fox jumps over the lazy dog.
The dog was not lazy, the fox was quick.
'''

# Tokenize and count
words = re.findall(r'\\w+', text.lower())
word_counts = Counter(words)

print("Word frequencies:")
for word, count in word_counts.most_common(5):
    print(f"  {word}: {count}")
"""

result = python_tool.call(code)
print(result)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Sandboxed_Code_Execution]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Sandbox_Execution_Environment]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
