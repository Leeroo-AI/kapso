# File: `benchmarks/benchmark_throughput.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Imports | sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated script that redirects users to the new vLLM CLI command.

**Mechanism:** Prints a deprecation message instructing users to use `vllm bench throughput` instead of running this script directly. Exits with status code 1 to indicate the script should not be executed. Provides alternative command invocations including the direct Python module path.

**Significance:** Migration utility that ensures backward compatibility during the transition to the unified vLLM CLI interface. Prevents confusion by clearly directing users to the new command structure for offline throughput benchmarks. Part of the broader effort to consolidate benchmark tools under `vllm bench` subcommands.
