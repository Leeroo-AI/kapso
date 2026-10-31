# Core Module - Shared fundamentals
#
# Contains configuration utilities and LLM backend. Nothing here may import
# from kapso.execution: the gate registry (gated_mcp/presets.py) reads the
# packaged config through kapso.core.config, and the execution package
# imports the gate registry — an execution import here closes that cycle
# on every cold start of the MCP server subprocess.
#
# Nothing is re-exported here either: a `from kapso.core import LLMBackend`
# imported litellm on every cold start of the package — a second and a half
# of the MCP server's startup, which Codex bounds at ten seconds — for the
# config loader's sake. Callers import `kapso.core.llm` and
# `kapso.core.config` by name.
