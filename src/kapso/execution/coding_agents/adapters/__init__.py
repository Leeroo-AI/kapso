# Coding Agent Adapters
#
# Each adapter wraps a specific coding tool/agent to conform to
# CodingAgentInterface. The roster lives in agents.yaml — the factory
# loads adapters from there via importlib, so this package deliberately
# re-exports nothing (a hand-maintained lazy roster here drifted from
# the registry; stale-code audit 2026-08-26).
