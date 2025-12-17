# File: `vllm/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 107 |
| Imports | typing, version, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization and public API definition

**Mechanism:** This file serves as the main entry point for the vLLM package. It imports and exposes the public API including the LLM class, AsyncLLM, SamplingParams, PoolingParams, and various output types. It also defines the __version__ and sets up module-level attributes using __getattr__ for lazy loading of the ModelRegistry. The file includes a deprecation warning system for moved classes and configures basic package metadata.

**Significance:** Critical package infrastructure file that defines what components are publicly accessible to users. Controls the package's public interface and manages backward compatibility through deprecation warnings. Establishes the primary user-facing classes for running LLM inference (LLM, AsyncLLM) and configuring generation (SamplingParams, PoolingParams).
