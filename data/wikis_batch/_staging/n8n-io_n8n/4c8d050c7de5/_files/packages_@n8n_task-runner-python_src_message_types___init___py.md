# File: `packages/@n8n/task-runner-python/src/message_types/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Imports | broker, runner |

## Understanding

**Status:** ✅ Explored

**Purpose:** Public API for message type imports

**Mechanism:** This module serves as the public interface for the message_types package, importing and re-exporting all message classes from the `broker` and `runner` submodules. It defines `__all__` to explicitly list all exported types: 8 broker message types (`BrokerMessage`, `BrokerInfoRequest`, `BrokerRunnerRegistered`, `BrokerTaskOfferAccept`, `BrokerTaskSettings`, `BrokerTaskCancel`, `BrokerRpcResponse`) and 8 runner message types (`RunnerMessage`, `RunnerInfo`, `RunnerTaskOffer`, `RunnerTaskAccepted`, `RunnerTaskRejected`, `RunnerTaskDone`, `RunnerTaskError`, `RunnerRpcCall`).

**Significance:** This is a standard Python package initialization pattern that provides a clean, centralized import interface. Instead of importing from nested modules (`from message_types.broker import BrokerMessage`), consumers can import directly from the package (`from message_types import BrokerMessage`). The explicit `__all__` declaration controls what's exposed when using wildcard imports and serves as documentation for the package's public API. This encapsulation makes it easier to reorganize internal module structure without breaking external code.
