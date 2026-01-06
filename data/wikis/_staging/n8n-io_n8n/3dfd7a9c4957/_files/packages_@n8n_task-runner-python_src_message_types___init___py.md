# File: `packages/@n8n/task-runner-python/src/message_types/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Imports | broker, runner |

## Understanding

**Status:** ✅ Explored

**Purpose:** Message types module initialization

**Mechanism:** Re-exports all message type classes from broker and runner submodules for convenient importing. Provides unified access to all protocol message definitions.

**Significance:** Module organization. Enables clean imports like `from src.message_types import BrokerTaskSettings`.
