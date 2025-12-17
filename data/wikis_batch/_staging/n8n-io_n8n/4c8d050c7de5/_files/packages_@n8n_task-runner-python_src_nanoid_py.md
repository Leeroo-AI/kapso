# File: `packages/@n8n/task-runner-python/src/nanoid.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 21 |
| Functions | `nanoid` |
| Imports | secrets, string |

## Understanding

**Status:** ✅ Explored

**Purpose:** Cryptographically secure unique ID generator

**Mechanism:** Generates 22-character IDs using:
1. Character set: uppercase + lowercase + digits (62 characters total)
2. Uses secrets.randbits(6) for cryptographically secure random bits
3. Rejection sampling: only accepts values < 62 to ensure uniform distribution
4. Continues until 22 characters are generated
5. Collision probability calculation included: ~1.8e-16 at 10^12 IDs generated

**Significance:** Provides secure unique identifiers for runner IDs, task IDs, and RPC call IDs. Using secrets module (not random) ensures unpredictability for security-sensitive contexts. The rejection sampling approach maintains uniform distribution across the charset. The 22-character length provides sufficient entropy (62^22 ≈ 2^131 possibilities) for practical uniqueness in distributed systems. This is a Python implementation of the popular nanoid algorithm.
