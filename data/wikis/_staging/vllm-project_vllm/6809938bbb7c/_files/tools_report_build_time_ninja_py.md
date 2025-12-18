# File: `tools/report_build_time_ninja.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 325 |
| Classes | `Target` |
| Functions | `ReadTargets`, `GetExtension`, `SummarizeEntries`, `main` |
| Imports | argparse, collections, errno, fnmatch, os, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Ninja build performance analyzer

**Mechanism:** Parses `.ninja_log` files to analyze build performance with weighted time calculation. Processes: (1) reads ninja log v5 format with start/end timestamps, (2) calculates weighted duration by dividing elapsed time by number of concurrent tasks, (3) groups build steps by file extension (.cpp.o, .cu.o, .so linking), (4) sorts and reports longest build steps showing both weighted and elapsed times, (5) computes overall parallelism factor. Weighted time reveals actual impact on total build time by accounting for serialization bottlenecks.

**Significance:** Critical for identifying build bottlenecks in large C++/CUDA projects like vLLM. Weighted duration analysis is more informative than simple elapsed time as it reveals which steps are blocking parallelism. Essential for optimizing compilation times, especially for CUDA kernels which can have long serial compilation phases. Modified from Chromium's build tools.
