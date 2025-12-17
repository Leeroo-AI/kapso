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

**Purpose:** Analyzes Ninja build logs to identify compilation bottlenecks and report weighted build times.

**Mechanism:** Parses .ninja_log file (v5 format) to extract build step timings. Calculates "weighted duration" for each task by dividing elapsed time by concurrent task count at each moment, representing each task's true impact on total build time. Groups results by file extension (.cu.o, .cpp.o, .so) and reports top 10 slowest steps per category with both weighted and elapsed times.

**Significance:** Build performance analysis tool critical for optimizing vLLM's lengthy compilation. The weighted duration metric (adapted from Chromium tools) accurately identifies serialization bottlenecks that simple elapsed time misses. Helps developers prioritize optimization efforts by revealing which files genuinely slow down the build.
