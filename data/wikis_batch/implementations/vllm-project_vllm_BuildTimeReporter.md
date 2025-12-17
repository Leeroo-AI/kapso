# Build Time Reporter (Ninja Log Analyzer)

**File:** `/tmp/praxium_repo_583nq7ea/tools/report_build_time_ninja.py`
**Type:** Build Performance Analysis Tool
**Lines of Code:** 325
**Language:** Python
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The Build Time Reporter is a specialized tool that analyzes Ninja build logs to identify compilation bottlenecks. Adapted from Chromium's build tools, it calculates "weighted duration" for each compilation task by accounting for concurrent task execution, revealing which files truly slow down the build versus those that simply take long but run in parallel.

This tool is critical for optimizing vLLM's lengthy compilation process, helping developers prioritize optimization efforts by identifying serialization bottlenecks that simple elapsed time metrics miss.

## Implementation

### Core Algorithm

**Weighted Duration Calculation:**
```python
# Track running tasks and calculate weighted time
running_tasks = {}
last_weighted_time = 0.0

for time, action, target in task_start_stop_times:
    num_running = len(running_tasks)
    if num_running > 0:
        # Distribute time across concurrent tasks
        last_weighted_time += (time - last_time) / float(num_running)

    if action == "start":
        running_tasks[target] = last_weighted_time
    elif action == "stop":
        weighted_duration = last_weighted_time - running_tasks[target]
        target.SetWeightedDuration(weighted_duration)
        del running_tasks[target]
```

**Example Output:**
```
Longest build steps for .cu.o:
  15.3 weighted s to build machete_mm_impl_part1.cu.o (183.5 s elapsed time)
  37.4 weighted s to build scaled_mm_c3x.cu.o (449.0 s elapsed time)
  344.8 weighted s to build attention_kernels.cu.o (1087.2 s elapsed time)
1110.0 s weighted time (10120.4 s elapsed time sum, 9.1x parallelism)
134 build steps completed, average of 0.12/s
```

### Key Classes

**Target Class:**
```python
class Target:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.targets = []  # Output files
        self.weighted_duration = 0.0

    def Duration(self):
        return self.end - self.start

    def WeightedDuration(self):
        """Returns impact on total build time"""
        return self.weighted_duration
```

**Extension Grouping:**
```python
def GetExtension(target, extra_patterns):
    """Group by file type for reporting"""
    for output in target.targets:
        root, ext1 = os.path.splitext(output)
        _, ext2 = os.path.splitext(root)
        extension = ext2 + ext1

        if ext1 in [".so", ".TOC"]:
            extension = ".so (linking)"
        elif ext1 in [".obj", ".o"]:
            break  # Use first extension found
    return extension
```

## Usage

```bash
# Analyze most recent build
python tools/report_build_time_ninja.py -C build

# Analyze specific log file
python tools/report_build_time_ninja.py --log-file build/.ninja_log

# Group custom file types
python tools/report_build_time_ninja.py -C build --step-types "*.cu;*.cpp"
```

## Key Insights

**Weighted vs. Elapsed Time:**
- **Elapsed:** Total wall-clock time for task (can be misleading)
- **Weighted:** Time divided by concurrent tasks (shows true impact)

**Example:** A file taking 100s but running with 10 other tasks has weighted time of 10s, meaning removing it saves only 10s from total build time.

**Bottleneck Identification:**
- High weighted time → serialization bottleneck (optimize this first)
- High elapsed, low weighted → parallel overhead (less critical)

## Summary

This tool provides actionable insights for build optimization by identifying true bottlenecks through weighted duration analysis. Essential for maintaining vLLM's build performance as the codebase grows.
