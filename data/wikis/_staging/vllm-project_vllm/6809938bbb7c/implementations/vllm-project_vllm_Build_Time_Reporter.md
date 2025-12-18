{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Build Analysis]], [[domain::Performance Profiling]], [[domain::Ninja Build]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A build time analysis tool that parses Ninja build logs to identify compilation bottlenecks and parallelism efficiency.

=== Description ===
This tool analyzes Ninja's .ninja_log to provide detailed build time insights, identifying the longest compilation steps by file extension (.cu.o for CUDA, .cpp.o for C++, .so for linking) and calculating weighted build times that account for parallelism. It computes how much each compilation step actually contributed to total build time by dividing elapsed time by concurrent tasks, revealing serialized bottlenecks. The tool reports the top 10 slowest steps per category, overall parallelism factor (e.g., 9.1x for 10K seconds elapsed becoming 1.1K weighted), and average build step completion rate. This is essential for optimizing large CUDA codebases like vLLM.

=== Usage ===
Use this tool after Ninja builds to identify compilation hotspots, diagnose poor parallelism, understand which kernel files are slowest to compile, or track build time improvements after code changes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/tools/report_build_time_ninja.py tools/report_build_time_ninja.py]

=== Signature ===
<syntaxhighlight lang="python">
class Target:
    def __init__(self, start: float, end: float)
    def Duration(self) -> float
    def WeightedDuration(self) -> float
    def SetWeightedDuration(self, weighted_duration: float)

def ReadTargets(log, show_all: bool) -> list[Target]
def SummarizeEntries(entries: list[Target], extra_step_types: str | None)
def GetExtension(target: Target, extra_patterns: str | None) -> str
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# After a Ninja build
python tools/report_build_time_ninja.py -C build/temp.linux-x86_64-cpython-310

# With custom patterns
python tools/report_build_time_ninja.py \
    -C build/temp.linux-x86_64-cpython-310 \
    --step-types "machete;flashinfer"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| -C BUILD_DIR || str || Build directory containing .ninja_log
|-
| --log-file || str || Specific .ninja_log file to analyze (overrides -C)
|-
| --step-types || str || Semicolon-separated fnmatch patterns for custom grouping
|-
| .ninja_log || File || Ninja build log (auto-detected in build dir)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| build_summary || stdout || Formatted analysis with slowest steps per category
|-
| weighted_times || float || Time accounting for parallelism (weighted seconds)
|-
| elapsed_times || float || Total wall-clock time per step
|-
| parallelism_factor || float || Average concurrent tasks (e.g., 9.1x)
|-
| completion_rate || float || Build steps per second
|}

== Output Format ==

<syntaxhighlight lang="text">
Longest build steps for .cpp.o:
      1.0 weighted s to build ...torch_bindings.cpp.o (12.4 s elapsed time)
      2.0 weighted s to build ..._attn_c.dir/csrc... (23.5 s elapsed time)

Longest build steps for .so (linking):
      0.1 weighted s to build _moe_C.abi3.so (1.0 s elapsed time)
      6.2 weighted s to build _C.abi3.so (6.2 s elapsed time)

Longest build steps for .cu.o:
     15.3 weighted s to build ...machete_mm_... (183.5 s elapsed time)
     37.4 weighted s to build ...scaled_mm_c3x.cu... (449.0 s elapsed time)
    344.8 weighted s to build ...attention_...cu.o (1087.2 s elapsed time)

1110.0 s weighted time (10120.4 s elapsed time sum, 9.1x parallelism)
134 build steps completed, average of 0.12/s
</syntaxhighlight>

== Key Metrics ==

{| class="wikitable"
|-
! Metric !! Meaning !! Good Value !! Poor Value
|-
| Weighted time || Actual contribution to build time || Close to longest step || Much longer than longest
|-
| Parallelism factor || Average concurrent jobs || 8-16x (8-16 cores) || 1-2x (serialized)
|-
| Completion rate || Steps/second || >0.5/s || <0.1/s
|-
| Max weighted step || Bottleneck severity || <100s || >300s
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Basic analysis
python tools/report_build_time_ninja.py -C build/temp.linux-x86_64-cpython-310

# With custom groupings
python tools/report_build_time_ninja.py \
    -C build/ \
    --step-types "machete;marlin;flashinfer;cutlass"

# Analyze specific log file
python tools/report_build_time_ninja.py \
    --log-file /path/to/.ninja_log

# Compare builds before/after optimization
python tools/report_build_time_ninja.py -C build_before/ > before.txt
python tools/report_build_time_ninja.py -C build_after/ > after.txt
diff before.txt after.txt

# Identify CUDA kernel compilation bottlenecks
python tools/report_build_time_ninja.py -C build/ | grep ".cu.o"
# Look for files with high weighted time but moderate elapsed time
# (indicates they block other compilations)
</syntaxhighlight>

== Interpreting Results ==

=== High Weighted Time, High Elapsed Time ===
* Intrinsically slow to compile (complex templates, large kernels)
* Consider: Splitting file, reducing template complexity, using explicit instantiations

=== High Weighted Time, Moderate Elapsed Time ===
* Serialization bottleneck (blocks other work)
* Consider: Reordering dependencies, moving to earlier compilation stage

=== Low Weighted Time, High Elapsed Time ===
* Runs in parallel with many other tasks (good parallelism)
* Less urgent optimization target

=== Poor Parallelism Factor (<4x on 8-core machine) ===
* Dependencies causing serialization
* Consider: Reducing interdependencies, using forward declarations, splitting libraries

== Related Pages ==
* [[Tool:Ninja_Build]]
* [[Analysis:Build_Time_Optimization]]
* [[Tool:Compilation_Profiling]]
* [[Build:CUDA_Compilation]]
