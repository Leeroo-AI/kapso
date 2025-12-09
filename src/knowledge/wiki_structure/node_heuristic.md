# Node Type: Heuristic

## Definition

**Role:** Wisdom (Tactical Intuition)

A **Heuristic** captures tribal knowledge, decision frameworks, and optimizations. It represents the "Art" of engineering - practical wisdom not explicitly stated in documentation.

## Purpose

- Documents best practices and gotchas
- Captures optimization techniques
- Provides decision frameworks (X vs Y)
- Preserves institutional knowledge

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Wisdom |
| Function | Tactical Intuition |
| Scope | Tribal knowledge, decision frameworks, optimizations |
| Nature | The "Art" of Engineering |

## Required Metadata

| Field | Description |
|-------|-------------|
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

## Template Structure

```mediawiki
= Heuristic: {Name_of_Insight} =
[[Category:Heuristics]]

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Repo URL
|| [{repoUrl} {repo_name}]
|-
! Domain(s)
|| {Domain_Tag_1}, {Domain_Tag_2}
|-
! Last Updated
|| {YYYY-MM-DD HH:MM GMT}
|}

== Context & Scenario ==
Applies when {specific condition, e.g., "training Llama-3 models with limited VRAM"}.

== The Insight (Rule of Thumb) ==
{The core advice.}
* **Guideline:** {e.g., "Prefer `bitsandbytes` 4-bit loading over 8-bit for this specific task."}
* **Parameter Tip:** {e.g., "Set `lora_alpha` to be exactly 2x of `lora_r`."}

== Reasoning ==
{Why this works.}
* **Experience:** "Experiments show that 8-bit quantization degrades performance significantly on reasoning tasks, while 4-bit (NF4) preserves it."
* **Source:** {Optional: "Derived from GitHub Issue #123" or "Observed in typical runs."}

== Code Pattern ==
<syntaxhighlight lang="python">
# Recommended Configuration
config = LoraConfig(
    r=16,
    lora_alpha=32,  # Adhering to the 2x rule
    ...
)
</syntaxhighlight>

== Code References ==
'''GitHub Repository:''' [{repoUrl} {repo_name}]

'''Related Files:'''
* [{repoUrl}/blob/{branch}/path/to/file.py file.py] - Related module

== Related Pages ==
* [[applies_to::Implementation:{repo_name}/{Target_Tool}]] - Implementation this applies to
* [[applies_to::Workflow:{repo_name}/{Target_Workflow}]] - Workflow this applies to
* [[related_to::Principle:{repo_name}/{Related_Principle}]] - Related theoretical principle
```

## Coverage Target

**Minimum:** 4-6 pages per repository

## Sources to Scan

- Comments with `TODO`, `NOTE`, `WARNING`, `HACK`
- GitHub Issues
- "Limitations" sections in README
- Non-obvious code patterns
- Docstrings with practical advice

## Types of Heuristics

### 1. Optimization Tactics
- Performance tuning tips
- Memory management strategies
- Speed vs accuracy tradeoffs

### 2. Selection Frameworks (X vs Y)
- When to use one approach over another
- Decision criteria for configuration choices

### 3. Correction Tactics
- Common pitfalls and how to avoid them
- Debugging strategies
- Known issues and workarounds

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `applies_to::Implementation` | Implementation | Code this heuristic applies to |
| `applies_to::Workflow` | Workflow | Workflow this heuristic applies to |
| `related_to::Principle` | Principle | Related theoretical principle |

## Example Heuristics

- "Always set `lora_alpha = 2 * lora_r` for stable training"
- "Use 4-bit quantization over 8-bit for reasoning tasks"
- "Batch size should be a power of 2 for optimal GPU utilization"
- "Add warmup steps when fine-tuning pretrained models"

