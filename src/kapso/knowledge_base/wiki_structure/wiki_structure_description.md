# Knowledge Base Structure

A top-down DAG of 5 page types covering ML/AI topics.

## Page Types

| Type | Role | Description |
|------|------|-------------|
| **Workflow** | Recipe | End-to-end process (e.g., "QLoRA Fine-tuning"). Links to a GitHub repo with step implementations. |
| **Principle** | Theory | Library-agnostic concept (e.g., "Self-Attention", "LoRA"). Must link to >=1 Implementation. |
| **Implementation** | Code | Concrete API/class/function with signatures, I/O contracts, usage examples. Principle-conditioned. |
| **Environment** | Context | Hardware, OS, dependencies, credentials needed to run code. |
| **Heuristic** | Wisdom | Tips, rules of thumb, optimization tricks, debugging tactics. |

## Graph Edges (top-down only)

- Principle —[implemented_by]→ Implementation (mandatory, 1:1)
- Principle —[uses_heuristic]→ Heuristic
- Implementation —[requires_env]→ Environment
- Implementation —[uses_heuristic]→ Heuristic
- Workflow links to external GitHub repo (no wiki edges)

## Page ID Format

`{PageType}/{Page_Name}` — e.g., `Principle/LoRA_Configuration`, `Heuristic/Learning_Rate_Warmup`

## Semantic Links (inside page content)

`[[edge_type::TargetType:Target_Name]]` — e.g., `[[implemented_by::Implementation:Unsloth_LoRA_Config]]`
