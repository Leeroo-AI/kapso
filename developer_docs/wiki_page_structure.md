# Wiki Page Structure

Overview of the 5 page types and their connections in `src/knowledge/wiki_structure/`.

---

## Page Types

| Type | Role | Description |
|------|------|-------------|
| **Workflow** | The Recipe | Ordered sequence of steps to achieve a goal (e.g., "Fine-tune Llama-2") |
| **Principle** | The Theory | Library-agnostic concept explaining *what* and *why* (e.g., "LoRA Adaptation") |
| **Implementation** | The Code | Concrete API/function with exact syntax (e.g., "HuggingFace PEFT LoraConfig") |
| **Environment** | The Context | Hardware, OS, dependencies required to run (e.g., "CUDA 12.1 + 24GB VRAM") |
| **Heuristic** | The Wisdom | Tips, tricks, optimizations, debugging advice (e.g., "Optimal LoRA rank selection") |

---

## Graph Structure (Top-Down DAG)

```
┌──────────────────────────────────────────────────────────────┐
│                         WORKFLOW                             │
│                      (Entry Point)                           │
│                                                              │
│    "Fine-tune LLM with QLoRA"                                │
└──────────────────────────────────────────────────────────────┘
         │ step                              │ uses_heuristic
         ▼                                   ▼
┌─────────────────────┐             ┌─────────────────────┐
│     PRINCIPLE       │             │     HEURISTIC       │
│   (Core Theory)     │────────────▶│     (Wisdom)        │
│                     │uses_heuristic                     │
│ "Low-Rank Adapt."   │             │ "Rank 16 for most"  │
└─────────────────────┘             └─────────────────────┘
         │ implemented_by                    ▲
         ▼                                   │ uses_heuristic
┌─────────────────────┐                      │
│  IMPLEMENTATION     │──────────────────────┘
│    (The Code)       │
│                     │
│ "peft.LoraConfig"   │
└─────────────────────┘
         │ requires_env
         ▼
┌─────────────────────┐
│    ENVIRONMENT      │
│    (Context)        │
│                     │
│ "CUDA 12, 16GB RAM" │
└─────────────────────┘
```

---

## Edge Types

| From | Edge | To | Meaning |
|------|------|----|---------|
| Workflow | `step` | Principle | "Step X uses theory Y" |
| Workflow | `uses_heuristic` | Heuristic | "Guided by wisdom Z" |
| Principle | `implemented_by` | Implementation | "Realized by code X" (**mandatory**) |
| Principle | `uses_heuristic` | Heuristic | "Optimized by tip Z" |
| Implementation | `requires_env` | Environment | "Needs context X to run" |
| Implementation | `uses_heuristic` | Heuristic | "Has config hack Z" |

---

## Rules

1. **Top-Down Only** — Edges flow downward (Intent → Theory → Code). No upward links.
2. **No Loops** — Graph must remain a DAG.
3. **Executable Constraint** — Every Principle must have at least one `implemented_by` link.
4. **Leaf Nodes** — Environment and Heuristic are targets only (no outgoing edges).

---

## File Structure

```
src/knowledge/wiki_structure/
├── page_connections.md          # Edge definitions and rules
├── workflow_page/
│   ├── page_definition.md       # What a Workflow is
│   └── sections_definition.md   # Required sections
├── principle_page/
│   ├── page_definition.md
│   └── sections_definition.md
├── implementation_page/
│   ├── page_definition.md
│   └── sections_definition.md
├── environment_page/
│   ├── page_definition.md
│   └── sections_definition.md
└── heuristic_page/
    ├── page_definition.md
    └── sections_definition.md
```

---

## Link Syntax in Wiki Pages

```mediawiki
[[step::Principle:Data_Loading]]
[[implemented_by::Implementation:HF_Trainer]]
[[requires_env::Environment:CUDA_12]]
[[uses_heuristic::Heuristic:Batch_Size_Tips]]
```

