# Page Connections (Edges)

This document defines all semantic relationships (edges) between wiki pages in the 7-Node Knowledge Graph.

---

## Overview

Connections use **Semantic MediaWiki property annotations** with the syntax:

```
[[property::Namespace:Page_Name]]
```

Where:
- `property` - The relationship type (e.g., `step`, `implements`)
- `Namespace` - The target page type (e.g., `Principle`, `Implementation`)
- `Page_Name` - The target page name (underscores for spaces)

---

## Complete Edge Registry

### Summary Table

| Edge Property | Source Node | Target Node | Description |
|---------------|-------------|-------------|-------------|
| `step` | Workflow | Principle | A step in the workflow |
| `realized_by` | Principle | Implementation | Code that implements this principle |
| `related_to` | Principle, Heuristic | Principle | Related theoretical concept |
| `step_of` | Principle | Workflow | Workflow that uses this principle |
| `implements` | Implementation | Principle | Theory this code implements |
| `requires_env` | Implementation | Environment | Required environment |
| `used_in` | Implementation | Workflow | Workflow using this implementation |
| `required_by` | Environment | Implementation, Workflow | What requires this environment |
| `applies_to` | Heuristic | Implementation, Workflow | What this heuristic applies to |

---

## Edges by Source Node Type

### 1. Workflow Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `step` | Principle | `[[step::Principle:X]]` | Links workflow step to its theoretical basis |

**Example:**
```mediawiki
== Execution Steps ==
=== Step 1: Data Loading ===
[[step::Principle:Data_Loading]]

=== Step 2: Model Training ===
[[step::Principle:Model_Training]]
```

---

### 2. Environment Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `required_by` | Implementation | `[[required_by::Implementation:X]]` | Implementations requiring this environment |
| `required_by` | Workflow | `[[required_by::Workflow:X]]` | Workflows requiring this environment |

**Example:**
```mediawiki
== Related Pages ==
* [[required_by::Implementation:ModelTrainer]] - Requires GPU environment
* [[required_by::Workflow:Training_Pipeline]] - Requires CUDA setup
```

---

### 3. Principle Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `realized_by` | Implementation | `[[realized_by::Implementation:X]]` | Code that implements this principle |
| `related_to` | Principle | `[[related_to::Principle:X]]` | Related theoretical principle |
| `step_of` | Workflow | `[[step_of::Workflow:X]]` | Workflow that uses this principle |

**Example:**
```mediawiki
== Related Implementation ==
* [[realized_by::Implementation:TransformerEncoder]] - Primary implementation
* [[realized_by::Implementation:AttentionModule]] - Core attention logic

== Related Pages ==
* [[related_to::Principle:Self_Attention]] - Parent theory
* [[step_of::Workflow:Text_Generation]] - Used in this workflow
```

---

### 4. Implementation Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `implements` | Principle | `[[implements::Principle:X]]` | Theoretical principle this implements |
| `requires_env` | Environment | `[[requires_env::Environment:X]]` | Required environment setup |
| `used_in` | Workflow | `[[used_in::Workflow:X]]` | Workflow that uses this implementation |

**Example:**
```mediawiki
== Related Pages ==
* [[implements::Principle:Gradient_Descent]] - Theoretical basis
* [[requires_env::Environment:GPU_Setup]] - Required environment
* [[used_in::Workflow:Model_Training]] - Part of training workflow
```

---

### 5. Heuristic Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `applies_to` | Implementation | `[[applies_to::Implementation:X]]` | Implementation this heuristic applies to |
| `applies_to` | Workflow | `[[applies_to::Workflow:X]]` | Workflow this heuristic applies to |
| `related_to` | Principle | `[[related_to::Principle:X]]` | Related theoretical principle |

**Example:**
```mediawiki
== Related Pages ==
* [[applies_to::Implementation:LoRAAdapter]] - Applies to this tool
* [[applies_to::Workflow:Fine_Tuning]] - Applies to this workflow
* [[related_to::Principle:Low_Rank_Adaptation]] - Related theory
```

---

## Inverse Relationships

Some edges have logical inverse relationships:

| Edge | Inverse Edge | Meaning |
|------|--------------|---------|
| `step` | `step_of` | Workflow → Principle ↔ Principle → Workflow |
| `realized_by` | `implements` | Principle → Implementation ↔ Implementation → Principle |
| `requires_env` | `required_by` | Implementation → Environment ↔ Environment → Implementation |

---

## Link Integrity Rules

### Rule 1: No Dangling Links

Every link MUST have a corresponding page. Before creating a link, verify the target page exists.

```
[[step::Principle:X]] → requires Principle_X.mediawiki
[[realized_by::Implementation:X]] → requires Implementation_X.mediawiki
[[related_to::Principle:X]] → requires Principle_X.mediawiki
[[step_of::Workflow:X]] → requires Workflow_X.mediawiki
```

### Rule 2: Workflow Executability

Every Principle used as a workflow step (`[[step::Principle:X]]`) **MUST** have at least one corresponding Implementation linked via `[[realized_by::Implementation:Y]]`.

This ensures:
- **Principle** explains the theory (WHY)
- **Implementation** provides the actual code (HOW)

### Rule 3: Fully-Qualified Links

All links must include the namespace:

```mediawiki
# ✅ CORRECT
[[step::Principle:Tokenization]]

# ❌ WRONG - Missing namespace
[[step::Tokenization]]
```

---

## Validation Checklist

Before completing page generation, verify:

- [ ] Every `[[step::Principle:X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[realized_by::Implementation:X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[related_to::Principle:X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[step_of::Workflow:X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[implements::Principle:X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[requires_env::Environment:X]]` has an `Environment_X.mediawiki`
- [ ] Every `[[used_in::Workflow:X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[applies_to::Implementation:X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[applies_to::Workflow:X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[required_by::Implementation:X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[required_by::Workflow:X]]` has a `Workflow_X.mediawiki`

---

## Graph Visualization

The knowledge graph forms this structure:

```
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │Environment│   │  Workflow │   │ Heuristic │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  │    step       │    applies_to │
                  │               ▼               │
                  │        ┌───────────┐          │
                  │        │ Principle │◄─────────┤
                  │        └─────┬─────┘          │
                  │              │                │
                  │   realized_by│                │
                  │              ▼                │
                  │       ┌─────────────┐         │
                  └──────►│Implementation│◄───────┘
    required_by   └──────────────┘  applies_to
```

---

## Cross-Repository Links

When referencing pages from a different repository, use that repo's name:

```mediawiki
# From a page in myorg_myrepo, linking to huggingface TGI:
See also: [[Workflow:huggingface_text-generation-inference/Model_Loading]]
```

This enables knowledge sharing across repositories while maintaining link integrity.

