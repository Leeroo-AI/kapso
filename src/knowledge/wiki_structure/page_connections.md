# Page Connections (Edges)

This document defines all semantic relationships (edges) between wiki pages in the 7-Node Knowledge Graph.

---

## Overview

Connections use **Semantic MediaWiki property annotations** with the syntax:

```
[[property::Namespace:repo_name/Page_Name]]
```

Where:
- `property` - The relationship type (e.g., `step`, `consumes`, `produces`)
- `Namespace` - The target page type (e.g., `Principle`, `Artifact`, `Implementation`)
- `repo_name` - Repository identifier in format `{owner}_{repo}`
- `Page_Name` - The target page name (underscores for spaces)

---

## Complete Edge Registry

### Summary Table

| Edge Property | Source Node | Target Node | Description |
|---------------|-------------|-------------|-------------|
| `step` | Workflow | Principle | A step in the workflow |
| `consumes` | Workflow, Implementation | Artifact | Input data consumed |
| `produces` | Workflow, Implementation | Artifact | Output data produced |
| `realized_by` | Principle | Implementation | Code that implements this principle |
| `related_to` | Principle, Heuristic | Principle | Related theoretical concept |
| `step_of` | Principle | Workflow | Workflow that uses this principle |
| `implements` | Implementation | Principle | Theory this code implements |
| `requires_env` | Implementation | Environment | Required environment |
| `used_in` | Implementation | Workflow | Workflow using this implementation |
| `required_by` | Environment | Implementation, Workflow | What requires this environment |
| `conforms_to` | Artifact | Principle | Theory this artifact conforms to |
| `consumed_by` | Artifact | Implementation | Code that consumes this artifact |
| `produced_by` | Artifact | Implementation | Code that produces this artifact |
| `applies_to` | Heuristic | Implementation, Workflow | What this heuristic applies to |

---

## Edges by Source Node Type

### 1. Workflow Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `step` | Principle | `[[step::Principle:repo_name/X]]` | Links workflow step to its theoretical basis |
| `consumes` | Artifact | `[[consumes::Artifact:repo_name/X]]` | Input data the workflow consumes |
| `produces` | Artifact | `[[produces::Artifact:repo_name/X]]` | Output data the workflow produces |

**Example:**
```mediawiki
== Execution Steps ==
=== Step 1: Data Loading ===
[[step::Principle:myorg_myrepo/Data_Loading]]

=== Step 2: Model Training ===
[[step::Principle:myorg_myrepo/Model_Training]]

== Data Flow ==
* **Input:** [[consumes::Artifact:myorg_myrepo/Training_Config]]
* **Output:** [[produces::Artifact:myorg_myrepo/Trained_Model]]
```

---

### 2. Environment Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `required_by` | Implementation | `[[required_by::Implementation:repo_name/X]]` | Implementations requiring this environment |
| `required_by` | Workflow | `[[required_by::Workflow:repo_name/X]]` | Workflows requiring this environment |

**Example:**
```mediawiki
== Related Pages ==
* [[required_by::Implementation:myorg_myrepo/ModelTrainer]] - Requires GPU environment
* [[required_by::Workflow:myorg_myrepo/Training_Pipeline]] - Requires CUDA setup
```

---

### 3. Principle Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `realized_by` | Implementation | `[[realized_by::Implementation:repo_name/X]]` | Code that implements this principle |
| `related_to` | Principle | `[[related_to::Principle:repo_name/X]]` | Related theoretical principle |
| `step_of` | Workflow | `[[step_of::Workflow:repo_name/X]]` | Workflow that uses this principle |

**Example:**
```mediawiki
== Related Implementation ==
* [[realized_by::Implementation:myorg_myrepo/TransformerEncoder]] - Primary implementation
* [[realized_by::Implementation:myorg_myrepo/AttentionModule]] - Core attention logic

== Related Pages ==
* [[related_to::Principle:myorg_myrepo/Self_Attention]] - Parent theory
* [[step_of::Workflow:myorg_myrepo/Text_Generation]] - Used in this workflow
```

---

### 4. Implementation Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `implements` | Principle | `[[implements::Principle:repo_name/X]]` | Theoretical principle this implements |
| `requires_env` | Environment | `[[requires_env::Environment:repo_name/X]]` | Required environment setup |
| `consumes` | Artifact | `[[consumes::Artifact:repo_name/X]]` | Input artifact consumed |
| `produces` | Artifact | `[[produces::Artifact:repo_name/X]]` | Output artifact produced |
| `used_in` | Workflow | `[[used_in::Workflow:repo_name/X]]` | Workflow that uses this implementation |

**Example:**
```mediawiki
== Related Pages ==
* [[implements::Principle:myorg_myrepo/Gradient_Descent]] - Theoretical basis
* [[requires_env::Environment:myorg_myrepo/GPU_Setup]] - Required environment
* [[consumes::Artifact:myorg_myrepo/Model_Config]] - Input configuration
* [[produces::Artifact:myorg_myrepo/Model_Weights]] - Output weights
* [[used_in::Workflow:myorg_myrepo/Model_Training]] - Part of training workflow
```

---

### 5. Artifact Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `conforms_to` | Principle | `[[conforms_to::Principle:repo_name/X]]` | Theoretical principle this artifact conforms to |
| `consumed_by` | Implementation | `[[consumed_by::Implementation:repo_name/X]]` | Implementations that consume this artifact |
| `produced_by` | Implementation | `[[produced_by::Implementation:repo_name/X]]` | Implementations that produce this artifact |

**Example:**
```mediawiki
== Related Pages ==
* [[conforms_to::Principle:myorg_myrepo/Configuration_Schema]] - Schema theory
* [[consumed_by::Implementation:myorg_myrepo/ModelTrainer]] - Used by trainer
* [[produced_by::Implementation:myorg_myrepo/ConfigLoader]] - Created by loader
```

---

### 6. Heuristic Edges

| Edge | Target | Syntax | Description |
|------|--------|--------|-------------|
| `applies_to` | Implementation | `[[applies_to::Implementation:repo_name/X]]` | Implementation this heuristic applies to |
| `applies_to` | Workflow | `[[applies_to::Workflow:repo_name/X]]` | Workflow this heuristic applies to |
| `related_to` | Principle | `[[related_to::Principle:repo_name/X]]` | Related theoretical principle |

**Example:**
```mediawiki
== Related Pages ==
* [[applies_to::Implementation:myorg_myrepo/LoRAAdapter]] - Applies to this tool
* [[applies_to::Workflow:myorg_myrepo/Fine_Tuning]] - Applies to this workflow
* [[related_to::Principle:myorg_myrepo/Low_Rank_Adaptation]] - Related theory
```

---

## Inverse Relationships

Some edges have logical inverse relationships:

| Edge | Inverse Edge | Meaning |
|------|--------------|---------|
| `step` | `step_of` | Workflow → Principle ↔ Principle → Workflow |
| `consumes` | `consumed_by` | Implementation → Artifact ↔ Artifact → Implementation |
| `produces` | `produced_by` | Implementation → Artifact ↔ Artifact → Implementation |
| `realized_by` | `implements` | Principle → Implementation ↔ Implementation → Principle |
| `requires_env` | `required_by` | Implementation → Environment ↔ Environment → Implementation |

---

## Link Integrity Rules

### Rule 1: No Dangling Links

Every link MUST have a corresponding page. Before creating a link, verify the target page exists.

```
[[step::Principle:repo_name/X]] → requires Principle_X.mediawiki
[[consumes::Artifact:repo_name/X]] → requires Artifact_X.mediawiki
[[produces::Artifact:repo_name/X]] → requires Artifact_X.mediawiki
[[realized_by::Implementation:repo_name/X]] → requires Implementation_X.mediawiki
[[related_to::Principle:repo_name/X]] → requires Principle_X.mediawiki
[[step_of::Workflow:repo_name/X]] → requires Workflow_X.mediawiki
```

### Rule 2: Workflow Executability

Every Principle used as a workflow step (`[[step::Principle:repo_name/X]]`) **MUST** have at least one corresponding Implementation linked via `[[realized_by::Implementation:repo_name/Y]]`.

This ensures:
- **Principle** explains the theory (WHY)
- **Implementation** provides the actual code (HOW)

### Rule 3: Fully-Qualified Links

All links must include the complete path:

```mediawiki
# ✅ CORRECT
[[step::Principle:myorg_myrepo/Tokenization]]

# ❌ WRONG - Missing repo_name
[[step::Principle:Tokenization]]
```

---

## Validation Checklist

Before completing page generation, verify:

- [ ] Every `[[step::Principle:repo_name/X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[consumes::Artifact:repo_name/X]]` has an `Artifact_X.mediawiki`
- [ ] Every `[[produces::Artifact:repo_name/X]]` has an `Artifact_X.mediawiki`
- [ ] Every `[[realized_by::Implementation:repo_name/X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[related_to::Principle:repo_name/X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[step_of::Workflow:repo_name/X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[implements::Principle:repo_name/X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[requires_env::Environment:repo_name/X]]` has an `Environment_X.mediawiki`
- [ ] Every `[[used_in::Workflow:repo_name/X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[conforms_to::Principle:repo_name/X]]` has a `Principle_X.mediawiki`
- [ ] Every `[[consumed_by::Implementation:repo_name/X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[produced_by::Implementation:repo_name/X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[applies_to::Implementation:repo_name/X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[applies_to::Workflow:repo_name/X]]` has a `Workflow_X.mediawiki`
- [ ] Every `[[required_by::Implementation:repo_name/X]]` has an `Implementation_X.mediawiki`
- [ ] Every `[[required_by::Workflow:repo_name/X]]` has a `Workflow_X.mediawiki`

---

## Graph Visualization

The knowledge graph forms this structure:

```
                    ┌─────────────┐
                    │  Resource   │ (Entry Point)
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │Environment│   │  Workflow │   │ Heuristic │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          │    step       │    applies_to │
          │    consumes   │               │
          │    produces   │               │
          │               ▼               │
          │        ┌───────────┐          │
          │        │ Principle │◄─────────┤
          │        └─────┬─────┘          │
          │              │                │
          │   realized_by│                │
          │              ▼                │
          │       ┌─────────────┐         │
          └──────►│Implementation│◄───────┘
    required_by   └──────┬──────┘  applies_to
                         │
              consumes   │   produces
                         ▼
                  ┌───────────┐
                  │  Artifact │
                  └───────────┘
```

---

## Cross-Repository Links

When referencing pages from a different repository, use that repo's name:

```mediawiki
# From a page in myorg_myrepo, linking to huggingface TGI:
See also: [[Workflow:huggingface_text-generation-inference/Model_Loading]]
```

This enables knowledge sharing across repositories while maintaining link integrity.

