# Composers (Dynamic Graph Structure)

Composers are intelligent functions that analyze the flat graph (Nodes + Edges) to generate higher-order structures like Hierarchies and Workflows. They create **Views** on top of the static data without altering the core executability constraints.

## 1. Hierarchy Composer (The Grouper)

**Goal:** Create a logical taxonomy of Principles (e.g., grouping `Adam`, `SGD` under `Optimization`).

### Logic
1.  **Scan:** Iterates through all **Principle** nodes.
2.  **Analyze:** Uses semantic analysis (keywords, embeddings, metadata) to identify siblings.
    *   *Example:* Finds `Adam`, `SGD`, `RMSProp`.
3.  **Synthesize:**
    *   Creates a new "Virtual" or "Abstract" **Principle Node** (e.g., `Optimization_Algorithms`).
    *   **Constraint Check:** Checks if a Base Class implementation exists for this new node.
        *   *If Yes:* Links `Optimization_Algorithms` -> `implemented_by` -> `Base_Optimizer`.
        *   *If No:* Flags as "Pure Virtual" (caveat to executability rule).
4.  **Merge:**
    *   The new node conceptually "contains" or "groups" the children.
    *   Instead of adding edges, the view presents the new node as the parent entity that subsumes the children.

### Output
A hierarchical tree view where new Principle nodes act as containers/parents for existing ones.

---

## 2. Workflow Composer (The Sequencer)

**Goal:** Discover potential Workflows by chaining compatible Principles.

### Logic
1.  **Scan:** Iterates through **Principle** and **Implementation** nodes.
2.  **Analyze I/O:** Looks at the `I/O Contract` of Implementations.
    *   *Match:* `Output(Step A)` == `Input(Step B)`.
    *   *Example:* `DataLoader` produces `Batch` -> `Trainer` consumes `Batch`.
3.  **Synthesize:**
    *   Proposes a candidate **Workflow Node**.
    *   Sequences the Principles: `Step 1: Data_Loading` -> `Step 2: Training`.
4.  **Verify:** Checks path executability (Is there a valid Implementation chain?).

### Output
New **Workflow** suggestions or validation of existing Workflows.

