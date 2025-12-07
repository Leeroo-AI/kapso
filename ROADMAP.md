## Roadmap

This roadmap outlines the planned development layers for Leeroo LABS.

### 1. Experimentation Observability

| Feature | Description |
|---------|-------------|
| **Tree Visualization** | Real-time visualization of the experimentation tree while user runs a task |
| **Checkpointing** | Save and restore experiment states for resumability and reproducibility |
| **Modularity & Extensibility** | Clear extension points for community contributions (search strategies, evaluation methods) |

### 2. Knowledge Graph (KG)

| Feature | Description |
|---------|-------------|
| **Knowledge Acquisition** | Multiple ingestion pathways: backward pass from experimentation, deep web research, repository ingestion, papers, documentation |
| **KG Serving** | CI/CD pipeline for self-serving KGs + Leeroo hosted version |
| **Search & Indexing** | Efficient search and indexing mechanisms for the knowledge graph |
| **Context Manager Integration** | Connection of KG to the context manager for intelligent retrieval |
| **KG Structure Customization** | User flow for modifying and extending the KG schema |

### 3. Inference Infrastructure

| Feature | Description |
|---------|-------------|
| **Leeroo API** | Hosted API to run the complete inference infrastructure |

### 4. Distribution & Integrations

| Feature | Description |
|---------|-------------|
| **MCP Protocol for KG** | Model Context Protocol support for KG connection |
| **MCP Protocol for Agent** | MCP support for the complete agent (with optional infra) |
| **Blueprint Integrations** | Ready-to-use blueprints and use-case templates |
| **Expert API** | High-level API for the "From Brain to Binary" workflow (implemented in `src/expert.py`) |

---