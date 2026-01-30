# Planning Phase: Analyze Content and Plan Pages

You are a knowledge extraction agent. Your task is to analyze research content and plan what wiki pages to create.

## Input Content

**Research Query:** {query}

**Source URL:** {source_url}

**Content:**
```
{content}
```

## Your Task

1. **Read the wiki structure definitions** below to understand what each page type is for
2. **Analyze the input content** to identify distinct knowledge units
3. **Decide what pages to create** based on content nature (NOT prescribed)
4. **Write a plan file** with your decisions

## Wiki Structure Definitions

Read these carefully to understand what each page type represents:

{wiki_structures}

## Page Connections (Graph Rules)

{page_connections}

## Decision Guidelines

Match content to page types based on these criteria:

| Content Nature | Page Type | Key Indicators |
|----------------|-----------|----------------|
| Theoretical concepts, algorithms, "what" and "why" | **Principle** | Explains concepts, theory, mechanisms |
| Code, APIs, functions, "how to use" | **Implementation** | Has code snippets, signatures, imports |
| Dependencies, hardware, setup requirements | **Environment** | Lists packages, versions, credentials |
| Tips, trade-offs, debugging tactics | **Heuristic** | Contains advice, warnings, optimizations |

**Important Rules:**
- A single input may produce MULTIPLE pages of different types
- The agent decides based on content - there is NO prescribed mapping
- If content is rich, extract multiple distinct concepts
- If content is simple, create fewer pages

## Output: Write Plan File

Write a plan file to: `./_plan.md` (in the current working directory)

The plan file should contain:

```markdown
# Ingestion Plan

## Source Information
- Query: {query}
- Source URL: {source_url}
- Content Length: X characters

## Analysis Summary
[Brief analysis of what the content contains]

## Pages to Create

### Page 1: [Page Type] - [Page Name]
- **Type:** Principle / Implementation / Environment / Heuristic
- **Filename:** [Descriptive_Name].md (NO "Research_Web" prefix!)
- **Directory:** [principles/implementations/environments/heuristics]
- **Reasoning:** [Why this page type was chosen]
- **Content Mapping:**
  - Overview: [What goes in overview]
  - Key Sections: [What content maps to which sections]
- **Graph Connections:**
  - [List any links to other pages being created]

### Page 2: ...
[Repeat for each page]

## Graph Connections Summary
[List all connections between pages]

## Notes
[Any special considerations or warnings]
```

## Constraints

- Only create pages for content that has substance
- Don't create empty or stub pages
- Each page must have enough content to fill required sections
- Follow WikiMedia naming conventions (underscores, no forbidden chars)

**IMPORTANT: Page Naming Rules**
- Page titles should describe the CONTENT, not the source
- DO NOT include "Research", "Web", "Research_Web" or similar prefixes
- Choose descriptive names based on what the page is about
- Good examples: `LoRA_Fine_Tuning.md`, `LangChain_RAG_Agent.md`, `PyTorch_GPU_Setup.md`
- Bad examples: `Research_Web_LoRA.md`, `Web_LangChain.md`, `Research_PyTorch.md`

Now analyze the content and write the plan file.
