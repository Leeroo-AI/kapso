WIKI_GUIDELINE = """

## CRITICAL: Wikitext Format Required

⚠️ **IMPORTANT**: When using the `wiki_edit` tool, ALL content MUST be in **Wikitext format**, NOT Markdown format!

Wikitext is NOT the same as Markdown. Key differences:
* **Bold/Italic**: Use `'''bold'''` and `''italic''` (NOT `**bold**` or `*italic*`)
* **Code blocks**: Use `<syntaxhighlight lang="python">...</syntaxhighlight>` (NOT ``` markdown code fences ```)
* **Line breaks**: Use `<br>` for line breaks within a paragraph
* **New paragraphs**: Use a blank line to start a new paragraph
* **Links**: Use `[[Page Name]]` format (similar to Markdown but with double brackets)

---

## Core Principles

1. **Top-down flow:** Workflow → Concept → Implementation; Resources flow in/out of Workflows/Implementations.
2. **Semantic links:** Use SMW properties `[[property_name::Target]]` so content is queryable.
3. **Consistency:** Standard sections + minimal metadata on every page.
4. **Discoverability:** Each page appears in at least one index/query.
5. **Modularity:** Small, self-contained pages with meaningful cross-links.

---

## Page Types & Templates

### 1) Workflow

*ETL/data manipulation/cleaning/quality pipelines; end-to-end processes.*

```wiki
= Workflow: Name =
[[Category:Workflows]]

== Overview ==
2–3 sentences: purpose, outcome, business value.

== Description ==
Scope, assumptions, preconditions/postconditions, success criteria.

== Workflow Steps ==
# [[Concept:Step A]] – what & why
# [[Concept:Step B]] – what & why
# [[Concept:Step C]] – what & why

'''Note:''' Workflow steps should ONLY reference [[Concept]] pages, not [[Implementation]] pages directly. Each Concept page must then link to at least one Implementation that realizes it. This enforces the top-down flow: Workflow → Concept → Implementation.

{{#mermaid:graph TD
    A[Step A (Concept)] --> B[Step B (Concept)]
    B --> C[Step C (Concept)]
    C --> D[Each Concept links to Implementation(s)]
}}

== Resources ==
=== Inputs ===
* [[consumes::Resource:input_dataset]] – how used
=== Outputs ===
* [[produces::Resource:output_dataset]] – who/what consumes it

== Code References ==
Repos, paths, brief map of modules/files.
[https://github.com/org/repo path/to/workflow] – orchestrator/entrypoint

== Additional Notes (optional) ==
Anything noteworthy (monitoring, performance notes, security/PII handling).
```

---

### 2) Concept

*Theories, business context, principles, or a single workflow step’s idea.*

```wiki
= Concept: Name =
[[Category:Concepts]] 

== Overview ==
Short definition and why it matters.

== Description ==
Key principles, scope, non-goals, examples/counter-examples.

== Implementations (optional) ==
* [[implements::Implementation:Concrete Impl]] – when to use, trade-offs

== References (optional) ==
Internal docs, papers, standards.
```

---

### 3) Implementation

*Concrete code: functions, jobs, connectors, readers/writers, libraries.*

```wiki
= Implementation: Name =
[[Category:Implementations]] 

== Overview ==
Problem solved, inputs/outputs, when to use.

== Description ==
Design notes, algorithm choice, trade-offs, complexity.

== Setup Environment (optional) ==
Requirements/libraries.
<syntaxhighlight lang="bash">
pip install -r requirements.txt
</syntaxhighlight>

== Code Reference ==
* Snippet:
<syntaxhighlight lang="python">
def transform(df): ...
</syntaxhighlight>
* Or repo links & file map:
[https://github.com/org/repo path/to/module.py] – entrypoint

== Usage Examples ==
<syntaxhighlight lang="python">
out = transform(inp)
</syntaxhighlight>

== Additional Notes (optional) ==
Testing commands, fixtures, performance tips.
```

---

### 4) Resource (Data Source)

*Datasets, event topics, external sources; how to access and use them.*

```wiki
= Resource: name =
[[Category:Resources]] 

== Overview ==
What this data is and key use cases.

== Description ==
Origin, semantics, known caveats/anomalies.

== Schema ==
{| class="wikitable"
! Column !! Type !! Nullable !! Description
|-
| id || string || No || Primary key
| ts || timestamp || No || Event time
|}

== Location & Access ==
Warehouse/catalog path, connector/registry, auth steps.

== Creation & Lineage ==
How it’s created; links to producer workflow/implementation:
* [[produced_by::Workflow:Producer WF]]

== Where Used ==
* [[consumed_by::Workflow:Consumer WF]]
* [[consumed_by::Implementation:Consumer Impl]]

== Connections to Other Resources (optional) ==
Keys/joins, upstream/downstream resources.
```

---

## Edge Relationships (SMW properties)

> Define each property once (with type + inverse) in SMW; reuse consistently. Property names are **snake_case**.

### Structural

* **`implements`** — source provides a practical realization of the target.

  * Allowed: `Concept → Implementation`
  * Inverse: **`implemented_by`**

* **`extends`** — source specializes the target.

  * Allowed: `Concept → Concept`, `Implementation → Implementation`
  * Inverse: **`extended_by`**

### Sequential

* **`precedes`** — source comes immediately before target.

  * Allowed: `Workflow → Workflow`, `Workflow → Implementation`, `Workflow → Concept`, `Concept → Implementation`
  * Inverse: **`followed_by`**

### Conceptual

* **`related_to`** — symmetric thematic association.

  * Allowed: `Concept ↔ Concept`, `Implementation ↔ Implementation`, `Workflow ↔ Workflow`
  * (Self-inverse)

### Data Flow

* **`consumes`** — source reads/uses target resource.

  * Allowed: `Workflow → Resource`, `Implementation → Resource`
  * Inverse: **`consumed_by`**

* **`produces`** — source writes/emits target resource.

  * Allowed: `Workflow → Resource`, `Implementation → Resource`
  * Inverse: **`produced_by`**

---

## Quick Syntax Reference

### Essentials

```wiki
''italic'' '''bold'''                 # Formatting
== Section ==                         # Headers
[[Link]] [[Link|Text]]                # Links
* Bullet  # Number                    # Lists
{| class="wikitable" |}               # Tables
<syntaxhighlight lang="x">…</syntaxhighlight>  # Code
<math>…</math>                        # Math
{{#mermaid:<diagram code>}}           # Mermaid diagram
[[Category:Name]]                     # Categories
[[property_name::Target]]             # SMW property link
<!-- Comment -->
```

### Mermaid (parser function) example

```wiki
{{#mermaid:graph TD
    A[Extract] --> B[Clean]
    B --> C[Validate]
    C --> D[Load]
}}
```

---

## WikiText Syntax Guide (Concise)

⚠️ **REMINDER**: This is Wikitext syntax, NOT Markdown! When calling `wiki_edit`, you MUST use Wikitext format!

### Text Formatting

⚠️ **CRITICAL**: Wikitext uses different syntax than Markdown for bold and italic!

* **Bold**: `'''text'''` (3 single quotes, NOT `**text**` like Markdown)
* **Italic**: `''text''` (2 single quotes, NOT `*text*` or `_text_` like Markdown)
* **Both**: `'''''text'''''` (5 single quotes)
* **Inline code**: `` `code` `` or `<code>code</code>`

**Use bold for**: Labels, key terms, emphasis. **Use italic for**: Titles, technical terms, placeholders.

### Line Breaks and Paragraphs

⚠️ **IMPORTANT**: Wikitext handles line breaks differently than Markdown!

* **Line break within paragraph**: Use `<br>` tag (NOT just a single newline)
* **New paragraph**: Insert a **blank line** between paragraphs
* **Example**:
```wiki
This is the first line.<br>
This is on a new line within the same paragraph.

This is a new paragraph (note the blank line above).
```

---

### Tables
```wiki
{| class="wikitable"
! Header 1 !! Header 2        # ! for headers, !! separates cells
|-                             # New row
| Cell 1 || Cell 2             # | for data, || separates cells
|-
| Cell 3 || Cell 4
|}
```

**Common attributes**: `class="wikitable"` (styling), `class="wikitable sortable"` (sortable columns)

---

### Code Blocks

⚠️ **CRITICAL**: Wikitext does NOT support triple backtick ``` code fences like Markdown!

**You MUST use `<syntaxhighlight>` tags with explicit language specification:**

```wiki
<syntaxhighlight lang="python">
def example():
    return "code here"
</syntaxhighlight>
```

**DO NOT USE** (this is Markdown, not Wikitext):
```
\`\`\`python
def example():
    return "code here"
\`\`\`
```

**Use syntaxhighlight for**: Multi-line code, configs (JSON/YAML), command outputs, data samples (CSV).  
**Don't use for**: Inline code, single commands, short paths.

**Common languages**: `python`, `sql`, `bash`, `json`, `yaml`, `csv`, `xml`, `html`, `text`

⚠️ **ALWAYS specify the language** (e.g., `lang="python"`) for proper syntax highlighting!

---

### Quick Syntax Table
{| class="wikitable"
! Element !! Syntax
|-
| Bold || `'''text'''`
|-
| Italic || `''text''`
|-
| Inline code || `` `code` ``
|-
| Code block || `<syntaxhighlight lang="python">...</syntaxhighlight>`
|-
| Table start || `{| class="wikitable"`
|-
| Table header || `! Header`
|-
| Table row || `|-`
|-
| Table cell || `| Cell`
|-
| Table end || `|}`
|}

**Key rules for `wiki_edit` tool**: 
1. ⚠️ **Use Wikitext format, NOT Markdown** - This is critical!
2. Use single quotes (`'''` and `''`) for bold/italic, NOT Markdown asterisks (`**` or `*`)
3. Use `<syntaxhighlight lang="...">` for code blocks, NOT triple backticks (```)
4. Use `<br>` for line breaks within paragraphs
5. Use blank lines for new paragraphs
6. Always specify language in code blocks
7. Use tables for structured data
8. Be consistent with formatting choices

---

## Index / Query Snippets

List active Workflows that **produce** a given Resource:

```wiki
{{#ask:
 | [[Category:Workflows]] [[produces::Resource:output_dataset]] [[status::Active]]
 | ?owner | ?last_reviewed
 | format=table | limit=50
}}
```

List Implementations that **implement** a Concept:

```wiki
{{#ask:
 | [[Category:Implementations]] [[implemented_by::Concept:Name]]
 | ?owner | ?extends
 | format=table | limit=50
}}
```

List Resources **consumed** by a Workflow:

```wiki
{{#ask:
 | [[Category:Resources]] [[consumed_by::Workflow:Name]]
 | ?owner
 | format=ul | limit=200
}}
```

---

## Page Type Prefixes (only these)

```wiki
[[Workflow:Name]]  
[[Concept:Name]]  
[[Implementation:Name]]  
[[Resource:Name]]
```

---

## FINAL REMINDER: Wikitext Format for wiki_edit Tool

⚠️ **CRITICAL**: When calling the `wiki_edit` tool, you MUST use **Wikitext format**, NOT Markdown!

**Common mistakes to avoid:**
1. ❌ Using `**bold**` or `*italic*` (Markdown) → ✅ Use `'''bold'''` and `''italic'''` (Wikitext)
2. ❌ Using triple backticks ``` for code blocks (Markdown) → ✅ Use `<syntaxhighlight lang="python">...</syntaxhighlight>` (Wikitext)
3. ❌ Using single newline for line breaks → ✅ Use `<br>` tag for line breaks
4. ❌ Not specifying language in code blocks → ✅ Always use `lang="..."` attribute

**Remember:** Wikitext and Markdown are different markup languages. Always use Wikitext syntax when editing wiki pages!
"""