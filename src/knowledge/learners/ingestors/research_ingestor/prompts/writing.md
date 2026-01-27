# Writing Phase: Create Wiki Pages

You are a knowledge extraction agent. Your task is to create wiki pages following the plan from the planning phase.

## Context

**Research Query:** {query}

**Source URL:** {source_url}

**Content:**
```
{content}
```

## Your Task

1. **Read the plan file** at `{plan_path}`
2. **For each planned page**, create the wiki page following the sections definition
3. **Write pages** to the correct subdirectories
4. **Establish graph connections** using semantic wiki links

## Plan File

Read the plan at: `{plan_path}`

## Wiki Structure Definitions

Use these to write properly structured pages:

{wiki_structures}

## Page Connections (Graph Rules)

{page_connections}

## Writing Instructions

For each page in the plan:

### 1. Create the Page File

Write to: `{wiki_dir}/[subdirectory]/[filename].md`

Subdirectories:
- Principle pages → `{wiki_dir}/principles/`
- Implementation pages → `{wiki_dir}/implementations/`
- Environment pages → `{wiki_dir}/environments/`
- Heuristic pages → `{wiki_dir}/heuristics/`

### 2. Follow the Page Structure

Each page MUST include ALL required sections from the sections_definition.md.

**Common Structure:**

```mediawiki
# [Type]: [Page_Name]

{{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
||
* [[source::Web|Research|{source_url}]]
|-
! Domains
|| [[domain::Research]], [[domain::[Relevant_Domain]]]
|-
! Last Updated
|| [[last_updated::{timestamp}]]
|}}

== Overview ==
[One sentence summary]

=== Description ===
[Detailed explanation]

=== Usage ===
[When to use this]

[... additional sections based on page type ...]

== Related Pages ==
[Graph connections using semantic wiki links]
```

### 3. Establish Graph Connections

Use semantic wiki links in the `== Related Pages ==` section:

**For Principle pages:**
```mediawiki
== Related Pages ==
* [[implemented_by::Implementation:Research_Web_[Slug]]]
* [[uses_heuristic::Heuristic:Research_Web_[Slug]]]
```

**For Implementation pages:**
```mediawiki
== Related Pages ==
* [[requires_env::Environment:Research_Web_[Slug]]]
* [[uses_heuristic::Heuristic:Research_Web_[Slug]]]
```

**For Environment pages:**
```mediawiki
== Related Pages ==
* [[required_by::Implementation:Research_Web_[Slug]]]
```

**For Heuristic pages:**
```mediawiki
== Related Pages ==
* [[used_by::Implementation:Research_Web_[Slug]]]
* [[used_by::Principle:Research_Web_[Slug]]]
```

### 4. WikiMedia Naming Rules

- First character capitalized
- Underscores only (NO hyphens, NO spaces)
- Forbidden characters: `# < > [ ] {{ }} | + : /`
- Format: `Research_Web_[Descriptive_Name].md`

## Quality Checklist

Before finishing each page, verify:
- [ ] Page title is on first line: `# [Type]: [Name]`
- [ ] Metadata block is present and formatted correctly
- [ ] All required sections from sections_definition.md are included
- [ ] Content is substantive (not placeholder text)
- [ ] Graph connections are established in Related Pages section
- [ ] Filename follows WikiMedia conventions

## Output

Create all pages specified in the plan file. Write each page to the correct subdirectory.

After creating all pages, print a summary:
```
WRITING COMPLETE
================
Pages created:
- [Type]: [Filename] → [Path]
- ...

Graph connections established:
- [Source] → [Target] (edge type)
- ...
```

Now read the plan and create the wiki pages.
