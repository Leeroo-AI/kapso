= Separator Configuration =

'''Sources:''' /tmp/praxium_repo_wjjl6pl8/libs/text-splitters/langchain_text_splitters/character.py:L171-793

'''Domains:''' Text Splitting, RAG, Code Processing, Natural Language Processing

'''Last Updated:''' 2025-12-17

== Overview ==

Separator Configuration is the principle of defining appropriate boundaries for splitting text based on the content's structure and semantics. Different types of content have different natural boundaries - paragraphs for prose, functions for code, headers for markdown. Properly configured separators preserve semantic coherence while respecting size constraints.

== Description ==

Text splitting effectiveness depends critically on using separators that respect the natural structure of the content. The separator configuration determines:

'''What gets split:''' Which patterns trigger chunk boundaries
'''What gets preserved:''' Whether semantic units remain intact
'''How context flows:''' Whether related content stays together

LangChain provides three approaches to separator configuration:

'''Default Separators:''' RecursiveCharacterTextSplitter uses hierarchical defaults:
* "\n\n" (paragraphs) - highest priority
* "\n" (lines)
* " " (words)
* "" (characters) - last resort

'''Language-specific Separators:''' Pre-configured patterns for 25+ programming languages that respect syntactic structure:
* Class and function definitions
* Control flow statements (if/for/while)
* Import and declaration statements
* Language-specific constructs

'''Custom Separators:''' User-defined patterns for specialized content:
* Domain-specific formats
* Structured data patterns
* Custom markup languages

== Theoretical Basis ==

The effectiveness of separator configuration relies on understanding content structure and semantic boundaries:

'''Hierarchical Splitting Strategy:'''

The RecursiveCharacterTextSplitter implements a preference-ordered approach:
1. Try the most semantic separator first (e.g., paragraph breaks)
2. If chunks are too large, recursively try more granular separators
3. Continue until chunks meet size requirements

This preserves maximum semantic coherence while respecting constraints.

'''Syntax-aware Boundaries:'''

Programming languages have syntactic structure that must be preserved:
* Function boundaries keep implementation logic together
* Class definitions keep related methods grouped
* Import blocks remain intact
* Control flow structures stay coherent

Breaking these boundaries produces chunks that:
* Lack necessary context for understanding
* Split declarations from implementations
* Separate related logic across chunks

'''Language-specific Patterns:'''

Each programming language has unique structural elements:

'''Python:''' Indentation-based, prioritizes class and def
* "\nclass " - class definitions
* "\ndef " - top-level functions
* "\n\tdef " - class methods
* "\n\n" - blank lines between blocks

'''JavaScript/TypeScript:''' Function and declaration focus
* "\nfunction " - function declarations
* "\nconst " / "\nlet " / "\nvar " - variable declarations
* "\nclass " - class definitions
* "\nif " / "\nfor " / "\nwhile " - control flow

'''Markup Languages:''' Heading-based hierarchy
* Markdown: "## " headings, code blocks "```\n"
* HTML: structural tags (<div>, <p>, <h1>-<h6>)
* LaTeX: sections, environments, math delimiters

'''Regex vs Literal Separators:'''

Separators can be:
* '''Literal strings:''' Exact matches, escaped for regex
* '''Regex patterns:''' Flexible matching with patterns

Regex separators enable:
* Multiple variations (e.g., "\n\n+" matches multiple blank lines)
* Whitespace flexibility
* Optional components
* Lookahead/lookbehind for context-sensitive splits

'''Separator Preservation:'''

Three options for handling separators in output:
* '''Discard:''' Remove separators (may lose structure cues)
* '''Keep at start:''' Preserve separators at chunk beginning
* '''Keep at end:''' Preserve separators at chunk end

Keeping separators helps:
* Maintain visual structure
* Preserve formatting cues
* Retain contextual markers (e.g., "def " shows it's a function)

'''Common Patterns by Content Type:'''

'''Natural Language:'''
* Prioritize paragraph breaks (\n\n)
* Fall back to sentence boundaries (. )
* Use line breaks for poetry/lists
* Character splitting as last resort

'''Code:'''
* Start with definitions (class/function)
* Split on control flow structures
* Respect block boundaries
* Preserve semantic units

'''Structured Documents:'''
* Use heading hierarchy
* Respect section boundaries
* Keep related subsections together
* Maintain document outline

'''Avoiding Anti-patterns:'''

Poor separator choices:
* Splitting mid-sentence (breaks semantic meaning)
* Breaking function implementations (loses context)
* Separating declarations from usage (incomplete information)
* Ignoring indentation in Python (breaks scope understanding)

Good separator choices:
* Natural semantic boundaries
* Syntax-respecting patterns
* Hierarchical preference ordering
* Context-preserving splits

== Related Pages ==

* [[langchain-ai_langchain_separator_config|separator_config]] - Implementation of separator configuration
* [[langchain-ai_langchain_Splitter_Selection|Splitter_Selection]] - Choosing appropriate text splitters
* [[langchain-ai_langchain_Chunk_Configuration|Chunk_Configuration]] - Configuring chunk size and overlap
* [[langchain-ai_langchain_text_splitter_types|text_splitter_types]] - Text splitter implementations
