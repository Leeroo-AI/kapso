# The referencing language's mechanical side — ref extraction.
#
# Design: learn-from-trajectories-design.md §3.2.2 (markdown links with an
# optional label) and the mined-format/report conventions: refs appear either
# as full markdown links `[label](path#fragment)` or as bare bracket refs
# `[path#fragment]` (the style the flow docs and hindcast reports use inline).
# A bare bracket ref must look like a path (contain a slash, no spaces) so
# card refs `[insight: name]` and prose brackets never parse as paths.

import re
from typing import List

_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
# The negative lookahead keeps markdown-link LABELS out: in
# `[Qwen/Qwen2.5-7B](https://...)` the bracket is a label, not a ref —
# only the (...) target is (matched by _MD_LINK above).
_BARE_BRACKET_REF = re.compile(
    r"\[([A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+(?:#[^\[\]]*)?)\](?!\()"
)


def extract_refs(text: str) -> List[str]:
    """Every path-shaped ref in the text: markdown-link targets plus bare
    bracket refs. Order preserved, duplicates kept (callers dedupe)."""
    refs = _MD_LINK.findall(text)
    for candidate in _BARE_BRACKET_REF.findall(text):
        refs.append(candidate)
    return refs
