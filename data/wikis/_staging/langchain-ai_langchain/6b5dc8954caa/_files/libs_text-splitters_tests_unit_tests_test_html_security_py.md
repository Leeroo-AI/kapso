# File: `libs/text-splitters/tests/unit_tests/test_html_security.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 130 |
| Classes | `TestHTMLSectionSplitterSecurity` |
| Imports | langchain_text_splitters, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security test suite preventing XXE (XML External Entity) attacks in HTML splitters.

**Mechanism:** Tests that HTMLSectionSplitter blocks XXE entity attacks, XSLT document() function attacks, network access attempts, DTD processing, and entity expansion. Validates secure parser configuration and safe default XSLT usage. Verifies malicious XML/HTML content cannot access local files (/etc/passwd) or make external requests (attacker.com).

**Significance:** Critical security validation ensuring HTML processing is hardened against XML injection attacks. Protects users from malicious documents that attempt file disclosure or SSRF attacks through XML entities and XSLT functions.
