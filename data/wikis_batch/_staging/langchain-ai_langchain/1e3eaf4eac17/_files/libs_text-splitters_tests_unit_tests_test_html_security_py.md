# File: `libs/text-splitters/tests/unit_tests/test_html_security.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 130 |
| Classes | `TestHTMLSectionSplitterSecurity` |
| Imports | langchain_text_splitters, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security-focused test suite that validates HTMLSectionSplitter is protected against XML External Entity (XXE) attacks and related injection vulnerabilities.

**Mechanism:** Tests that external entity attacks are blocked, document() function attacks are prevented, network access is disabled, DTD processing is disabled, and the default XSLT is used securely. Validates that malicious HTML with external entity references, DTD declarations, and network-based attacks cannot extract sensitive data or execute arbitrary code. All tests require lxml and bs4 dependencies.

**Significance:** Critical security test suite that prevents XXE vulnerabilities which could lead to file disclosure (e.g., /etc/passwd), SSRF attacks, or DoS. Ensures the HTML splitter cannot be exploited to read local files or make unauthorized network requests when processing untrusted HTML content.
