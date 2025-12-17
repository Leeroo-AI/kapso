# File: `method_comparison/test_sanitizer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 38 |
| Functions | `df_products`, `test_exploit_fails`, `test_operations` |
| Imports | pandas, pytest, sanitizer |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest test suite that validates the sanitizer module's ability to correctly parse safe filter expressions and reject malicious code injection attempts.

**Mechanism:** Creates a test DataFrame fixture with product data (product_id, category, price, stock). The test_exploit_fails() function verifies that injection attacks (e.g., attempts to call os.system()) raise ValueError exceptions. The test_operations() function uses parametrized testing to validate 9 different filter expressions including comparisons, membership tests, boolean combinations, negations, and bitwise operators, ensuring each returns the correct product IDs.

**Significance:** Essential security testing that ensures the sanitizer properly defends against code injection while maintaining correct filtering behavior. Provides regression protection for this critical security boundary in the web application, verifying that legitimate queries work while malicious ones are blocked.
