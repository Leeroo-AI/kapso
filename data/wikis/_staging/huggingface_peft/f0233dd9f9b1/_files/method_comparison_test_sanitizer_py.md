# File: `method_comparison/test_sanitizer.py`

**Category:** test suite

| Property | Value |
|----------|-------|
| Lines | 39 |
| Functions | `df_products` (fixture), `test_exploit_fails`, `test_operations` |
| Imports | pandas, pytest, sanitizer.parse_and_filter |

## Understanding

**Status:** Explored

**Purpose:** Test suite for the sanitizer module that validates safe filtering functionality and ensures security vulnerabilities are blocked.

**Mechanism:**
- `df_products`: Pytest fixture that creates a sample DataFrame with product data (6 rows with columns: product_id, category, price, stock)

- `test_exploit_fails`: Security test that verifies malicious code injection attempts are rejected:
  - Tests expression with `@os.system()` call
  - Ensures ValueError is raised with "Invalid filter syntax" message

- `test_operations`: Parameterized test covering 9 different filtering scenarios:
  - Simple comparisons: `price < 50`
  - Membership tests: `product_id in [101, 102]`
  - Compound conditions with 'and': `price < 50 and category == 'Electronics'`
  - Compound conditions with 'or': `stock < 100 or category == 'Home Goods'`
  - Nested parentheses: `(price > 100 and stock < 20) or category == 'Books'`
  - Negation: `not (price > 50 or stock > 100)`, `not price > 50`
  - Bitwise operators: `(price < 50) & (category == 'Electronics')`
  - Each test case validates expected product_ids are returned

**Significance:** Essential quality assurance for the security-critical sanitizer module. Demonstrates that the AST-based filtering safely handles legitimate query syntax while blocking code injection attempts. The comprehensive parameterized tests ensure correct behavior across various operator combinations and precedence rules.
