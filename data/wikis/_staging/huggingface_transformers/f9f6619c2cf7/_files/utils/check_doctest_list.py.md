# utils/check_doctest_list.py

## Understanding

### Purpose
Maintains doctest exclusion lists

### Mechanism
The script validates two doctest list files (`not_doctested.txt` and `slow_documentation_tests.txt`) by checking that all listed paths exist in the repository and are sorted alphabetically. It can automatically fix ordering issues and reports any non-existent paths that need to be removed.

### Significance
Keeps the doctest configuration clean and organized, preventing broken references to non-existent files and making it easy to review which documentation examples are excluded from testing. The alphabetical ordering makes the lists more maintainable and easier to navigate.
