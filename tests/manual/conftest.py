"""These are runnable scripts, not pytest tests.

Each was named test_* but defines no test function, and several call
exit() at import when their data is missing — which aborted collection
for the entire suite. They are kept because they are useful to run by
hand; pytest is told to leave them alone.

Run one directly, for example:
    python tests/manual/test_index_wikis.py
"""

collect_ignore_glob = ["test_*.py"]
