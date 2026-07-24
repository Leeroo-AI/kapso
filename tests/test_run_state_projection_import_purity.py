"""Fresh-process import boundary for pure run-state projection."""

import subprocess
import sys


def test_run_state_projections_import_without_provider_or_runtime_dependencies():
    script = "\n".join(
        (
            "import sys",
            "sys.modules['openai'] = None",
            "sys.modules['litellm'] = None",
            "sys.modules['kapso.core.embedding_provider'] = None",
            "sys.modules['kapso.core.llm'] = None",
            "sys.modules['kapso.execution.coding_agents.factory'] = None",
            "sys.modules['kapso.execution.search_strategies.base'] = None",
            (
                "from kapso.execution.search_strategies.generic.ideation"
                ".archive_projection import encode_archive_state"
            ),
            (
                "from kapso.cross_run.launch.run_state_projection "
                "import ReconciledRunStateProjection"
            ),
            (
                "assert encode_archive_state.__module__"
                ".endswith('.archive_projection')"
            ),
            (
                "assert ReconciledRunStateProjection.__module__"
                ".endswith('.run_state_projection')"
            ),
        )
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
