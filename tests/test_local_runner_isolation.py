"""Two local deployments in one process must each run their own helper modules."""

import pytest

from kapso.deployment.strategies.local.runner import LocalRunner


def _write_solution(path, value):
    path.mkdir()
    (path / "helper.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    (path / "main.py").write_text(
        "import helper\n\n"
        "def predict(inputs):\n"
        "    return {'value': helper.VALUE}\n",
        encoding="utf-8",
    )


@pytest.fixture
def two_solutions(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_solution(first, "first")
    _write_solution(second, "second")
    return first, second


def test_second_deployment_uses_its_own_helper(two_solutions):
    first, second = two_solutions
    assert LocalRunner(code_path=str(first)).run({})["value"] == "first"
    assert LocalRunner(code_path=str(second)).run({})["value"] == "second"


def test_redeploying_first_after_second_uses_first_helper_again(two_solutions):
    first, second = two_solutions
    LocalRunner(code_path=str(first))
    LocalRunner(code_path=str(second))
    assert LocalRunner(code_path=str(first)).run({})["value"] == "first"


def test_already_loaded_runner_keeps_working(two_solutions):
    first, second = two_solutions
    first_runner = LocalRunner(code_path=str(first))
    LocalRunner(code_path=str(second))
    assert first_runner.run({})["value"] == "first"
