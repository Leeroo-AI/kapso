"""The support-agent example: the bar its repository states is what its files hold, and the files a build
depends on agree with each other. The bar (Fable 5.1 at 10 of the 20 development conversations) is read from
a reference run; an edited episode, summary or adapter would move it without a word, so each reference summary
is derived again from its episodes with the repository's own adapter. The evaluation refuses to run unless
its protected copies are the bench files byte for byte, so that is held here too, and every reference ran
exactly the conversations the development list names."""
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "support-agent-harness"
BENCH = EXAMPLE / "repo" / "bench"
# The conversations each reference run resolved: the bar the repository's README states is read from them.
RESOLVED = {"fable-5.1": 10, "opus-5.5": 8, "qwen-harness-0.1.0": 8}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repository_adapter():
    """The names the repository's adapter defines, run from its source so no bytecode lands in the
    example, which a build copies as it stands."""
    path = BENCH / "adapter.py"
    names = {"__name__": "support_agent_adapter", "__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), names)
    return names


@pytest.mark.parametrize("reference, resolved", sorted(RESOLVED.items()))
def test_a_reference_summary_is_what_the_adapter_derives_from_its_episodes(reference, resolved):
    directory = BENCH / "reference" / reference
    episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary = read_json(directory / "summary.json")["summary"]
    assert repository_adapter()["summarize"](episodes, summary["k"]) == summary
    assert (summary["passes_total"], summary["tasks"]) == (resolved, 20)


def test_every_reference_ran_the_development_conversations_and_no_other():
    development = read_json(BENCH / "dev_tasks.json")["development"]
    assert len(development) == len(set(development)) == 20
    for reference in RESOLVED:
        assert sorted(read_json(BENCH / "reference" / reference / "summary.json")["summary"]["per_task"]) == sorted(development)


def test_the_evaluation_holds_the_bench_files_byte_for_byte():
    # evaluate.py compares these before every run and exits without a score on a difference.
    for name in ("run_dev.py", "adapter.py", "dev_tasks.json", "fixed_conditions.json"):
        assert (BENCH / name).read_bytes() == (EXAMPLE / "evaluation" / "protected" / name).read_bytes(), name


def test_the_evaluation_compares_runs_against_the_shipped_references():
    references = read_json(EXAMPLE / "evaluation" / "protected" / "references.json")
    for reference in ("fable-5.1", "opus-5.5"):
        assert references[reference]["per_task"] == read_json(BENCH / "reference" / reference / "summary.json")["summary"]["per_task"]
