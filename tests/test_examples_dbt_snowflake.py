"""The dbt on Snowflake example: its pieces name the same seventeen tickets, its evaluation is the check
client byte for byte, and `assemble.py` builds the team repository from a benchmark checkout the way the
README says. A ticket is known in four places (its file, the client's list, the platform's task map, its
build recipe), and one missing from any of them fails only at run time, on a warehouse, so the agreement is
held here. `assemble.py` runs on a stand-in checkout: the seven starting files land where the tickets name
them, the template arrives whole, the checkout's own connection profile and build output stay behind, and a
directory that is no checkout, or an output directory that exists, is refused untouched."""
import ast
import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "dbt-snowflake"
TEMPLATE = EXAMPLE / "repo-template"
ASSEMBLE = EXAMPLE / "assemble.py"
PROJECT = Path("base-image") / "dbt_models_snowflake"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def starting_files() -> dict:
    """The script's own map of task file to place in the warehouse project, read from its source: the
    script runs on import, so it is parsed, not imported."""
    tree = ast.parse(ASSEMBLE.read_text(encoding="utf-8"))
    named = [
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "BROKEN"
    ]
    return ast.literal_eval(named[0])


def stand_in_checkout(root: Path, leave_out: str = "") -> Path:
    checkout = root / "data-eng-bench"
    project = checkout / PROJECT
    (project / "models" / "staging").mkdir(parents=True)
    (project / "target").mkdir()
    (project / "dbt_project.yml").write_text("name: retail_dw_master\n", encoding="utf-8")
    (project / "models" / "staging" / "stg_orders.sql").write_text("select 1\n", encoding="utf-8")
    (project / "profiles.yml").write_text("the image's own connection profile\n", encoding="utf-8")
    (project / "target" / "manifest.json").write_text("{}\n", encoding="utf-8")
    for source in starting_files():
        if source != leave_out:
            path = checkout / "tasks" / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"-- {source}\n", encoding="utf-8")
    return checkout


def assemble(checkout: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ASSEMBLE), str(checkout), str(out)], capture_output=True, text=True)


def test_a_ticket_is_known_in_every_place_that_needs_it():
    tickets = sorted(path.stem for path in (TEMPLATE / "tickets").glob("*.md"))
    client = read_json(TEMPLATE / "checks" / "tickets.json")
    recipes = EXAMPLE / "platform" / "service" / "recipes"
    assert len(tickets) == 17
    assert sorted(ticket["key"] for ticket in client["tickets"]) == tickets
    assert set(client["smoke"]) <= set(tickets)
    assert sorted(read_json(EXAMPLE / "platform" / "service" / "tickets.json")) == tickets
    assert sorted(path.stem for path in recipes.glob("*.sh") if path.stem != "lib") == tickets


def test_the_evaluation_is_the_check_client_byte_for_byte():
    # The evaluation is the client Kapso protects: it must speak to the platform exactly as the repository's does.
    for name in ("run_checks.py", "tickets.json", "service.json"):
        assert (EXAMPLE / "evaluation" / name).read_bytes() == (TEMPLATE / "checks" / name).read_bytes(), name


def test_assemble_builds_the_repository_from_the_template_and_a_checkout(tmp_path):
    out = tmp_path / "repo"
    done = assemble(stand_in_checkout(tmp_path), out)
    assert done.returncode == 0, done.stderr
    placed = starting_files()
    assert len(placed) == 7
    for source, target in placed.items():
        assert (out / "warehouse" / target).read_text(encoding="utf-8") == f"-- {source}\n"
    assert (out / "warehouse" / "models" / "staging" / "stg_orders.sql").is_file()
    assert not (out / "warehouse" / "profiles.yml").exists() and not (out / "warehouse" / "target").exists()
    for path in TEMPLATE.rglob("*"):
        if path.is_file():
            assert (out / path.relative_to(TEMPLATE)).read_bytes() == path.read_bytes(), path.name


def test_assemble_refuses_what_is_not_a_checkout_and_an_output_that_exists(tmp_path):
    missing = next(iter(starting_files()))
    refused = assemble(stand_in_checkout(tmp_path / "incomplete", leave_out=missing), tmp_path / "repo")
    assert refused.returncode != 0 and "is not a checkout of data-eng-bench" in refused.stderr
    assert not (tmp_path / "repo").exists()

    existing = tmp_path / "kept"
    existing.mkdir()
    (existing / "note.txt").write_text("mine\n", encoding="utf-8")
    refused = assemble(stand_in_checkout(tmp_path / "complete"), existing)
    assert refused.returncode != 0 and "exists" in refused.stderr
    assert [path.name for path in existing.iterdir()] == ["note.txt"]
