"""The verb is `build`. Nothing under the code, the tests, the shipped
skill, the examples or the bot names the old verb, the old result class or
the old constructor keyword: the rename reached every executable contract
and every text a model or a person reads. Two citations are not uses: the
design documents whose filenames carry the old verb (`evolve-hub-design.md`,
`evolve-observability-design.md`, excluded by the hyphen) and the published
documentation paths under `docs/evolve/`."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("src", "tests", "skills", "examples", "moltbook_bot", "pyproject.toml")
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".cfg", ".ini"}
OLD_VERB = re.compile(r"(?<![A-Za-z])evolve(?![A-Za-z-])", re.IGNORECASE)
OLD_NAMES = re.compile(r"SolutionResult|Kapso\(config_path|config_path=None, kg_index")
ALLOWED = ("docs/evolve/", "0.7.10's verb")   # the released baseline's own verb, named where the S1 driver counts it


def _files():
    for entry in SCOPE:
        path = ROOT / entry
        if path.is_file():
            yield path
            continue
        for candidate in sorted(path.rglob("*")):
            if (
                candidate.is_file()
                and candidate.suffix in SUFFIXES
                and "__pycache__" not in candidate.parts
                and ".kapso" not in candidate.parts
                and candidate != Path(__file__)
                and not candidate.name.startswith("s1-released-")   # the released baseline's report names 0.7.10's own verb
            ):
                yield candidate


def test_nothing_names_the_old_verb_or_the_old_result():
    offenders = []
    for path in _files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(marker in line for marker in ALLOWED):
                continue
            if OLD_VERB.search(line) or OLD_NAMES.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
