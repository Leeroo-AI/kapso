"""scripts/check_claims.py: the two claim rules of the docs design
(docs/plans/docs-and-examples-redesign.md, Checks) on fixture pages — a
words-out phrase in prose fails, with an `s` or `ly` ending, in the front
matter (folded over lines or not), in a heading, in a tag's visible text,
across a line break and in a manifest's strings at any depth (the package's
and the CLI's descriptions too), and in code does not; the front-page-only
section binds only the front pages; a result beside a metric word with no
provenance fails while one with provenance (a verb with who or when), a
provenance table's row whose cell is filled, a rule's number, a target and
a heading do not, a mark belongs to its nearest number, and an integer
beside a rating word is a result; a paragraph that runs into a fence is
still read — and the shipped pages pass, and the list is the direction
document's."""

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_claims", ROOT / "scripts" / "check_claims.py")
check_claims = importlib.util.module_from_spec(spec)
sys.modules["check_claims"] = check_claims
spec.loader.exec_module(check_claims)

LISTS = {"everywhere": ["self-improving", "team of agents", "autonomous", "long-running agent"], "front": ["judge"]}
DIRECTION = ROOT / "docs" / "research" / "kapso-redesign.md"


def page(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def findings(path: Path, lists=LISTS):
    return [f.split(": ", 1)[1] for f in check_claims.scan(path, lists)]


def test_a_words_out_phrase_in_prose_fails_and_in_code_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    inner = page(tmp_path, "docs/x.mdx", (
        "---\ntitle: X\n---\n\nA Self-Improving loop, and a team-of-agents mention.\n\n"
        "```python\nself_improving = 1  # team of agents\n```\n\nThe `self-improving` flag names the judge.\n"
    ))
    assert findings(inner) == ["words-out 'self-improving'"]      # the hyphenated form; not in the fence, not in the code span


def test_an_inflection_of_a_phrase_is_the_phrase(tmp_path, monkeypatch):
    # Regression: "fully autonomously" and "long-running agents" passed a
    # whole-phrase match on the docs index.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    inner = page(tmp_path, "docs/x.mdx", "Competing fully autonomously, the long-running agents ran.\n")
    assert findings(inner) == ["words-out 'autonomous'", "words-out 'long-running agent'"]
    assert findings(page(tmp_path, "docs/y.mdx", "An autonomy-first design.\n")) == []


def test_the_front_matter_is_prose_for_the_words_out_rule(tmp_path, monkeypatch):
    # Regression: a page's description said "fully autonomous" while its body said "unattended".
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    inner = page(tmp_path, "docs/x.mdx", (
        '---\ntitle: "Two autonomous sessions"\nsidebarTitle: "Sessions"\ndescription: "Unattended."\n'
        'keywords: ["autonomous AI", "olympiad"]\n---\n\nUnattended sessions.\n'
    ))
    assert findings(inner) == ["words-out 'autonomous' in the front matter's title", "words-out 'autonomous' in the front matter's keywords"]


def test_the_manifests_strings_are_prose_for_the_words_out_rule(tmp_path, monkeypatch):
    # Regression: the marketplace manifest kept the retired tagline after
    # every plugin manifest lost it; the Codex manifest's interface strings,
    # the package's description and the CLI's help line were outside the rule.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    monkeypatch.setattr(check_claims, "MANIFESTS", ("store.json", "ctx.json", "codex.json"))
    page(tmp_path, "store.json", json.dumps({"description": "the long-running agents", "plugins": [{"description": "clean"}]}))
    page(tmp_path, "ctx.json", json.dumps({"description": "clean", "rules": ["Run it autonomously."]}))
    page(tmp_path, "codex.json", json.dumps({"interface": {"shortDescription": "A team of agents", "defaultPrompt": ["Be autonomous"]}}))
    page(tmp_path, "pyproject.toml", '[project]\nname = "x"\ndescription = "A self-improving loop"\n')
    page(tmp_path, "src/kapso/cli/__init__.py", 'parser = argparse.ArgumentParser(description="Kapso: the software factory")\n')
    assert check_claims.scan_manifests(LISTS) == [
        "store.json: words-out 'long-running agent' in description",
        "ctx.json: words-out 'autonomous' in rules[0]",
        "codex.json: words-out 'team of agents' in interface.shortDescription",
        "codex.json: words-out 'autonomous' in interface.defaultPrompt[0]",
        "pyproject.toml: words-out 'self-improving' in project.description",
    ]
    assert check_claims.scan_manifests({"everywhere": ["software factory"], "front": []}) == [
        "src/kapso/cli/__init__.py: words-out 'software factory' in parser description",
    ]


def test_headings_tag_text_folded_front_matter_and_wrapped_phrases_are_read(tmp_path, monkeypatch):
    # Regression: a heading, a card's title, an image's alt text, a folded
    # description and a phrase broken over a line all passed the rule.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    inner = page(tmp_path, "docs/x.mdx", (
        "---\ntitle: X\ndescription: >\n  a team of\n  agents at work\nkeywords:\n  - clean\n  - autonomous AI\n---\n\n"
        "## An autonomous software factory\n\n<CardGroup>\n  <Card title=\"Self-improving agents\" href=\"/x\">\n    Clean.\n  </Card>\n</CardGroup>\n\n"
        "<img src=\"a.png\" alt=\"a team of agents\" />\n\nA team of\nagents, wrapped.\n"
    ))
    assert findings(inner) == [
        "words-out 'team of agents' in the front matter's description",
        "words-out 'autonomous' in the front matter's keywords",
        "words-out 'autonomous' in a heading or a tag's text",
        "words-out 'self-improving' in a heading or a tag's text",
        "words-out 'team of agents' in a heading or a tag's text",
        "words-out 'team of agents'",
    ]


def test_the_front_section_binds_the_front_pages_only(tmp_path, monkeypatch):
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    inner = page(tmp_path, "docs/build/x.mdx", "The judge reads the evaluation.\n")
    front = page(tmp_path, "docs/index.mdx", "The judge reads the evaluation.\n")
    readme = page(tmp_path, "README.md", "We judge nothing; the evaluation does.\n")
    assert findings(inner) == []
    assert findings(front) == ["words-out 'judge'"]
    assert findings(readme) == ["words-out 'judge'"]


def test_a_result_beside_a_metric_word_needs_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    bare = page(tmp_path, "docs/a.mdx", "The model reaches an AUROC of 0.82 on the holdout.\n")
    assert findings(bare) == ["a number with no provenance: 0.82 beside 'auroc'"]
    sourced = page(tmp_path, "docs/b.mdx", "The model reaches an AUROC of 0.82 on the holdout, measured by us on the provided evaluator on 2026-10-01.\n")
    assert findings(sourced) == []
    table = page(tmp_path, "docs/c.mdx", "| Task | Score | Who ran it |\n|---|---|---|\n| churn | 0.82 AUROC | our campaign, 2026-10-01 |\n")
    assert findings(table) == []
    blank_cell = page(tmp_path, "docs/c2.mdx", "| Task | Score | Who ran it |\n|---|---|---|\n| churn | 0.82 AUROC | |\n| fraud | 0.71 AUROC | us, 2026-10-01 |\n")
    assert findings(blank_cell) == ["a number with no provenance: 0.82 beside 'auroc'"]
    threshold = page(tmp_path, "docs/d.mdx", "The frame rejects any score outside centre ± 0.2.\n")
    assert findings(threshold) == []
    heading = page(tmp_path, "docs/e.mdx", "### 3. Extract score\n\nThe generator parses the score.\n")
    assert findings(heading) == []
    percent = page(tmp_path, "docs/f.mdx", "Accuracy rose by 12% after the change.\n")
    assert findings(percent) == ["a number with no provenance: 12% beside 'accuracy'"]


def test_the_product_name_is_not_provenance_and_one_decimal_is_a_result(tmp_path, monkeypatch):
    # Regression: "Kapso scored 536.07" passed because the product's name counted as
    # provenance; "81.2 AUROC" passed because a result needed two decimals.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    named = page(tmp_path, "docs/a.mdx", "Kapso scored 536.07 in the track, above the table below.\n")
    assert findings(named) == ["a number with no provenance: 536.07 beside 'scored'"]
    short = page(tmp_path, "docs/b.mdx", "Kapso leads with 81.2 AUROC over twelve tasks.\n")
    assert findings(short) == ["a number with no provenance: 81.2 beside 'auroc'"]
    ranked = page(tmp_path, "docs/c.mdx", "The rank of 3.5 came from a run graded by the official scorer on 2026-10-11.\n")
    assert findings(ranked) == []


def test_provenance_is_a_verb_with_who_or_when(tmp_path, monkeypatch):
    # Regression: "measured." alone passed as provenance, a named person with
    # an ISO date was flagged, and an integer beside a rating word escaped.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    bare_verb = page(tmp_path, "docs/a.mdx", "Accuracy is 0.91, measured.\n")
    assert findings(bare_verb) == ["a number with no provenance: 0.91 beside 'accuracy'"]
    named = page(tmp_path, "docs/b.mdx", "Alice measured accuracy_score at 0.91 on 2026-10-01.\n")
    assert findings(named) == []
    linked = page(tmp_path, "docs/c.mdx", "An accuracy of 0.91, recorded in [the run's report](https://example.org/r).\n")
    assert findings(linked) == []
    elo = page(tmp_path, "docs/d.mdx", "Kapso reaches a final rating of 1909 Elo against the other agent's 1879.\n")
    assert findings(elo) == ["a number with no provenance: 1909 beside 'rating'"]     # 1879 sits out of a rating word's reach
    sourced_elo = page(tmp_path, "docs/e.mdx", "A final rating of 1909 Elo in runs by Leeroo, scored by the official scorer.\n")
    assert findings(sourced_elo) == []
    plain_integer = page(tmp_path, "docs/f.mdx", "The score over 12 tasks and 100 charts.\n")
    assert findings(plain_integer) == []


def test_a_mark_covers_the_numbers_of_its_own_clause(tmp_path, monkeypatch):
    # Regression: a "toward" beside the target exempted the measured
    # baseline in the clause before it.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    close = page(tmp_path, "docs/a.mdx", "The shipped model's accuracy is 0.30, toward 0.50.\n")
    assert findings(close) == ["a number with no provenance: 0.30 beside 'accuracy'"]
    own = page(tmp_path, "docs/b.mdx", "An accuracy expected near 0.10 to 0.20, no run recorded, toward the brief's 0.50.\n")
    assert findings(own) == []
    span = page(tmp_path, "docs/c.mdx", "From a dummy at about 0.50 accuracy toward the brief's 0.78 on the data.\n")
    assert findings(span) == []


def test_a_tag_line_is_read_for_words_and_not_as_a_paragraph(tmp_path, monkeypatch):
    # Regression: an image's alt text repeating the figure's numbers was a
    # paragraph of its own, with no provenance of its own; its words are
    # still read.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    figure = page(tmp_path, "docs/a.mdx", (
        "Runs by Leeroo, recorded on 2026-10-11: a rating of 1909 Elo.\n\n"
        '<Frame caption="A rating of 1909 Elo by a team of agents">\n  <img\n    src="/x.png"\n    alt="Kapso at 1909 Elo"\n  />\n</Frame>\n\n'
        "![Kapso at 1909 Elo](x.png)\n\n"
        '<p align="center"><img src="/x.png" alt="Kapso at 1909 Elo" width="574"></p>\n'
    ))
    assert findings(figure) == ["words-out 'team of agents' in a heading or a tag's text"]


def test_a_target_or_a_rule_is_read_beside_its_number_not_per_paragraph(tmp_path, monkeypatch):
    # Regression: one "toward" exempted a whole paragraph, including a measured baseline.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    mixed = page(tmp_path, "docs/a.mdx", (
        "The shipped model's accuracy is 0.30 on the chart set. Much later in the same paragraph the brief asks for a rise "
        "toward 0.50 accuracy.\n"
    ))
    assert findings(mixed) == ["a number with no provenance: 0.30 beside 'accuracy'"]
    rule = page(tmp_path, "docs/b.mdx", "A candidate keeps its score within 0.5 of the floor; the budget is 1.5 hours.\n")
    assert findings(rule) == []


def test_a_paragraph_that_runs_into_a_fence_is_read(tmp_path, monkeypatch):
    # Regression: prose with no blank line before a fence was yielded as fence
    # content and skipped the number rule.
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    abutting = page(tmp_path, "docs/a.mdx", "An accuracy of 0.91 on the holdout:\n```bash\nkapso build\n```\n")
    assert findings(abutting) == ["a number with no provenance: 0.91 beside 'accuracy'"]
    pasted = page(tmp_path, "docs/b.mdx", "Paste this:\n```text\nreach an accuracy of 0.91 with a team of agents\n```\n")
    assert findings(pasted) == ["words-out 'team of agents'"]


def test_the_list_is_the_direction_documents():
    # Regression: the lint's list is a copy of the document's; a phrase added
    # to the document must reach the file (the file may hold more: the front
    # section's split, an inflection the pattern needs).
    text = DIRECTION.read_text(encoding="utf-8")
    start = text.index("- Words out:")
    bullet = text[start:text.index("\n- ", start + 1)]
    documented = re.findall(r'"([^"]+)"', bullet)
    assert len(documented) >= 10
    lists = check_claims.words_out()
    patterns = [check_claims.phrase_pattern(phrase) for phrase in lists["everywhere"] + lists["front"]]
    missing = [phrase for phrase in documented if not any(pattern.search(phrase) for pattern in patterns)]
    assert missing == []


def test_the_shipped_pages_pass():
    # Regression: the public pages and manifests hold to both rules — a hit is
    # fixed on its page, never by widening the allow-list without a note in the plan.
    lists = check_claims.words_out()
    hits = check_claims.scan_manifests(lists) + [finding for path in check_claims.pages() for finding in check_claims.scan(path, lists)]
    assert hits == []
