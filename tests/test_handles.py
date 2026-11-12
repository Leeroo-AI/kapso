"""The build handles: `Budget` validates what it holds; `Brief` is one
grammar for the file and the object, parsed whole and rendered back the
same, and every rule it rejects is named here."""

import math

import pytest

from kapso.handles import Brief, BriefError, Budget

SAMPLE = """\
Objective    Raise the churn model's out-of-sample AUC above 0.85
             without changing the prediction contract.
Measured by  kapso_evaluation/evaluate.py, ROC AUC on the held-out rows
Baseline     0.797 (logistic regression, 2026-10-24)
Target       0.85
Rules
  - the evaluator is never modified
  - no fitting on the held-out rows
Budget       minutes=480 experiments=10
Harness      predictive-modelling
Experience   mine, leeroopedia
"""
PARSED = Brief(
    objective=(
        "Raise the churn model's out-of-sample AUC above 0.85\n"
        "without changing the prediction contract."
    ),
    measured_by="kapso_evaluation/evaluate.py, ROC AUC on the held-out rows",
    baseline="0.797 (logistic regression, 2026-10-24)",
    target="0.85",
    rules=["the evaluator is never modified", "no fitting on the held-out rows"],
    budget=Budget(minutes=480, experiments=10),
    harness="predictive-modelling",
    experience=["mine", "leeroopedia"],
)


# ------------------------------------------------------------------ Budget

@pytest.mark.parametrize("field,value", [
    ("minutes", 0), ("minutes", -5), ("minutes", math.inf), ("minutes", math.nan),
    ("minutes", True), ("minutes", "10"),
    ("usd", 0), ("reserve_minutes", -1),
    ("experiments", 0), ("experiments", 2.5), ("experiments", True), ("experiments", "3"),
])
def test_budget_rejects_what_cannot_be_spent(field, value):
    with pytest.raises(ValueError, match=f"Budget.{field}"):
        Budget(**{field: value})


def test_budget_reserve_must_fit_inside_the_minutes():
    with pytest.raises(ValueError, match="reserve_minutes"):
        Budget(minutes=10, reserve_minutes=10)
    assert Budget(minutes=10, reserve_minutes=2).reserve_minutes == 2
    assert Budget(reserve_minutes=30).minutes is None   # the config's minutes, checked at resolve time


def test_budget_merges_field_by_field_with_the_call_winning():
    call, brief = Budget(experiments=2), Budget(minutes=30, experiments=4)
    assert call.merged_over(brief) == Budget(minutes=30, experiments=2)
    assert Budget().merged_over(None) == Budget() and Budget().empty
    assert Budget(minutes=5, experiments=3, usd=2.5, reserve_minutes=1).tokens() == (
        "minutes=5 experiments=3 usd=2.5 reserve_minutes=1"
    )


# ------------------------------------------------------------------- Brief

def test_the_sample_brief_parses_to_its_object_and_renders_back_byte_for_byte():
    brief = Brief.parse(SAMPLE)
    assert brief == PARSED
    assert brief.render() == SAMPLE
    assert Brief.parse(brief.render()) == brief
    assert brief.render() == brief.render()


def test_a_sentence_is_a_brief_with_only_an_objective():
    brief = Brief(objective="Build a web scraper for news articles")
    assert brief.render() == "Objective    Build a web scraper for news articles\n"
    assert Brief.parse(brief.render()) == brief


def test_blank_lines_inside_a_field_are_paragraph_breaks_and_survive_the_round_trip():
    text = (
        "Objective    First paragraph.\n"
        "\n"
        "             Second paragraph, after one blank line.\n"
        "\n"
        "\n"
        "             Third, after two.\n"
        "Target       1\n"
    )
    brief = Brief.parse(text)
    assert brief.objective == (
        "First paragraph.\n\nSecond paragraph, after one blank line.\n\n\nThird, after two."
    )
    assert brief.render() == text
    # Blank lines between fields, and at the ends, are nothing.
    assert Brief.parse("\n\nObjective    a\n\n\nTarget   1\n\n") == Brief(objective="a", target="1")


def test_continuation_lines_keep_their_relative_indentation_and_the_first_line_stands_alone():
    """The lines after the first are dedented together: a pasted code
    block keeps its structure, and the first line's own indentation, which
    the file cannot hold (it follows the field name), is not pretended."""
    text = (
        "Objective    Fix the failing test below\n"
        "                 def test_contract():\n"
        "                     assert predict(rows).shape == (len(rows),)\n"
    )
    brief = Brief.parse(text)
    assert brief.objective == (
        "Fix the failing test below\n"
        "def test_contract():\n"
        "    assert predict(rows).shape == (len(rows),)"
    )
    assert Brief.parse(brief.render()) == brief


@pytest.mark.parametrize("objective", [
    "Fix this:\n    def f():\n        return 1",
    "\n    Find the minimum of the Rosenbrock function.\n\n    Your solution should:\n    1. implement it\n    2. print the point\n\n    Target: f(x,y) < 0.001\n    ",
    "Keep the contract:\n- preserve the API\n- no new dependencies",
    "one line",
])
def test_a_python_built_brief_renders_and_parses_back_to_itself(objective):
    """What a resume relies on: render is a fixed point over parse, for a
    triple-quoted goal, a bullet list in the objective and an indented
    block alike, so the checkpoint's bytes are what a bare resume
    re-renders."""
    brief = Brief(objective=objective, rules=["a rule"], budget=Budget(minutes=90.5, usd=0.00001))
    rendered = brief.render()
    assert Brief.parse(rendered) == brief
    assert Brief.parse(rendered).render() == rendered


def test_a_bullet_under_a_text_field_is_text():
    brief = Brief.parse("Objective    Keep the contract:\n             - preserve the API\n             - no new dependencies\n")
    assert brief.objective == "Keep the contract:\n- preserve the API\n- no new dependencies"


def test_a_list_item_may_continue_on_the_next_lines():
    text = (
        "Objective    a\n"
        "Rules\n"
        "  - the evaluator is never modified, and a candidate\n"
        "    that edits it is rejected\n"
        "  - no peeking\n"
    )
    brief = Brief.parse(text)
    assert brief.rules == [
        "the evaluator is never modified, and a candidate\nthat edits it is rejected",
        "no peeking",
    ]
    assert brief.render() == text


def test_experience_takes_items_too_and_renders_items_when_a_reference_holds_a_comma():
    assert Brief.parse("Objective a\nExperience\n  - mine\n  - acme/bank@v3\n").experience == ["mine", "acme/bank@v3"]
    brief = Brief(objective="a", experience=["one,two", "three"])
    assert brief.render() == "Objective    a\nExperience\n  - one,two\n  - three\n"
    assert Brief.parse(brief.render()) == brief


def test_field_names_match_as_whole_words_in_any_case():
    brief = Brief.parse("OBJECTIVE a\nmeasured BY  the script\nTARGET\t0.9\n")
    assert (brief.objective, brief.measured_by, brief.target) == ("a", "the script", "0.9")


def test_a_budget_may_span_lines_and_keeps_its_numbers():
    brief = Brief.parse("Objective a\nBudget minutes=90.5\n       experiments=3 usd=12\n")
    assert brief.budget == Budget(minutes=90.5, experiments=3, usd=12)
    assert brief.render() == "Objective    a\nBudget       minutes=90.5 experiments=3 usd=12\n"


def test_reference_strings_are_kept_verbatim():
    brief = Brief.parse("Objective a\nHarness acme/harnesses@v3#trading\nExperience ~/banks/mine.git, leeroopedia\n")
    assert brief.harness == "acme/harnesses@v3#trading"
    assert brief.experience == ["~/banks/mine.git", "leeroopedia"]


@pytest.mark.parametrize("text,message", [
    ("Goal  x\n", "not a field"),
    ("Objective a\nObjective b\n", "Objective given twice"),
    ("Target 1\n", "Objective is required"),
    ("Objective\nTarget 1\n", "Objective has no value"),
    ("  hello\nObjective a\n", "indented line before any field"),
    ("Objective a\nRules\n  text\n", "expected a `- ` item"),
    ("Objective a\nRules b\n", "takes `- ` items on the lines below"),
    ("Objective a\nRules\n", "Rules has no items"),
    ("Objective a\nRules\n  -\n", "empty item"),
    ("Objective a\nExperience mine\n  - x\n", "not both"),
    ("Objective a\nExperience mine,\n", "empty item"),
    ("Objective a\nBudget\n", "needs name=value tokens"),
    ("Objective a\nBudget hours=2\n", "not name=value with a name from"),
    ("Objective a\nBudget minutes=abc\n", "must be a number"),
    ("Objective a\nBudget experiments=10.0\n", "must be an integer"),
    ("Objective a\nBudget minutes=1 minutes=2\n", "Budget minutes given twice"),
    ("Objective a\nsome text at column zero\n", "not a field"),
    ("Objective a\nTargeted 0.9\n", "not a field"),
])
def test_what_the_grammar_rejects_and_how_it_says_so(text, message):
    with pytest.raises(BriefError, match=message):
        Brief.parse(text, source="kapso-brief.md")


@pytest.mark.parametrize("text,message", [
    ("Objective a\nTarget 1\nTarget 2\n", r"^kapso-brief.md:3: Target given twice"),
    ("Objective\nTarget 1\n", r"^kapso-brief.md:1: Objective has no value"),
    ("Objective a\nRules\n", r"^kapso-brief.md:2: Rules has no items"),
    ("Objective a\nBudget\n", r"^kapso-brief.md:2: Budget needs name=value tokens"),
    ("Objective a\nBudget minutes=5\n       hours=2\n", r"^kapso-brief.md:3: Budget token 'hours=2'"),
])
def test_errors_name_the_file_and_the_line(text, message):
    with pytest.raises(BriefError, match=message):
        Brief.parse(text, source="kapso-brief.md")


def test_a_budget_writes_numbers_the_parser_reads_back():
    assert Budget(usd=0.00001).tokens() == "usd=0.00001"
    assert Budget(minutes=1e20).tokens() == "minutes=100000000000000000000"
    assert Budget(minutes=90.5, usd=2.25).tokens() == "minutes=90.5 usd=2.25"
    assert Budget(usd=1e-7).tokens() == "usd=0.0000001"
    assert Brief.parse(f"Objective a\nBudget {Budget(usd=1e-7).tokens()}\n").budget == Budget(usd=1e-7)


def test_a_budget_the_engine_cannot_spend_is_refused_at_parse_time():
    with pytest.raises(ValueError, match="positive"):
        Brief.parse("Objective a\nBudget minutes=0\n")


def test_the_object_validates_and_normalizes_what_it_holds():
    assert Brief(objective="  hi  \n\n").objective == "hi"
    assert Brief(objective="a", budget=Budget()).budget is None
    with pytest.raises(ValueError, match="objective"):
        Brief(objective="   ")
    with pytest.raises(ValueError, match="rules"):
        Brief(objective="a", rules=["", "x"])
    with pytest.raises(ValueError, match="target"):
        Brief(objective="a", target="")
    with pytest.raises(ValueError, match="budget"):
        Brief(objective="a", budget={"minutes": 1})


def test_read_and_write_are_the_file_forms(tmp_path):
    path = tmp_path / "kapso-brief.md"
    PARSED.write(path)
    assert path.read_text() == SAMPLE
    assert Brief.read(path) == PARSED
    with pytest.raises(FileNotFoundError, match="brief not found"):
        Brief.read(tmp_path / "missing.md")


def test_the_record_form_round_trips():
    assert Brief.from_dict(PARSED.to_dict()) == PARSED
    assert PARSED.to_dict()["budget"] == {"minutes": 480, "experiments": 10, "usd": None, "reserve_minutes": None}
