#!/usr/bin/env python3
"""Generate docs/translations/README.<lang>.md from README.md, through the
Claude Code CLI, and keep every language switcher in step.

  python3 scripts/translate_readme.py zh-CN ja-JP     # translate these languages
  python3 scripts/translate_readme.py --all            # every language this script knows
  python3 scripts/translate_readme.py --switchers      # only rewrite the switcher blocks

One language at a time: the switcher is cut out of README.md and the rest is
piped to `claude -p` with the rules below (prose translated; links, code,
numbers, badges and names byte-identical; code comments translated). The
reply is cut to the document, relative link targets are prefixed with ../../
for the folder the file lives in, a header comment and the language switcher
are added, and scripts/check_readme_translations.py is run on the result.
A translation that drifts is still written, so it can be read, and the drift
is printed; the exit status says whether every requested language passed.

The switcher in README.md and in every translation lists the files present
under docs/translations/, so adding a language updates them all.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_readme_translations import PREFIX, SWITCHER_BLOCK, SWITCHER_LINK, TRANSLATIONS_DIR, check_switchers, compare, translations  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"

# Every language the switcher can name, in the order it lists them: the code
# the file carries and the name a reader of that language looks for.
LANGUAGES = {
    "zh-CN": "简体中文", "ja-JP": "日本語", "ko-KR": "한국어", "de-DE": "Deutsch", "fr-FR": "Français",
    "es-ES": "Español", "hi-IN": "हिन्दी", "pt-BR": "Português", "ru-RU": "Русский", "ar-SA": "العربية",
    "fa-IR": "فارسی", "it-IT": "Italiano", "pl-PL": "Polski", "nl-NL": "Nederlands", "tr-TR": "Türkçe",
    "uk-UA": "Українська", "vi-VN": "Tiếng Việt", "id-ID": "Bahasa Indonesia", "sv-SE": "Svenska",
    "el-GR": "Ελληνικά", "ro-RO": "Română", "cs-CZ": "Čeština", "fi-FI": "Suomi", "da-DK": "Dansk",
    "no-NO": "Norsk", "hu-HU": "Magyar", "th-TH": "ภาษาไทย", "uz-UZ": "Oʻzbekcha", "zh-TW": "繁體中文",
    "fil-PH": "Filipino", "he-IL": "עברית",
}
ENGLISH_NAMES = {
    "zh-CN": "Simplified Chinese", "ja-JP": "Japanese", "ko-KR": "Korean", "de-DE": "German", "fr-FR": "French",
    "es-ES": "Spanish", "hi-IN": "Hindi", "pt-BR": "Brazilian Portuguese", "ru-RU": "Russian", "ar-SA": "Arabic",
    "fa-IR": "Persian", "it-IT": "Italian", "pl-PL": "Polish", "nl-NL": "Dutch", "tr-TR": "Turkish",
    "uk-UA": "Ukrainian", "vi-VN": "Vietnamese", "id-ID": "Indonesian", "sv-SE": "Swedish", "el-GR": "Greek",
    "ro-RO": "Romanian", "cs-CZ": "Czech", "fi-FI": "Finnish", "da-DK": "Danish", "no-NO": "Norwegian",
    "hu-HU": "Hungarian", "th-TH": "Thai", "uz-UZ": "Uzbek", "zh-TW": "Traditional Chinese",
    "fil-PH": "Filipino", "he-IL": "Hebrew",
}

PLACEHOLDER = "<!-- translations:switcher -->"
SWITCHER_DIV = re.compile(r'<div align="center">\n<!-- translations:start -->.*?<!-- translations:end -->\n</div>', re.S)
SUMMARY = "Read this in other languages"
LINK = re.compile(r'\]\(([^)\s]+)\)|(?:href|src)="([^"]+)"')
WRAPPED = re.compile(r"\A\s*```(?:markdown|md)?\n(.*)\n```\s*\Z", re.S)

INSTRUCTIONS = """Translate the Markdown document on stdin, the README of Kapso, from English into {name} ({code}). Output the translated document only: no preamble, no closing remarks, no code fence around it. Begin with the same first line as the input.

Rules:
1. Translate the prose, the headings, the table cells, the list items, the blockquotes, and the visible text inside HTML tags such as <h4>, <sub>, <p> and <summary>.
2. Keep byte-identical: every URL and link target; every image and badge; every HTML tag and attribute, alt text included; every inline code span; every fenced code block except its comments; product and brand names (Kapso, Leeroo, Leeroopedia, Claude Code, Codex, Cursor, OpenCode, Docker, Modal, BentoCloud, LangGraph, PyPI, RelBench, MLE-Bench, ALE-Bench, IOAI, Kaggle, Stanford, OpenAI, Sakana AI, KumoRFM, LinkedIn, Meta AI, Moltbook); command names and flags; file and directory names; and every number exactly as written, with ASCII digits and the same decimal point.
3. Inside a fenced code block translate only the comments, the text after a # on a line, and leave every other character unchanged.
4. Copy the line `{placeholder}` unchanged, where it stands.
5. Keep the structure: the same headings in the same order and at the same level, the same tables with the same columns, the same lists, the same paragraph breaks.
6. Render these terms with their standard equivalent in {name} and use each consistently: objective, success criteria, evaluation, experiment, tree of experiments, hypothesis, candidate, campaign, lesson, bank of lessons, credence score, boundary, evidence, deployment, target stack, trajectory, coding agent, plugin. Keep the English word "harness", and keep the CLI verbs (build, learn, deploy, experience, harness, watch, inbox, trace, report, doctor) in English where they name a command.
7. No emojis beyond those already in the input, no em dashes, no translator's notes.
"""


def header_comment(code: str) -> str:
    return (
        f"<!-- {ENGLISH_NAMES[code]} translation of README.md, generated with scripts/translate_readme.py "
        "(Claude Code). Keep it in step: README.md and this file change in the same commit, and "
        "scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->\n"
    )


def present_codes() -> list[str]:
    """The languages with a file under docs/translations/, in switcher order."""
    names = {p.name for p in translations()}
    return [code for code in LANGUAGES if f"README.{code}.md" in names]


def switcher(current: str | None, codes: list[str]) -> str:
    """The switcher block for README.md (current None) or for one translation."""
    entries = []
    if current is None:
        entries.append("<b>English</b>")
    else:
        entries.append(f'<a href="{PREFIX}README.md">English</a>')
    for code in codes:
        if code == current:
            entries.append(f"<b>{LANGUAGES[code]}</b>")
        else:
            target = f"docs/translations/README.{code}.md" if current is None else f"README.{code}.md"
            entries.append(f'<a href="{target}">{LANGUAGES[code]}</a>')
    line = " | ".join(entries)
    return (
        '<div align="center">\n<!-- translations:start -->\n'
        f"<details><summary><b>{SUMMARY}</b></summary>\n\n{line}\n\n</details>\n"
        "<!-- translations:end -->\n</div>"
    )


def rewrite_switchers() -> None:
    codes = present_codes()
    source = SOURCE.read_text(encoding="utf-8")
    if not SWITCHER_DIV.search(source):
        raise SystemExit("README.md has no switcher block to rewrite")
    SOURCE.write_text(SWITCHER_DIV.sub(lambda _: switcher(None, codes), source), encoding="utf-8")
    for code in codes:
        path = TRANSLATIONS_DIR / f"README.{code}.md"
        text = path.read_text(encoding="utf-8")
        if not SWITCHER_DIV.search(text):
            raise SystemExit(f"{path.name} has no switcher block to rewrite")
        path.write_text(SWITCHER_DIV.sub(lambda _: switcher(code, codes), text), encoding="utf-8")
    print(f"switchers rewritten for English and {len(codes)} translation(s)")


def relative_targets(source: str) -> list[str]:
    targets = {a or b for a, b in LINK.findall(source)}
    return sorted(
        t for t in targets
        if not t.startswith(("http://", "https://", "#", "mailto:")) and not SWITCHER_LINK.fullmatch(t)
    )


def prefix_relative_links(text: str, targets: list[str]) -> str:
    for target in sorted(targets, key=len, reverse=True):
        pattern = re.compile(r'(\]\(|href="|src=")(?!\.\./\.\./)' + re.escape(target) + r'(?=[)"])')
        text = pattern.sub(lambda m: f"{m.group(1)}{PREFIX}{target}", text)
    return text


def cut_to_document(reply: str, first_line: str) -> str:
    wrapped = WRAPPED.match(reply)
    if wrapped:
        reply = wrapped.group(1)
    start = reply.find(first_line)
    if start < 0:
        raise SystemExit(f"the reply does not contain the document's first line {first_line!r}")
    return reply[start:].rstrip() + "\n"


def translate(code: str, model: str | None, timeout_seconds: int) -> bool:
    source = SOURCE.read_text(encoding="utf-8")
    if not SWITCHER_DIV.search(source):
        raise SystemExit("README.md has no switcher block; add one between <!-- translations:start --> and <!-- translations:end -->")
    document = SWITCHER_DIV.sub(PLACEHOLDER, source)
    first_line = document.splitlines()[0]
    command = ["claude", "-p", INSTRUCTIONS.format(name=ENGLISH_NAMES[code], code=code, placeholder=PLACEHOLDER), "--output-format", "text"]
    if model:
        command += ["--model", model]
    with tempfile.TemporaryDirectory() as scratch:   # away from the repository, so no project context is loaded
        result = subprocess.run(command, input=document, capture_output=True, text=True, cwd=scratch, timeout=timeout_seconds)
    if result.returncode != 0:
        raise SystemExit(f"claude exited {result.returncode} for {code}:\n{result.stderr}")
    text = cut_to_document(result.stdout, first_line)
    text = prefix_relative_links(text, relative_targets(document))
    if PLACEHOLDER in text:
        text = text.replace(PLACEHOLDER, switcher(code, present_codes_with(code)), 1)
    else:
        marker = "</sub></p>\n"
        at = text.find(marker)
        if at < 0:
            raise SystemExit(f"{code}: neither the switcher placeholder nor the subtitle line survived the translation")
        at += len(marker)
        text = text[:at] + "\n" + switcher(code, present_codes_with(code)) + "\n" + text[at:]
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSLATIONS_DIR / f"README.{code}.md"
    path.write_text(header_comment(code) + text, encoding="utf-8")
    problems = compare(source, path.read_text(encoding="utf-8"))
    for problem in problems:
        print(f"{path.name}: {problem}")
    print(f"{path.name}: {'written, with the drift above' if problems else 'written, links, numbers and code match README.md'}")
    return not problems


def present_codes_with(code: str) -> list[str]:
    codes = present_codes()
    if code not in codes:
        codes = [c for c in LANGUAGES if c in codes or c == code]
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate README.md into docs/translations/README.<lang>.md with the Claude Code CLI")
    parser.add_argument("codes", nargs="*", help=f"language codes, one of: {', '.join(LANGUAGES)}")
    parser.add_argument("--all", action="store_true", help="every language this script knows")
    parser.add_argument("--switchers", action="store_true", help="only rewrite the switcher blocks from the files present")
    parser.add_argument("--model", default=None, help="a model for the claude CLI (default: the CLI's own)")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="per-language deadline for the claude call")
    args = parser.parse_args()
    codes = list(LANGUAGES) if args.all else args.codes
    unknown = [c for c in codes if c not in LANGUAGES]
    if unknown:
        parser.error(f"unknown language code(s): {', '.join(unknown)}")
    if not codes and not args.switchers:
        parser.error("name at least one language code, or --all, or --switchers")
    passed = True
    for code in codes:
        print(f"==> {code} ({ENGLISH_NAMES[code]})")
        passed = translate(code, args.model, args.timeout_seconds) and passed
    rewrite_switchers()
    source = SOURCE.read_text(encoding="utf-8")
    for problem in check_switchers(source, translations()):
        print(problem)
        passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
