"""The experience reference grammar: `mine`, `leeroopedia`, a bank path, a
bank git reference, or the name of an ingested index. Design:
docs/research/onboarding.md section 2; the plan:
docs/plans/redesign/wp3-experience.md ("The reference").

Five forms, tried in this order: the two reserved names (`mine` with or
without `:all`, which lifts the scope rule); a path, written as one
(`./`, `../`, `/`, `~`), which must exist; a string with a scheme,
`git+` or an scp-like `user@host:`, or an `owner/repo` shorthand (with
`@rev`, `#subdirectory` or neither), as a git reference; an index name
that exists under the indexes home. A bare name is never a path: a
directory of the same name beside the run (the checkout an index was
made from, say) cannot shadow the index. A string that matches none
lists the forms; loading never ingests and never creates.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kapso.core.git_reference import (
    DEFAULT_REV,
    GitReference,
    is_shorthand,
    looks_like_git,
    parse_git_reference,
    shorthand_spec,
)

MINE = "mine"
HOSTED = "leeroopedia"
RESERVED = (MINE, HOSTED)
INDEX_RECORD = "index.json"
SCOPE_HARNESS = "harness"
SCOPE_ALL = "all"
SCOPES = (SCOPE_HARNESS, SCOPE_ALL)
ALL_SUFFIX = ":" + SCOPE_ALL
MINE_ALL = MINE + ALL_SUFFIX
PATH_PREFIXES = ("./", "../", "/", "~")
FORMS = (
    "  mine                                          your own bank, the one learn() writes, the cards your harness covers\n"
    "  mine:all                                      your own bank, every card whatever the harness\n"
    "  leeroopedia                                   the hosted gate, with its key in .env\n"
    "  ./bank.git  ~/team/bank.git                   a bank home, used in place\n"
    "  acme/kapso-bank  acme/kapso-bank@v2           a bank repository you read, GitHub shorthand\n"
    "  git+https://github.com/acme/kapso-bank@main   the same, any git URL\n"
    "  org-x                                         an index ingested under that name (kapso experience add --name org-x)"
)
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ExperienceReferenceError(ValueError):
    """A reference that matches no form, or names nothing that exists."""


@dataclass(frozen=True)
class ExperienceReference:
    raw: str
    kind: str                       # "mine" | "hosted" | "path" | "git" | "index"
    path: Optional[Path] = None     # path: the bank home; index: its directory
    url: Optional[str] = None       # git: the clone URL
    rev: str = DEFAULT_REV          # git: a tag, a branch or a sha, as written
    name: Optional[str] = None      # index: the name
    scope: str = SCOPE_HARNESS      # mine: "all" for `mine:all`; every other form is "all" by nature

    @property
    def slug(self) -> str:
        return GitReference(url=self.url or "", rev=self.rev).slug

    def describe(self) -> str:
        if self.kind == "git":
            return GitReference(url=self.url or "", rev=self.rev).describe()
        if self.kind in ("path", "index"):
            return str(self.path)
        return self.raw


def valid_index_name(name: str) -> bool:
    return bool(_NAME.match(name))


def recorded_reference(record: dict) -> str:
    """The reference string that re-loads a recorded source (a campaign's
    launch record): `mine` at every scope is `mine:all`; a bank read by
    path is its recorded location, absolute, so a resume or a reply from
    another directory finds the same bank; every other source is its
    reference as recorded."""
    reference = record["reference"]
    if reference == MINE and record["scope"] == SCOPE_ALL:
        return MINE_ALL
    if record["kind"] == "bank" and reference != MINE and not looks_like_git(reference) and not is_shorthand(reference):
        return record["resolved"]
    return reference


def parse_experience_reference(
    text: str, *, indexes_root: Path, cwd: Optional[Path] = None, must_exist: bool = True,
) -> ExperienceReference:
    """The reference `text` names. A path or an index name must exist,
    unless `must_exist` is off: then the reference is returned for the
    caller to check (a host session lists what the config names and
    leaves a missing source's gate off rather than stopping)."""
    raw = text.strip()
    if not raw:
        raise ExperienceReferenceError(f"an empty experience reference; the forms are:\n{FORMS}")
    if raw in (MINE, MINE_ALL):
        return ExperienceReference(raw=raw, kind=MINE, scope=SCOPE_ALL if raw == MINE_ALL else SCOPE_HARNESS)
    if raw == HOSTED:
        return ExperienceReference(raw=raw, kind="hosted", scope=SCOPE_ALL)
    if raw.endswith(ALL_SUFFIX):
        raise ExperienceReferenceError(
            f"`{ALL_SUFFIX}` is for `mine` alone: a bank you read, the hosted gate and an index are always every "
            f"scope (`{raw[:-len(ALL_SUFFIX)]}` loads it)"
        )
    if raw.startswith(PATH_PREFIXES):
        base = Path(cwd) if cwd is not None else Path.cwd()
        candidate = Path(raw).expanduser()
        directory = (candidate if candidate.is_absolute() else base / candidate).resolve()
        if must_exist and not directory.is_dir():
            raise ExperienceReferenceError(
                f"no directory {directory}: a reference written as a path must exist; the forms are:\n{FORMS}"
            )
        return ExperienceReference(raw=raw, kind="path", path=directory, scope=SCOPE_ALL)
    if looks_like_git(raw):
        parsed = parse_git_reference(raw[4:] if raw.startswith("git+") else raw)
        return ExperienceReference(raw=raw, kind="git", url=parsed.url, rev=parsed.rev, scope=SCOPE_ALL)
    if is_shorthand(raw):
        parsed = parse_git_reference(shorthand_spec(raw))
        return ExperienceReference(raw=raw, kind="git", url=parsed.url, rev=parsed.rev, scope=SCOPE_ALL)
    index = Path(indexes_root).expanduser() / raw
    if valid_index_name(raw) and ((index / INDEX_RECORD).is_file() or not must_exist):
        return ExperienceReference(raw=raw, kind="index", path=index, name=raw, scope=SCOPE_ALL)
    if valid_index_name(raw):
        raise ExperienceReferenceError(
            f"`{raw}` is not a reserved name, a path, a git reference, or an index under {indexes_root} "
            f"(`kapso experience add --repo <url> --name {raw}` ingests one; `kapso experience list` shows what "
            f"exists); the forms are:\n{FORMS}"
        )
    raise ExperienceReferenceError(
        f"`{raw}` is not a reserved name, a git reference or an index name, and a path is written as one "
        f"(`./`, `../`, `/`, `~`); the forms are:\n{FORMS}"
    )
