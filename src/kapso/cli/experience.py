"""`kapso experience`: what must exist before it can be loaded, what
exists, and sharing your bank. Design: docs/research/onboarding.md section
2; plan: docs/plans/redesign/wp3-experience.md."""

import argparse
import sys
from pathlib import Path

from kapso.core.config import load_config
from kapso.experience.sources import experience_settings
from kapso.experience import (
    HOSTED,
    HOSTED_KEY,
    MINE,
    ExperienceStore,
    bank_head,
    count_cards,
    hosted_key_missing,
    hosted_key_present,
    probe_hosted_gate,
)
from kapso.experience.browse import render_hits, render_page
from kapso.cli.common import _detach, _without_detach_flags, learning_handoff
from kapso.kapso import Kapso
from kapso.learning.bank_remote import bank_origin, connect_bank
from kapso.learning.update_frame import init_bank

NAME = "experience"


def cmd_experience(args) -> None:
    if args.experience_command == "add":
        _add(args)
        return
    if args.experience_command == "update":
        Kapso(config=args.config).update_experience(args.ref)
        return
    if args.experience_command == "remove":
        Kapso(config=args.config).remove_experience(args.ref)
        return
    if args.experience_command == "search":
        print(render_hits(Kapso(config=args.config).search_experience(args.index, args.query, top_k=args.top)))
        return
    if args.experience_command == "show":
        print(render_page(Kapso(config=args.config).show_experience(args.index, args.page)))
        return
    settings = experience_settings(load_config(args.config))
    home = Path(settings["bank_home"]).expanduser()
    if args.experience_command == "list":
        _list(settings, home)
    else:
        _share(args.url, home)


def _add(args) -> None:
    given = [value for value in (args.ref, args.repo, args.research, args.paper) if value]
    if len(given) != 1:
        print(
            f"Error: `add` takes exactly one of `{MINE}`, `{HOSTED}`, --repo <url>, --research <question> or --paper <file or url> "
            "(an index needs --name <n>)",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.repo or args.research or args.paper:
        if not args.name:
            print("Error: an index needs a name: --name <n> (`kapso experience list` shows the names taken)", file=sys.stderr)
            sys.exit(1)
        if args.detach:
            # Hours of ingestion: in a session of its own, the log under the
            # indexes' home, the status under the learning status directory.
            config = load_config(args.config)
            indexes = Path(experience_settings(config)["indexes"]).expanduser()
            _detach(
                "kapso experience add", _without_detach_flags(sys.argv[1:]),
                log_path=indexes / f"{args.name}.log", receipt_path=None,
                handoff=learning_handoff(config["learning"]["status_dir"]),
            )
            return
        # The facade owns the ingestion (the preflight, the pipeline, the
        # record); it reads the config itself.
        Kapso(config=args.config).add_experience(repo=args.repo, research=args.research, paper=args.paper, name=args.name)
        return
    if args.detach:
        print(f"Error: --detach goes with --repo, --research or --paper (an ingest); `{args.ref}` is instant", file=sys.stderr)
        sys.exit(1)
    settings = experience_settings(load_config(args.config))
    home = Path(settings["bank_home"]).expanduser()
    if args.ref == MINE:
        if home.exists():
            head = bank_head(home)
            print(f"your bank exists at {home}: head {head[:12]}, {count_cards(home, head)} cards")
            return
        init_bank(str(home))
        print(f"created your bank at {home} (empty; learn() fills it, --experience mine serves it)")
        return
    if args.ref == HOSTED:
        if not hosted_key_present():
            print(f"Error: {hosted_key_missing()}", file=sys.stderr)
            sys.exit(1)
        server = probe_hosted_gate(settings)
        print(
            f"{HOSTED}: {HOSTED_KEY} present and the gate answers as {server}; "
            f"`--experience {HOSTED}` mounts it in ideation and implementation"
        )
        return
    print(
        f"Error: `add` creates what must exist before it is loaded: `{MINE}` or `{HOSTED}`; a bank by path "
        "or git reference needs no add (load it), and an index is ingested with --repo <url> --name <n>, "
        "--research <question> --name <n> or --paper <file or url> --name <n>",
        file=sys.stderr,
    )
    sys.exit(1)


def _list(settings, home: Path) -> None:
    store = ExperienceStore(settings["banks"], settings["indexes"])
    banks = store.banks()
    indexes = store.indexes()
    key = hosted_key_present()
    if not home.exists() and not banks and not indexes and not key:
        print(
            f"no experience yet: `kapso experience add {MINE}` creates your bank (a first learn() does too); "
            f"`kapso experience add {HOSTED}` once {HOSTED_KEY} is in .env; "
            "`kapso experience add --repo <url> --name <n>` ingests a repository"
        )
        return
    if home.exists():
        head = bank_head(home)
        origin = bank_origin(home)
        remote = f"remote {origin}" if origin else "no remote (kapso experience share <url>)"
        print(f"{MINE:<13}{str(home):<44} head {head[:12]}   {count_cards(home, head)} cards   {remote}")
    else:
        print(f"{MINE:<13}not created: kapso experience add {MINE} (or a first learn())")
    print(f"{HOSTED:<13}hosted{'':<38} {'key present' if key else f'no key: put {HOSTED_KEY}=... in .env'}")
    if banks:
        print("banks you read")
        for bank in banks:
            path = Path(bank["path"])
            head = bank_head(path)
            print(f"  {bank['remote']}@{bank['rev']}   head {head[:12]}   {count_cards(path, head)} cards   {path}")
    if indexes:
        print("indexes")
        for record in indexes:
            print(f"  {record.name:<12} index {record.id[:12]}   {record.page_count} pages   ingested {record.created_at[:10]}   from {record.source}")


def _share(url: str, home: Path) -> None:
    if not home.exists():
        init_bank(str(home))
        print(f"created your bank at {home}")
    connect_bank(home, url)
    print(
        f"your bank {home} shares to {url}; learn() pushes there from now on, `kapso experience update` pulls "
        f"from there, and the intro says the cards are measured by you and your team"
    )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "experience",
        help="What a build is served that was measured before: your bank, the hosted gate, ingested corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        description=(
            "Experience is loaded at the call (`--experience mine,leeroopedia`, the brief's Experience line, "
            "load_experience()); this verb makes what must exist first, lists what exists, shares your bank, "
            "and looks inside an index (search a query, show a page)."
        ),
        epilog="""
Examples:
  kapso experience add mine                                   # your bank, empty, where learn() writes
  kapso experience add leeroopedia                            # checks the key in .env
  kapso experience add --repo https://github.com/org/x --name org-x      # ingests a repository into an index
  kapso experience add --research "QLoRA memory tricks" --name qlora     # researches the web, ingests the findings
  kapso experience add --paper https://arxiv.org/abs/2305.14314 --name qlora-paper   # ingests a paper's whole text
  kapso experience list                                       # mine, the hosted gate, banks you read, indexes
  kapso experience search org-x "paged optimizers"            # the ranked pages an index answers a query with
  kapso experience search org-x "paged optimizers" --top 3
  kapso experience show org-x Principle/Paged_Optimizers      # one page of the index, whole
  kapso experience share https://github.com/acme/kapso-bank.git
  kapso experience update                                     # pulls mine from its remote, re-fetches every bank you read
  kapso experience remove acme/kapso-bank                     # a bank you read, or an index by name
  kapso build --experience mine,org-x --brief kapso-brief.md ...
""",
    )
    experience_sub = parser.add_subparsers(dest="experience_command", required=True)
    add = experience_sub.add_parser(
        "add",
        help="Create your bank (`mine`), check the hosted gate's key (`leeroopedia`), or ingest a corpus into an index",
    )
    add.add_argument("ref", type=str, nargs="?", default=None, help="`mine` or `leeroopedia`")
    add.add_argument("--repo", type=str, default=None, help="A repository URL to ingest into the index --name")
    add.add_argument("--research", type=str, default=None, help="A question to research on the web and ingest into the index --name")
    add.add_argument("--paper", type=str, default=None, help="A paper (a PDF, markdown or text file, or a URL such as an arXiv link) to ingest into the index --name")
    add.add_argument("--name", type=str, default=None, help="The index's name (letters, digits, dashes)")
    add.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    add.add_argument(
        "--detach", action="store_true",
        help="Run an ingest (--repo, --research or --paper) in a session of its own and return at once with its pid, log and watch line",
    )
    listing = experience_sub.add_parser("list", help="Every source this machine has, with its state")
    listing.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    search = experience_sub.add_parser("search", help="The ranked pages an ingested index answers a query with")
    search.add_argument("index", type=str, help="An index name (`kapso experience list` shows them)")
    search.add_argument("query", type=str, help="What to look for, as a session would ask it")
    search.add_argument("--top", type=int, default=None, help="How many pages to print (default: the config's experience.search_top_k)")
    search.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    show = experience_sub.add_parser("show", help="One page of an ingested index, whole")
    show.add_argument("index", type=str, help="An index name (`kapso experience list` shows them)")
    show.add_argument("page", type=str, help="A page id as search prints it, e.g. Principle/Paged_Optimizers")
    show.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    share = experience_sub.add_parser("share", help="Attach a git remote to your bank and push it")
    share.add_argument("url", type=str, help="Git URL of the bank repository (https or ssh)")
    share.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    update = experience_sub.add_parser(
        "update", help="Pull mine from its remote and re-fetch a bank you read at a moving reference; every bank when none is named",
    )
    update.add_argument("ref", type=str, nargs="?", default=None, help="`mine` or a bank's git reference (default: every bank)")
    update.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    remove = experience_sub.add_parser("remove", help="Delete a bank you read (its clone) or an index (its namespace, pages and record)")
    remove.add_argument("ref", type=str, help="A bank's git reference, or an index name")
    remove.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    return cmd_experience
