"""What a failed build tells the submitter: the kind of error and the names involved, never free text.

A build runs submitted code inside the check's own database, so anything the build log prints can be made to
carry what that code reads there (a model can raise an error with a query's result in its message). The log
therefore never leaves this machine. What leaves is a short classification made here:

  - dbt's own error headers ("Compilation Error in model x", "Database Error in model x");
  - the Snowflake error code and class of each database error;
  - a fixed set of recognised messages, rewritten from a template, with the identifiers they name;
  - the recipe's own "CHECK:" lines, and dbt's closing count of results.

An identifier is passed on only when it already appears in the submitted project (or is a schema of the
warehouse, or the check database itself): a name the build computed while it ran is replaced by a
placeholder, so a build cannot spell out data through the names in its errors. Values quoted by an error
(the text that was not a number, the timestamp that did not parse) are always withheld. A message the table
below does not know is reduced to its header; the submitter can reproduce the build in the development
database, which holds the same data, and read the full message there.
"""
import pathlib
import re

from pool import load_config

MAX_LINES = 14
MAX_NAME = 96
WAREHOUSE_SCHEMAS = {"main", "analytics", "snapshots", "public", "information_schema", "raw", "staging"}
WITHHELD = "<name not in the submitted project>"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
# A check database's name, from the pool's own prefix so the two cannot drift apart.
CHECK_DATABASE = re.compile(rf"\b{re.escape(load_config()['pool']['clone_prefix'])}[0-9A-F]+\b", re.IGNORECASE)
TEXT_SUFFIXES = {".sql", ".yml", ".yaml", ".md", ".csv", ".txt", ".json"}

HEADER = re.compile(r"\b(Compilation|Database|Runtime|Parsing|Dependency|Validation|Internal) Error(?: in (model|snapshot|macro|test|seed|operation|analysis|sql operation) ([A-Za-z0-9_.]+))?")
SNOWFLAKE_CODE = re.compile(r"\b(\d{6}) \(([0-9A-Z]{5})\): ?(SQL compilation error|SQL execution error|SQL access control error)?")
SUMMARY = re.compile(r"Done\. PASS=(\d+) WARN=(\d+) ERROR=(\d+) SKIP=(\d+)(?: NO-OP=(\d+))? TOTAL=(\d+)")

# (pattern, template, which groups are identifiers). Groups that are not identifiers are digits or fixed words.
MESSAGES = [
    (re.compile(r"'([A-Za-z_][\w.]*)' is undefined"), "'{0}' is undefined: a macro or variable this build's project does not have", (0,)),
    (re.compile(r"depends on a node named '([\w.]+)' which was not found"), "the model depends on a node named '{0}', which was not found", (0,)),
    (re.compile(r"depends on a source named '([\w.]+)' which was not found"), "the model depends on a source named '{0}', which was not found", (0,)),
    (re.compile(r"dbt found two macros named \"([\w.]+)\""), "two macros are named \"{0}\" in this build's project", (0,)),
    (re.compile(r"No materialization '([\w.]+)' was found"), "no materialization '{0}' was found", (0,)),
    (re.compile(r"Object '([^']{1,300})' does not exist or not authorized"), "object {0} does not exist or is not authorized", (0,)),
    (re.compile(r"Object '([^']{1,300})' already exists"), "object {0} already exists", (0,)),
    (re.compile(r"Schema '([^']{1,300})' does not exist or not authorized"), "schema {0} does not exist or is not authorized", (0,)),
    (re.compile(r"invalid identifier '([^']{1,300})'"), "invalid identifier {0}", (0,)),
    (re.compile(r"ambiguous column name '([^']{1,300})'"), "ambiguous column name {0}", (0,)),
    (re.compile(r"Unknown functions? ([\w.]+)"), "unknown function {0}", (0,)),
    (re.compile(r"Unknown user-defined function ([\w.]+)"), "unknown user-defined function {0}", (0,)),
    (re.compile(r"Invalid argument types for function '([\w.]+)'"), "invalid argument types for function '{0}'", (0,)),
    (re.compile(r"Function ([\w.]+) does not support"), "function {0} does not support the type it was given", (0,)),
    (re.compile(r"Insufficient privileges to operate on (\w+) '([^']{1,300})'"), "insufficient privileges to operate on {0} {1}", (1,)),
    (re.compile(r"syntax error line (\d+) at position (\d+)"), "syntax error at line {0}, position {1} of the compiled statement", ()),
    (re.compile(r"error line (\d+) at position (\d+)"), "error at line {0}, position {1} of the compiled statement", ()),
    (re.compile(r"(Numeric value|Timestamp|Date|Time|Boolean value) '.*' is not recognized"), "a value could not be read as the type the statement asked for ({0}; the value is withheld)", ()),
    (re.compile(r"Division by zero"), "division by zero", ()),
    (re.compile(r"is not a valid group by expression"), "an expression is not a valid group by expression", ()),
    (re.compile(r"Single-row subquery returns more than one row"), "a single-row subquery returns more than one row", ()),
    (re.compile(r"Statement reached its statement or warehouse timeout"), "the statement reached its timeout", ()),
    (re.compile(r"Could not find profile named '([\w.]+)'"), "could not find the profile named '{0}'", (0,)),
    (re.compile(r"dbt_packages|packages\.yml.*not installed|Run \"dbt deps\" to install"), "the project's packages are missing: the build runs dbt deps on the submitted packages.yml", ()),
]


def known_names(candidate_dir):
    """Every identifier-like word of the submitted project's text files, lower-cased."""
    names = set(WAREHOUSE_SCHEMAS)
    for path in pathlib.Path(candidate_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            names.update(word.lower() for word in TOKEN.findall(path.read_text(errors="replace")))
            names.add(path.stem.lower())
    return names


def safe_name(raw, names):
    """A dotted, possibly quoted name with every part the submission does not contain replaced."""
    text = CHECK_DATABASE.sub("<check database>", raw.strip())[:MAX_NAME * 3]
    parts = []
    for part in text.split("."):
        bare = part.strip().strip('"')
        if bare == "<check database>" or (TOKEN.fullmatch(bare) and bare.lower() in names and len(bare) <= MAX_NAME):
            parts.append(bare)
        else:
            parts.append(WITHHELD)
    return ".".join(parts)


def classify(log_text, candidate_dir):
    """The lines a failed build's submitter may read."""
    names = known_names(candidate_dir)
    lines, seen = [], set()

    def add(line):
        if line not in seen and len(lines) < MAX_LINES:
            seen.add(line)
            lines.append(line)

    summary = None
    for raw in ANSI.sub("", log_text).splitlines():
        text = raw.strip()
        if text.startswith("CHECK: "):
            add(CHECK_DATABASE.sub("the check database", text)[:200])
            continue
        header = HEADER.search(text)
        if header:
            kind, resource, name = header.group(1), header.group(2), header.group(3)
            add(f"{kind} Error" + (f" in {resource} {safe_name(name, names)}" if resource and name else ""))
        code = SNOWFLAKE_CODE.search(text)
        if code:
            add(f"Snowflake error {code.group(1)} ({code.group(2)})" + (f": {code.group(3)}" if code.group(3) else ""))
        for pattern, template, identifier_groups in MESSAGES:
            match = pattern.search(text)
            if match:
                values = [safe_name(value, names) if index in identifier_groups else value for index, value in enumerate(match.groups())]
                add("  " + template.format(*values))
        counts = SUMMARY.search(text)
        if counts:
            summary = "dbt finished with PASS={0} WARN={1} ERROR={2} SKIP={3} TOTAL={5}".format(*counts.groups())
    explained = any(line.startswith("CHECK: the submission has no") for line in lines)
    if not explained and not any(not line.startswith("CHECK: ") for line in lines):
        lines.append("The build failed with an error this check does not classify.")
    if summary:
        lines.append(summary)
    lines.append("The full message is not returned by a check. Build the same models in the development database to read it.")
    return lines
