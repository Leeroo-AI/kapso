"""The data platform's service for the retail warehouse: the ticket checks, the development reset, releases.

    POST /runs?tickets=a,b,c     body: a gzipped tar of the repository's warehouse/ and marketing/ directories
                                 -> 202 {"run": "<id>", "tickets": [...]}        400 with a reason when refused
    GET  /runs/<id>              -> {"run", "state": queued|running|done, "passed", "total", "tickets": [...]}
    GET  /health                 -> {"status": "ok", "ready_clones", "runs_in_progress"}
    POST /dev/reset              -> the development database put back to production's state (mirror.py)
    POST /releases               body: {"commit": "<sha on the base branch>"?, "also": ["model", ...]?}
                                 -> 202 {"release": "<id>", ...}; the release is staged in the background
    GET  /releases/<id>          -> the release's record: state, selection, what built in staging and in production
    POST /releases/<id>/publish  -> 202; the staged release is built in production after a backup
    POST /releases/<id>/rollback -> 202; production put back to what it was before that release
    GET  /production             -> the commit production stands at and the releases published

A ticket's entry holds its name, `passed`, how its build went (`ok`, `failed`, `timeout`, `not_run`), the seconds
it took and, when the build failed, the classification of the build's errors (build_errors.py). Nothing of the
acceptance checks themselves leaves this machine: not their code, not their output, not their expected values;
neither does a build's log. A release takes its code from the repository's base branch on GitHub, never from
the request (release.py): merging is the approval.

The service listens on the loopback address only; a build machine reaches it through a tunnel. It keeps the pool
of fresh clones filled in the background (pool.py) and runs at most `max_parallel_checks` checks at a time
across all runs (check_runner.py).

    python server.py            # serve until stopped; log on stdout
"""
import io
import json
import pathlib
import secrets
import tarfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from check_runner import run_ticket_check
from mirror import Mirror
from pool import CLEAN, CheckDatabases, load_config
from release import ReleaseRefused, Releases

CONFIG = load_config()
TICKETS = json.loads(pathlib.Path(CONFIG["tickets_file"]).read_text())
RUNS_DIR = pathlib.Path(CONFIG["runs_dir"])
POOL = CheckDatabases(CONFIG)
RELEASES = Releases(CONFIG)
CHECK_SLOTS = threading.Semaphore(CONFIG["max_parallel_checks"])
RUNS = {}
RUNS_LOCK = threading.Lock()
REQUIRED_FILES = ("warehouse/dbt_project.yml", "marketing/dbt_project.yml")
POOL_INTERVAL_SECONDS = 60


def log(message):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def check_ticket(run, ticket, candidate_dir, run_dir):
    with CHECK_SLOTS:
        database = POOL.acquire(f"{run['run']}/{ticket}")
        result = run_ticket_check(CONFIG, ticket, TICKETS[ticket], candidate_dir, database, run_dir / ticket)
        reset = POOL.release(database)
        if not reset["clean"]:
            log(f"PROBLEM: {database} did not come back clean after {ticket}: {reset['problems']}")
    return result


def ticket_thread(run, ticket, candidate_dir, run_dir):
    outcome = {"ticket": ticket, "passed": False, "build": "not_run"}
    worker = {"result": None, "error": None}

    def work():
        worker["result"] = check_ticket(run, ticket, candidate_dir, run_dir)

    # A check that raises (docker or Snowflake trouble) is this ticket's infrastructure error, not the run's end.
    inner = threading.Thread(target=work)
    inner.start()
    inner.join()
    if worker["result"] is not None:
        outcome = worker["result"]
    else:
        outcome["check_error"] = "the check could not be run; see the service log"
    with RUNS_LOCK:
        run["results"][ticket] = outcome
        log(f"run {run['run']} {ticket}: {'PASS' if outcome['passed'] else 'FAIL'} build={outcome['build']} {outcome.get('seconds', '')}s")


def execute_run(run, candidate_dir, run_dir):
    with RUNS_LOCK:
        run["state"] = "running"
    threads = [threading.Thread(target=ticket_thread, args=(run, ticket, candidate_dir, run_dir)) for ticket in run["tickets"]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with RUNS_LOCK:
        run["state"] = "done"
        (run_dir / "result.json").write_text(json.dumps(public_view(run), indent=2) + "\n")
    log(f"run {run['run']} done: {sum(1 for r in run['results'].values() if r['passed'])} of {len(run['tickets'])} pass")


def public_view(run):
    results = [run["results"].get(ticket, {"ticket": ticket, "state": "pending"}) for ticket in run["tickets"]]
    return {"run": run["run"], "state": run["state"], "tickets": results, "total": len(run["tickets"]),
            "passed": sum(1 for ticket in run["tickets"] if run["results"].get(ticket, {}).get("passed"))}


def unpack_submission(body, target):
    """Extract the gzipped tar safely; returns a refusal reason or None."""
    if not body:
        return "empty submission"
    if not tarfile.is_tarfile(io.BytesIO(body)):
        return "the submission is not a tar archive"
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            top = member.name.split("/")[0]
            if top not in ("warehouse", "marketing"):
                return f"unexpected top-level entry {top!r}; send warehouse/ and marketing/ only"
            if not (member.isfile() or member.isdir()):
                return f"{member.name} is neither a file nor a directory"
        archive.extractall(target, filter="data")
    for required in REQUIRED_FILES:
        if not (target / required).is_file():
            return f"the submission has no {required}"
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def reply(self, status, payload):
        body = (json.dumps(payload) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            databases = POOL.databases()
            clean = sum(1 for database in databases if database["comment"].startswith(CLEAN))
            with RUNS_LOCK:
                in_progress = sum(1 for run in RUNS.values() if run["state"] != "done")
            self.reply(200, {"status": "ok", "check_databases": len(databases), "clean": clean,
                             "runs_in_progress": in_progress, "tickets": list(TICKETS)})
            return
        if path.startswith("/runs/"):
            with RUNS_LOCK:
                run = RUNS.get(path.split("/")[2])
                view = public_view(run) if run else None
            if view is None:
                self.reply(404, {"error": "no such run"})
            else:
                self.reply(200, view)
            return
        if path == "/production":
            self.answer(RELEASES.state)
            return
        if path.startswith("/releases/"):
            self.answer(lambda: RELEASES.public_view(RELEASES.load(path.split("/")[2])))
            return
        self.reply(404, {"error": "not found"})

    def answer(self, work, status=200):
        """Reply with what `work` returns; a refusal is a 400 with its reason, anything else a 500 named in the log."""
        try:
            payload = work()
        except ReleaseRefused as refusal:
            self.reply(400, {"error": str(refusal)})
            return
        except Exception as error:   # noqa: BLE001 - the request's failure, reported and logged, never the service's end
            log(f"PROBLEM: {self.command} {self.path} failed: {type(error).__name__}: {error}")
            self.reply(500, {"error": f"{type(error).__name__}: the platform service could not do this; see its log"})
            return
        self.reply(status, payload)

    def json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        text = self.rfile.read(length).decode() if length else ""
        return json.loads(text) if text.strip() else {}

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/dev/reset":
            def reset():
                outcome = Mirror(CONFIG, "development").reset()
                log(f"development reset: restored {outcome['restored']} objects in {outcome['seconds']} s, clean={outcome['clean']}")
                return outcome
            self.answer(reset)
            return
        if parsed.path == "/releases":
            def request():
                body = self.json_body()
                release = RELEASES.request(commit=body.get("commit"), also=body.get("also"))
                log(f"release {release['release']} requested at {release['commit'][:12]}: staging")
                threading.Thread(target=release_step, args=("stage", release["release"]), daemon=True).start()
                return RELEASES.public_view(release)
            self.answer(request, status=202)
            return
        if len(parts) == 3 and parts[0] == "releases" and parts[2] in ("publish", "rollback"):
            def step():
                release = RELEASES.load(parts[1])
                allowed = ("staged",) if parts[2] == "publish" else ("published", "publish_failed", "publish_error", "publishing",
                                                                     "rollback_incomplete", "rollback_error", "rolling_back")
                if release["state"] not in allowed:
                    raise ReleaseRefused(f"release {parts[1]} is {release['state']}; {parts[2]} needs one of: {', '.join(allowed)}")
                log(f"release {parts[1]}: {parts[2]} requested")
                if parts[2] == "rollback" and release["state"] != "publishing":
                    # Whoever polls must not read the end state of an earlier attempt as this one's.
                    release["state"] = "rolling_back"
                    RELEASES.save(release)
                threading.Thread(target=release_step, args=(parts[2], parts[1]), daemon=True).start()
                return {"release": parts[1], "requested": parts[2]}
            self.answer(step, status=202)
            return
        if parsed.path != "/runs":
            self.reply(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > CONFIG["server"]["max_upload_bytes"]:
            self.reply(400, {"error": f"the submission is larger than {CONFIG['server']['max_upload_bytes']} bytes"})
            return
        requested = urllib.parse.parse_qs(parsed.query).get("tickets", [""])[0]
        tickets = [name for name in requested.split(",") if name] or list(TICKETS)
        unknown = [name for name in tickets if name not in TICKETS]
        if unknown:
            self.reply(400, {"error": f"unknown tickets: {', '.join(unknown)}", "known": list(TICKETS)})
            return
        body = self.rfile.read(length)
        run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{secrets.token_hex(3)}"
        run_dir = RUNS_DIR / run_id
        candidate_dir = run_dir / "candidate"
        candidate_dir.mkdir(parents=True)
        refusal = unpack_submission(body, candidate_dir)
        if refusal:
            self.reply(400, {"error": refusal})
            return
        run = {"run": run_id, "state": "queued", "tickets": tickets, "results": {}}
        with RUNS_LOCK:
            RUNS[run_id] = run
        threading.Thread(target=execute_run, args=(run, candidate_dir, run_dir), daemon=True).start()
        log(f"run {run_id} accepted: {len(tickets)} tickets, {length} bytes")
        self.reply(202, {"run": run_id, "tickets": tickets})


def release_step(step, release_id):
    """One step of a release in the background. A step that raises leaves the release marked with what failed,
    so whoever polls it reads an end state and not a step that never finishes."""
    try:
        release = getattr(RELEASES, step)(release_id)
        log(f"release {release_id} {step}: now {release['state']}")
    except Exception as error:   # noqa: BLE001 - recorded on the release and in the log
        log(f"PROBLEM: release {release_id} {step} failed: {type(error).__name__}: {error}")
        release = RELEASES.load(release_id)
        release["state"] = f"{step}_error"
        release["error"] = f"{type(error).__name__} during {step}; see the platform service's log"
        RELEASES.save(release)


def pool_round():
    recovered = POOL.recover_stale()
    if recovered:
        log(f"recovered check databases a check never gave back: {recovered}")
    dropped = POOL.replace_dirty()
    if dropped:
        log(f"PROBLEM: dropped dirty check databases {dropped}; {POOL.clones_left_today()} new clones allowed today")
    POOL.fill(wait=False)


def keep_pool_filled():
    """One round every interval, each in a thread of its own: a round that raises (a network blip, Snowflake
    refusing a statement) prints its traceback to the log and the next round still runs."""
    while True:
        worker = threading.Thread(target=pool_round)
        worker.start()
        worker.join()
        time.sleep(POOL_INTERVAL_SECONDS)


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=keep_pool_filled, daemon=True).start()
    address = (CONFIG["server"]["host"], CONFIG["server"]["port"])
    log(f"ticket-check service on http://{address[0]}:{address[1]}, {len(TICKETS)} tickets, {CONFIG['pool']['databases']} check databases")
    ThreadingHTTPServer(address, Handler).serve_forever()
