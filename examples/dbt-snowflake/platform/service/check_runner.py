"""Run one ticket's acceptance check on a submitted project, in a container, against one fresh clone.

The steps are the benchmark runner's own, reproduced one for one (Harbor 0.22: environments/docker, agents/oracle,
verifier/verifier), so the hidden tests see the environment they were written for:

  1. A container from the benchmark's base image, started the way the task's compose file starts it (the same
     entrypoint override, the same environment variables), with the submission mounted read-only at /candidate.
  2. Where Harbor's healthcheck would clone the source database, the clone the pool prepared is assigned: the
     same /tmp/snowflake_env.sh and /tmp/snowflake_clone_name.txt the image's own clone step writes.
  3. The ticket's build recipe as /solution/solve.sh, run as `bash -c "(/solution/solve.sh) > /logs/agent/oracle.txt 2>&1"`:
     the benchmark's reference scaffolding for that task with the submission's files in place of the reference's.
  4. The task's own tests directory copied to /tests and run as
     `bash -c "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"`; the verdict is /logs/verifier/reward.txt.

What leaves this module per ticket is the verdict, how the build went, and, when the build failed, the
classification of the build's errors (build_errors.py): the build's log and the tests' output stay here.

    python check_runner.py <candidate dir> <ticket> [<ticket> ...]      # a manual run, results under runs/manual-<time>
"""
import base64
import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from build_errors import classify
from pool import CheckDatabases, load_config

PASSTHROUGH = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_HOST", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_PRIVATE_KEY",
               "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_ROLE", "SNOWFLAKE_SCHEMA"]


def container_environment(config):
    """The variables the task compose file gives the container, for the checks identity."""
    snowflake = config["snowflake"]
    checks = snowflake["checks"]
    private_key = base64.b64encode(pathlib.Path(checks["private_key_file"]).read_bytes()).decode()
    return {
        "TEST_DIR": "/tests", "DB_TYPE": "snowflake", "DUCKDB_PATH": "/app/database/retail.duckdb",
        "SNOWFLAKE_ACCOUNT": snowflake["account"], "SNOWFLAKE_HOST": f"{snowflake['account']}.snowflakecomputing.com",
        "SNOWFLAKE_PASSWORD": "", "SNOWFLAKE_USER": checks["user"], "SNOWFLAKE_PRIVATE_KEY": private_key,
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": "", "SNOWFLAKE_SOURCE_DATABASE": snowflake["source_database"],
        "SNOWFLAKE_SCHEMA": snowflake["schema"], "SNOWFLAKE_WAREHOUSE": checks["warehouse"], "SNOWFLAKE_ROLE": checks["role"],
        # The reference scaffolding of some tasks creates a schema "with the admin role"; inside its own clone
        # the checks role is that role.
        "SNOWFLAKE_ADMIN_ROLE": checks["role"],
    }


def clone_environment_file(environment, clone):
    """The text of /tmp/snowflake_env.sh exactly as the image's /snowflake_clone.py writes it for `clone`."""
    lines = [f'export SNOWFLAKE_DATABASE="{clone}"', f'export SNOWFLAKE_CLONE_NAME="{clone}"', 'export DB_TYPE="snowflake"']
    for name in PASSTHROUGH:
        escaped = environment[name].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'export {name}="{escaped}"')
    return "\n".join(lines) + "\n"


def docker(arguments, extra_environment=None, timeout=None):
    """One docker command; returns (return code, combined output). With a timeout the output is discarded
    (the commands run that way redirect their own output to a file) and running out of time returns code 124."""
    environment = {**os.environ, **(extra_environment or {})}
    if timeout is None:
        completed = subprocess.run(["docker", *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
        return completed.returncode, completed.stdout or ""
    process = subprocess.Popen(["docker", *arguments], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment)
    deadline = time.time() + timeout
    while process.poll() is None and time.time() < deadline:
        time.sleep(1)
    if process.poll() is None:
        process.kill()
        process.wait()
        return 124, ""
    return process.returncode, ""


def must(result, what):
    code, output = result
    if code != 0:
        raise RuntimeError(f"{what} failed with code {code}: {output[-600:]}")
    return output


def run_ticket_check(config, ticket, spec, candidate_dir, clone, work_dir):
    """The four steps for one ticket. Returns {"ticket", "passed", "build", "seconds", "build_errors"?}."""
    started = time.time()
    work_dir = pathlib.Path(work_dir)
    for name in ("agent", "verifier", "staging"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)
        (work_dir / name).chmod(0o777)
    environment = container_environment(config)
    container = f"check-{ticket}-{clone[-6:].lower()}"
    limits = config["container"]
    task_dir = pathlib.Path(config["tasks_dir"]) / spec["task"]
    recipe = pathlib.Path(config["recipes_dir"]) / f"{ticket}.sh"
    tests_dir = task_dir / "tests"
    for required in (recipe, tests_dir / "test.sh", tests_dir / "test_outputs.py"):
        if not required.exists():
            raise FileNotFoundError(required)

    docker(["rm", "-f", container])   # a container an interrupted check left under this name, if any
    run_arguments = ["run", "-d", "--name", container, "--entrypoint", "sh", "--cpus", str(limits["cpus"]),
                     "--memory", limits["memory"], "-v", f"{work_dir / 'verifier'}:/logs/verifier",
                     "-v", f"{work_dir / 'agent'}:/logs/agent", "-v", f"{pathlib.Path(candidate_dir).resolve()}:/candidate:ro"]
    for name in environment:
        run_arguments += ["-e", name]
    run_arguments += [config["image"], "-c", "touch /tmp/entrypoint_ready && sleep infinity"]
    # Values travel in the docker client's own environment, so no secret appears on a command line.
    must(docker(run_arguments, extra_environment=environment), "starting the container")

    result = {"ticket": ticket, "passed": False, "build": "not_run"}
    # Step 2: the clone assignment, as the image's clone step leaves it.
    (work_dir / "staging" / "snowflake_env.sh").write_text(clone_environment_file(environment, clone))
    (work_dir / "staging" / "snowflake_clone_name.txt").write_text(clone + "\n")
    must(docker(["cp", str(work_dir / "staging" / "snowflake_env.sh"), f"{container}:/tmp/snowflake_env.sh"]), "placing the clone environment")
    must(docker(["cp", str(work_dir / "staging" / "snowflake_clone_name.txt"), f"{container}:/tmp/snowflake_clone_name.txt"]), "placing the clone name")

    # Step 3: the build recipe, run as Harbor's oracle agent runs a solution.
    solution_dir = work_dir / "staging" / "solution"
    solution_dir.mkdir(exist_ok=True)
    shutil.copy(recipe, solution_dir / "solve.sh")
    shutil.copy(recipe.parent / "lib.sh", solution_dir / "lib.sh")
    must(docker(["cp", str(solution_dir), f"{container}:/solution"]), "copying the build recipe")
    must(docker(["exec", "-u", "root", container, "bash", "-c", "chmod +x /solution/solve.sh"]), "chmod of the build recipe")
    build_code, _ = docker(["exec", "-e", "DEBIAN_FRONTEND=noninteractive", container, "bash", "-c",
                            "(/solution/solve.sh) > /logs/agent/oracle.txt 2>&1"], timeout=limits["build_timeout_seconds"])
    result["build"] = "ok" if build_code == 0 else ("timeout" if build_code == 124 else "failed")
    build_log = work_dir / "agent" / "oracle.txt"
    if build_code != 0 and build_log.exists():
        # The log itself stays here; the submitter gets its classification (build_errors.py says why).
        result["build_errors"] = classify(build_log.read_text(errors="replace"), candidate_dir)

    # Step 4: the task's own tests, run as Harbor's verifier runs them, whatever the build did.
    must(docker(["cp", str(tests_dir), f"{container}:/tests"]), "copying the tests")
    must(docker(["exec", "-u", "root", container, "bash", "-c", "chmod +x /tests/test.sh"]), "chmod of the test script")
    docker(["exec", container, "bash", "-c", "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"],
           timeout=limits["test_timeout_seconds"])
    reward_file = work_dir / "verifier" / "reward.txt"
    reward = reward_file.read_text().strip() if reward_file.exists() else ""
    result["passed"] = reward in ("1", "1.0")
    docker(["rm", "-f", container])
    result["seconds"] = round(time.time() - started)
    (work_dir / "result.json").write_text(json.dumps({**result, "clone": clone, "reward": reward}, indent=2) + "\n")
    return result


def check_one(config, pool, tickets, ticket, candidate_dir, run_dir, run_label):
    database = pool.acquire(f"{run_label}/{ticket}")
    result = run_ticket_check(config, ticket, tickets[ticket], candidate_dir, database, pathlib.Path(run_dir) / ticket)
    reset = pool.release(database)
    return {**result, "database": database, "reset": reset}


def run_checks(config, candidate_dir, ticket_names, run_dir, run_label):
    """Every named ticket's check, several at a time; one ticket's infrastructure error does not stop the others."""
    tickets = json.loads(pathlib.Path(config["tickets_file"]).read_text())
    unknown = [name for name in ticket_names if name not in tickets]
    if unknown:
        raise ValueError(f"unknown tickets: {unknown}")
    pool = CheckDatabases(config)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_parallel_checks"]) as executor:
        futures = {executor.submit(check_one, config, pool, tickets, name, candidate_dir, run_dir, run_label): name
                   for name in ticket_names}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            error = future.exception()
            if error is None:
                results[name] = future.result()
            else:
                results[name] = {"ticket": name, "passed": False, "build": "not_run", "check_error": f"{type(error).__name__}: {error}"}
    ordered = [results[name] for name in ticket_names]
    summary = {"label": run_label, "tickets": ordered, "passed": sum(1 for item in ordered if item["passed"]), "total": len(ordered)}
    (pathlib.Path(run_dir) / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    configuration = load_config()
    candidate = pathlib.Path(sys.argv[1]).resolve()
    names = sys.argv[2:]
    if names == ["all"]:
        names = list(json.loads(pathlib.Path(configuration["tickets_file"]).read_text()))
    label = f"manual-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    directory = pathlib.Path(configuration["runs_dir"]) / label
    directory.mkdir(parents=True)
    outcome = run_checks(configuration, candidate, names, directory, label)
    for item in outcome["tickets"]:
        note = item.get("check_error") or ""
        reset = item.get("reset") or {}
        after = f"{item.get('database', '')[-6:]} restored {reset.get('restored')} {'clean' if reset.get('clean') else 'NOT CLEAN ' + str(reset.get('problems'))}"
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['ticket']:<32} build {item['build']:<8} {item.get('seconds', '')!s:>5} s  {after}  {note}")
    print(f"{outcome['passed']} of {outcome['total']} tickets pass   ({directory})")
