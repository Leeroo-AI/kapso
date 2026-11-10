# Shared by the pull-request target's scripts: where the deployment keeps its files, and how the repository,
# the base branch and the label are named. Sourced, never run.
MANIFEST=deploy/pull_requests/manifest.json
PREPARED=deploy/pull_requests/prepared.json
BRANCH_CHECKS=deploy/pull_requests/branch_checks.json
OPENED=deploy/pull_requests/opened.json
BASE="${KAPSO_VALUE_BASE:-${KAPSO_OPTION_BASE:-main}}"
LABEL="${KAPSO_VALUE_LABEL:-kapso-${KAPSO_DEPLOYMENT_NAME:?the engine sets KAPSO_DEPLOYMENT_NAME}}"

# owner/name from a GitHub remote URL, or nothing when the URL is not GitHub's.
github_slug() {
    echo "$1" | sed -nE 's#^(https://github\.com/|git@github\.com:)([^/]+/[^/]+)$#\2#p' | sed -E 's#\.git$##'
}

# owner/name of the repository the pull requests are opened against: the recorded value, the call's option,
# where the engine says the solution's code came from (KAPSO_SOURCE_REMOTE: a deployment workspace is a copy
# of the solution with no remote of its own), or the workspace's own origin.
repo_slug() {
    if [ -n "${KAPSO_VALUE_REPO:-}" ]; then echo "$KAPSO_VALUE_REPO"; return; fi
    if [ -n "${KAPSO_OPTION_REPO:-}" ]; then echo "$KAPSO_OPTION_REPO"; return; fi
    local url slug
    for url in "${KAPSO_SOURCE_REMOTE:-}" "$(git remote get-url origin 2>/dev/null || true)"; do
        slug="$(github_slug "$url")"
        if [ -n "$slug" ]; then echo "$slug"; return; fi
    done
    echo "no GitHub repository named: pass --option repo=<owner>/<name> (the solution's code came from '${KAPSO_SOURCE_REMOTE:-nowhere the engine knows}')" >&2
    return 1
}

# The numbers of the pull requests this deployment opened, one per line.
pr_numbers() {
    if [ -n "${KAPSO_VALUE_PR_NUMBERS:-}" ]; then
        echo "$KAPSO_VALUE_PR_NUMBERS" | tr ',' '\n'
        return
    fi
    if [ ! -f "$OPENED" ]; then
        echo "no $OPENED and no recorded pr_numbers: nothing was opened by this deployment" >&2
        return 1
    fi
    python3 -c 'import json, sys; print("\n".join(str(pr["number"]) for pr in json.load(open(sys.argv[1]))["pull_requests"]))' "$OPENED"
}
