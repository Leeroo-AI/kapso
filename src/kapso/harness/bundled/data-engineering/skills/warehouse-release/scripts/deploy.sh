# The warehouse release: ask the platform to rehearse what is merged, audit the rehearsal with every released
# ticket's checks, publish, and check production. Run from the workspace. Prints KAPSO_ENDPOINT and the
# KAPSO_VALUE_ lines the result file carries.
set -euo pipefail
python3 "$(dirname "$0")/release.py" deploy
