# Stop the deployment: the platform puts production back to before this release.
set -euo pipefail
python3 "$(dirname "$0")/release.py" rollback
