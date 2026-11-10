"""Put the development database back to production's state.

    python checks/dev_reset.py

The data platform restores in RETAIL_DW_DEV every table a build replaced or created, from production
(RETAIL_DW), drops the schemas production does not have, and compares the two inventories. It takes seconds,
and up to a minute after a build that replaced a hundred tables.
Run it before taking a ticket's baseline and before building a ticket's models, so that what is measured
starts from what production holds: the ticket checks build each ticket's models on production's state too.

Exit codes: 0 restored and verified; 2 the service could not be reached, or the database did not come back
equal to production (the reason is printed).
"""
import sys

from platform_service import call, require_service

require_service("dev reset")
status, outcome = call("POST", "/dev/reset", timeout_key="reset_timeout_seconds")
if status != 200:
    print(f"dev reset: the platform service could not reset the development database ({status}): {outcome.get('error')}")
    sys.exit(2)
if not outcome["clean"]:
    print(f"dev reset: {outcome['database']} still differs from production after the reset: {outcome.get('problems')}")
    sys.exit(2)
dropped = f", dropped schemas {', '.join(outcome['dropped_schemas'])}" if outcome.get("dropped_schemas") else ""
print(f"dev reset: {outcome['database']} is in production's state ({outcome['restored']} object(s) restored{dropped}, {outcome['seconds']} s)")
