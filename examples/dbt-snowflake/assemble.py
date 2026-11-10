"""Assemble the team repository from this example's template and a checkout of data-eng-bench.

    python assemble.py <data-eng-bench checkout> <out dir>

`<out dir>/warehouse` is the benchmark's Snowflake dbt project (`base-image/dbt_models_snowflake`) with the
seven files its fix tasks start from in place: the six models and the snapshot that produce the wrong numbers.
`<out dir>/marketing` and everything else (the README, the notice, the tickets, the checks' clients, the Kapso
config) come from `repo-template/`. Only the tasks' `environment/` files are read: nothing of the benchmark's
solutions or tests is opened or copied.
"""
import pathlib
import shutil
import sys

TEMPLATE = pathlib.Path(__file__).resolve().parent / "repo-template"
BASE_PROJECT = pathlib.Path("base-image") / "dbt_models_snowflake"
# The starting files the benchmark installs for its fix tasks: task file -> path in the warehouse project.
BROKEN = {
    "dbt-fix-daily-revenue/environment/rpt_order_daily_summary.sql": "models/marts/sales/rpt_order_daily_summary.sql",
    "dbt-fix-customer-ltv/environment/dim_customers.sql": "models/marts/core/dim_customers.sql",
    "dbt-fix-timezone-sales/environment/int_sales__orders_enriched.sql": "models/intermediate/sales/int_sales__orders_enriched.sql",
    "dbt-fix-refund-reconciliation/environment/data/rpt_return_reconciliation.sql": "models/marts/sales/rpt_return_reconciliation.sql",
    "dbt-fix-inventory-model/environment/data/rpt_product_inventory_metrics.sql": "models/marts/inventory/rpt_product_inventory_metrics.sql",
    "dbt-fix-category-revenue/environment/rpt_category_performance.sql": "models/marts/product/rpt_category_performance.sql",
    "dbt-fix-customer-snapshot-and-build-dimension/environment/data/customer_snapshot.sql": "snapshots/customer_snapshot.sql",
}

if len(sys.argv) != 3:
    raise SystemExit("usage: python assemble.py <data-eng-bench checkout> <out dir>")
checkout, out = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
required = [checkout / BASE_PROJECT / "dbt_project.yml", *(checkout / "tasks" / source for source in BROKEN)]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"{checkout} is not a checkout of data-eng-bench with these tasks; missing: {', '.join(missing[:3])}")
if out.exists():
    raise SystemExit(f"{out} exists: remove it or name another directory")

shutil.copytree(TEMPLATE, out)
warehouse = out / "warehouse"
shutil.copytree(checkout / BASE_PROJECT, warehouse, ignore=shutil.ignore_patterns("profiles.yml", "target", "logs", "dbt_packages"))
for source, target in BROKEN.items():
    destination = warehouse / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkout / "tasks" / source, destination)

files = sum(1 for path in out.rglob("*") if path.is_file())
print(f"{out}: {files} files, {sum(1 for _ in warehouse.rglob('*.sql'))} SQL files in warehouse, "
      f"{len(BROKEN)} starting files placed, {len(list((out / 'tickets').glob('*.md')))} tickets")
