{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Performance_Testing]], [[domain::Metrics]], [[domain::Data_Collection]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

A dual-mode metrics recording system that collects benchmark data into both PostgreSQL databases and CSV files with automatic benchmark discovery and execution.

=== Description ===

The `benchmarks_entrypoint.py` module provides a comprehensive benchmarking framework with flexible data storage options. The core `MetricsRecorder` class handles data collection to either PostgreSQL (via psycopg2) or CSV files (via pandas), or both simultaneously.

Key architectural features:
* **Dual storage mode**: Database and/or CSV with configurable fallback
* **UUID-based benchmark tracking**: Each benchmark run gets a unique identifier
* **Structured data collection**: Separate tables/DataFrames for benchmarks, device measurements, and model measurements
* **Automatic benchmark discovery**: Scans `benches/` directory for modules with `run_benchmark()` functions
* **Flexible CSV export**: Generates individual and summary CSV files with pandas-based aggregations
* **Graceful degradation**: Continues with CSV-only mode if database is unavailable

The module supports three data collection patterns:
1. **Benchmarks metadata**: Repository, branch, commit info, and custom metadata
2. **Device measurements**: CPU/GPU utilization and memory usage over time
3. **Model measurements**: Custom timing metrics (load time, forward pass, generation, etc.)

The entrypoint script (`__main__` section) orchestrates the entire benchmarking pipeline:
* Parses command-line arguments (repository, branch, commit, CSV options)
* Creates a global `MetricsRecorder` instance
* Discovers benchmark modules in `benches/` directory
* Executes each benchmark's `run_benchmark()` function
* Exports aggregated CSV results with comprehensive summaries

=== Usage ===

Use this module as the main entrypoint for running all benchmarks in the Transformers repository. It's designed for CI/CD pipelines where you need to track performance metrics over time, with the flexibility to store results in a database for long-term analysis or CSV files for immediate inspection and portability.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/benchmark/benchmarks_entrypoint.py benchmark/benchmarks_entrypoint.py]
* '''Lines:''' 1-502

=== Signature ===
<syntaxhighlight lang="python">
class MetricsRecorder:
    def __init__(
        self,
        connection,
        logger: logging.Logger,
        repository: str,
        branch: str,
        commit_id: str,
        commit_msg: str,
        collect_csv_data: bool = True,
    ):
        pass

    def initialise_benchmark(self, metadata: dict[str, str]) -> str:
        """Create benchmark and return UUID"""
        pass

    def collect_device_measurements(
        self,
        benchmark_id: str,
        cpu_util: float,
        mem_megabytes: float,
        gpu_util: float,
        gpu_mem_megabytes: float
    ):
        """Record device utilization metrics"""
        pass

    def collect_model_measurements(
        self,
        benchmark_id: str,
        measurements: dict[str, float]
    ):
        """Record model timing measurements"""
        pass

    def export_to_csv(self, output_dir: str = "benchmark_results"):
        """Export all collected data to CSV files"""
        pass

    def close(self):
        """Close database connection"""
        pass


def create_global_metrics_recorder(
    repository: str,
    branch: str,
    commit_id: str,
    commit_msg: str,
    generate_csv: bool = False
) -> MetricsRecorder:
    """Factory function for creating recorder"""
    pass


def create_database_connection():
    """Attempt database connection, return None on failure"""
    pass


def import_from_path(module_name: str, file_path: str):
    """Dynamic module import from file path"""
    pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from benchmarks_entrypoint import MetricsRecorder, create_global_metrics_recorder
import logging
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| repository || str || Yes || Repository name (CLI positional arg 1)
|-
| branch || str || Yes || Branch name (CLI positional arg 2)
|-
| commit_id || str || Yes || Commit SHA (CLI positional arg 3)
|-
| commit_msg || str || Yes || Commit message truncated to 70 chars (CLI positional arg 4)
|-
| --csv || flag || No || Enable CSV output generation (disabled by default)
|-
| --csv-output-dir || str || No || Directory for CSV files (default: benchmark_results)
|-
| connection || psycopg2.connection || No || Database connection (None for CSV-only mode)
|-
| metadata || dict[str, str] || Yes (for initialise) || Custom metadata for benchmark run
|-
| measurements || dict[str, float] || Yes (for collect_model) || Model timing measurements
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| benchmark_id || str (UUID) || Unique identifier for each benchmark run
|-
| benchmarks_{timestamp}.csv || CSV file || Benchmark metadata records
|-
| device_measurements_{timestamp}.csv || CSV file || Device utilization time series
|-
| model_measurements_{timestamp}.csv || CSV file || Model timing measurements (flattened)
|-
| benchmark_summary_{timestamp}.csv || CSV file || Comprehensive aggregated summary with stats
|-
| Database records || PostgreSQL || Records inserted into benchmarks, device_measurements, model_measurements tables
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Basic CLI usage (database + CSV)
import subprocess

cmd = [
    "python", "benchmark/benchmarks_entrypoint.py",
    "huggingface/transformers",  # repository
    "main",                       # branch
    "abc123def456",               # commit_id
    "Fix generation bug",         # commit_msg
    "--csv",                      # Enable CSV output
    "--csv-output-dir", "results"
]
subprocess.run(cmd)


# Example 2: Programmatic usage - create recorder and collect metrics
from benchmarks_entrypoint import create_global_metrics_recorder
import logging

logger = logging.getLogger(__name__)

# Create recorder (tries database first, falls back to CSV-only)
recorder = create_global_metrics_recorder(
    repository="huggingface/transformers",
    branch="main",
    commit_id="abc123",
    commit_msg="Optimize attention",
    generate_csv=True
)

# Initialize a new benchmark
metadata = {
    "model_name": "google/gemma-2b",
    "batch_size": "1",
    "sequence_length": "512"
}
benchmark_id = recorder.initialise_benchmark(metadata)
print(f"Started benchmark: {benchmark_id}")

# Collect device measurements (e.g., in a monitoring loop)
recorder.collect_device_measurements(
    benchmark_id=benchmark_id,
    cpu_util=45.2,
    mem_megabytes=8192,
    gpu_util=78.5,
    gpu_mem_megabytes=16384
)

# Collect model measurements
measurements = {
    "model_load_time": 2.34,
    "first_eager_forward_pass_time_secs": 0.123,
    "time_to_first_token_secs": 0.045,
    "time_to_next_token_mean_secs": 0.012
}
recorder.collect_model_measurements(benchmark_id, measurements)

# Export results to CSV
recorder.export_to_csv("my_benchmark_results")
recorder.close()


# Example 3: CSV-only mode (no database)
from benchmarks_entrypoint import MetricsRecorder
import logging

logger = logging.getLogger(__name__)

# Create recorder with no database connection
recorder = MetricsRecorder(
    connection=None,  # CSV-only mode
    logger=logger,
    repository="huggingface/transformers",
    branch="dev",
    commit_id="xyz789",
    commit_msg="Add new model",
    collect_csv_data=True
)

# Use same API as before
benchmark_id = recorder.initialise_benchmark({"model": "gpt2"})
recorder.collect_device_measurements(benchmark_id, 30.0, 4096, 50.0, 8192)
recorder.export_to_csv()
recorder.close()


# Example 4: Creating a custom benchmark module
# Save as benchmark/benches/my_custom_benchmark.py
import time
import logging

def run_benchmark(
    logger: logging.Logger,
    repository: str,
    branch: str,
    commit_id: str,
    commit_msg: str,
    metrics_recorder  # MetricsRecorder instance
):
    """
    This function will be auto-discovered and executed by benchmarks_entrypoint.py
    """
    logger.info("Running custom benchmark")

    # Initialize benchmark
    metadata = {"test_name": "custom_test", "framework": "pytorch"}
    benchmark_id = metrics_recorder.initialise_benchmark(metadata)

    # Simulate some work and collect metrics
    start = time.time()
    # ... do benchmark work ...
    elapsed = time.time() - start

    # Record measurements
    metrics_recorder.collect_model_measurements(
        benchmark_id,
        {"total_time_secs": elapsed, "iterations": 100}
    )

    logger.info(f"Completed benchmark {benchmark_id}")


# Example 5: Analyzing CSV output with pandas
import pandas as pd
import glob

# Find the latest benchmark results
csv_files = glob.glob("benchmark_results/benchmark_summary_*.csv")
latest_csv = sorted(csv_files)[-1]

# Load and analyze
df = pd.read_csv(latest_csv)

# Show summary statistics
print("Device utilization statistics:")
print(df[['cpu_util_mean', 'cpu_util_max', 'mem_megabytes_max']].describe())

# Filter by specific model
gemma_runs = df[df['metadata'].str.contains('gemma')]
print(f"\nFound {len(gemma_runs)} Gemma benchmark runs")

# Compare timing metrics
timing_cols = [col for col in df.columns if 'time' in col.lower()]
print("\nTiming metrics:")
print(df[timing_cols].head())


# Example 6: Database-only mode (disable CSV)
recorder = create_global_metrics_recorder(
    repository="huggingface/transformers",
    branch="main",
    commit_id="abc123",
    commit_msg="Test commit",
    generate_csv=False  # No CSV files will be created
)

benchmark_id = recorder.initialise_benchmark({"model": "bert-base"})
# Data only goes to database
recorder.collect_device_measurements(benchmark_id, 40.0, 6144, 60.0, 12288)
recorder.close()
</syntaxhighlight>

== Related Pages ==

* (Leave empty for orphan pages)
