{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Testing]], [[domain::Utilities]], [[domain::Quality Assurance]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Comprehensive testing utilities module providing decorators, fixtures, and helper functions for testing PyTorch models across multiple hardware backends in the Transformers library.

=== Description ===
The testing_utils.py module is a 4366-line comprehensive testing infrastructure that provides extensive utilities for testing transformer models. It includes:

* Environment-based test decorators for conditional test execution (slow tests, pipeline tests, agent tests, training tests)
* Hardware requirement decorators (GPU, multi-GPU, TPU, NPU, XPU, HPU, etc.)
* Dependency requirement decorators (various libraries like torch, tensorflow, accelerate, etc.)
* Device-agnostic testing utilities for cross-platform compatibility
* Memory monitoring and profiling utilities
* Context managers for capturing stdout/stderr and logging
* Subprocess execution utilities for distributed testing
* Docker testing support via pytest integration
* Helper classes for test case management (TestCasePlus)
* Temporary directory and Hub repository management
* Flaky test handling with retry mechanisms
* Custom assertion formatters for debugging

Key features include backend abstraction (supporting CUDA, ROCm, XPU, NPU, MLU, HPU), pytest plugin integration for custom doctest parsing, and comprehensive memory tracking utilities for both CPU and GPU.

=== Usage ===
Use this module when writing or running tests for Transformers models, particularly when you need to:
* Skip tests based on hardware availability or environment variables
* Test models across different hardware backends
* Monitor memory usage during testing
* Execute distributed training tests
* Capture and analyze test output
* Handle flaky tests with automatic retries
* Create temporary test environments
* Validate model behavior across platforms

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/testing_utils.py

=== Signature ===
<syntaxhighlight lang="python">
# Key decorators for test execution control
def slow(test_case)
def require_torch(test_case)
def require_torch_gpu(test_case)
def require_torch_multi_gpu(test_case)
def require_accelerate(test_case, min_version: str = ACCELERATE_MIN_VERSION)

# Memory monitoring
class CPUMemoryMonitor:
    def __init__(self)
    def get_stats(self) -> MemoryStats
    def reset_peak_stats(self) -> None

# Device-agnostic utilities
def backend_manual_seed(device: str, seed: int)
def backend_empty_cache(device: str)
def backend_device_count(device: str)

# Test case base class
class TestCasePlus(unittest.TestCase):
    def setUp(self)
    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None)
    def get_env(self)

# Context managers
class CaptureStd:
    def __init__(self, out=True, err=True, replay=True)

class CaptureLogger:
    def __init__(self, logger)

# Flaky test handling
def is_flaky(max_attempts: int = 5, wait_before_retry: float | None = None,
             description: str | None = None)

# Subprocess execution
def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180,
                             quiet=False, echo=True) -> _RunOutput
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.testing_utils import (
    slow,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    TestCasePlus,
    CaptureStd,
    CaptureLogger,
    is_flaky,
    backend_manual_seed,
    backend_empty_cache,
    CPUMemoryMonitor,
)
</syntaxhighlight>

== I/O Contract ==

=== Environment Variables ===
{| class="wikitable"
! Variable !! Type !! Description
|-
| RUN_SLOW || bool || Enable slow tests (default: False)
|-
| RUN_FLAKY || bool || Enable flaky tests (default: True)
|-
| RUN_PIPELINE_TESTS || bool || Enable pipeline tests (default: True)
|-
| RUN_AGENT_TESTS || bool || Enable agent tests (default: False)
|-
| RUN_TRAINING_TESTS || bool || Enable training tests (default: True)
|-
| TRANSFORMERS_TEST_DEVICE || str || Override test device (e.g., "cuda", "cpu")
|-
| TRANSFORMERS_TEST_BACKEND || str || Override test backend module
|-
| PYTEST_CURRENT_TEST || str || Current test identifier (set by pytest)
|}

=== Device Detection Outputs ===
{| class="wikitable"
! Variable !! Type !! Description
|-
| torch_device || str || Selected device ("cuda", "cpu", "xpu", "npu", etc.)
|-
| IS_CUDA_SYSTEM || bool || True if CUDA is available
|-
| IS_ROCM_SYSTEM || bool || True if ROCm is available
|-
| IS_XPU_SYSTEM || bool || True if Intel XPU is available
|-
| IS_NPU_SYSTEM || bool || True if NPU is available
|}

=== Memory Statistics ===
{| class="wikitable"
! Field !! Type !! Description
|-
| rss_gib || float || Resident Set Size in GiB
|-
| rss_pct || float || RSS as percentage of total memory
|-
| vms_gib || float || Virtual Memory Size in GiB
|-
| peak_rss_gib || float || Peak RSS in GiB
|-
| peak_rss_pct || float || Peak RSS as percentage
|-
| available_gib || float || Available system memory in GiB
|-
| total_gib || float || Total system memory in GiB
|}

== Usage Examples ==

=== Basic Test Decorator Usage ===
<syntaxhighlight lang="python">
import unittest
from transformers.testing_utils import slow, require_torch, require_torch_gpu

class MyModelTests(unittest.TestCase):

    @require_torch
    def test_model_forward(self):
        # Test that requires PyTorch to be installed
        pass

    @slow
    @require_torch_gpu
    def test_large_model(self):
        # Test that runs only with RUN_SLOW=1 and requires GPU
        pass
</syntaxhighlight>

=== Device-Agnostic Testing ===
<syntaxhighlight lang="python">
from transformers.testing_utils import (
    backend_manual_seed,
    backend_empty_cache,
    torch_device,
)

# Set seed for any device
backend_manual_seed(torch_device, 42)

# Run model
model = MyModel().to(torch_device)
output = model(input_ids)

# Clear cache for any device
backend_empty_cache(torch_device)
</syntaxhighlight>

=== Memory Monitoring ===
<syntaxhighlight lang="python">
from transformers.testing_utils import build_cpu_memory_monitor, init_test_logger

logger = init_test_logger()
monitor = build_cpu_memory_monitor(logger)

# Your test code here
model.train()

# Get memory statistics
stats = monitor.get_stats()
logger.info(f"Peak RSS: {stats.peak_rss_gib:.2f} GiB ({stats.peak_rss_pct:.1f}%)")
</syntaxhighlight>

=== Capture Output ===
<syntaxhighlight lang="python">
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    print("Secret message")
    print("Warning!", file=sys.stderr)

assert "message" in cs.out
assert "Warning" in cs.err
</syntaxhighlight>

=== Handling Flaky Tests ===
<syntaxhighlight lang="python">
from transformers.testing_utils import is_flaky

@is_flaky(max_attempts=3, wait_before_retry=2.0,
          description="Network-dependent test that may timeout")
def test_hub_download(self):
    model = AutoModel.from_pretrained("bert-base-uncased")
    assert model is not None
</syntaxhighlight>

=== TestCasePlus Usage ===
<syntaxhighlight lang="python">
from transformers.testing_utils import TestCasePlus

class MyTests(TestCasePlus):
    def test_with_temp_dir(self):
        # Automatically cleaned up temporary directory
        tmp_dir = self.get_auto_remove_tmp_dir()

        # Access repository paths
        model_path = os.path.join(self.src_dir_str, "transformers", "models")

        # Get environment with proper PYTHONPATH
        env = self.get_env()
</syntaxhighlight>

=== Subprocess Execution for Distributed Tests ===
<syntaxhighlight lang="python">
from transformers.testing_utils import execute_subprocess_async

cmd = ["torchrun", "--nproc_per_node", "2", "train_script.py"]
result = execute_subprocess_async(cmd, env=os.environ.copy())

# Check results
assert result.returncode == 0
</syntaxhighlight>

=== Expectations for Multi-Device Testing ===
<syntaxhighlight lang="python">
from transformers.testing_utils import Expectations

# Define expected values per device
expected_output = Expectations({
    ("cuda", (8, 0)): 0.123,  # CUDA 8.0
    ("rocm", 9): 0.124,        # ROCm 9.x
    (None, None): 0.125,       # Default fallback
})

# Get expectation for current device
expected = expected_output.get_expectation()
assert abs(output - expected) < 1e-3
</syntaxhighlight>

== Related Pages ==
* (Empty)
