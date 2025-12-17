# File: `scripts/launch_notebook_mp.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 47 |
| Classes | `MyModule` |
| Functions | `init`, `main` |
| Imports | accelerate, peft, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Regression test that verifies PEFT can be launched via Accelerate's notebook_launcher with multiprocessing without encountering CUDA re-initialization errors.

**Mechanism:** The init() function creates a simple PyTorch linear layer model, infers the device, and wraps it with LoraConfig targeting the linear module. The main() function uses notebook_launcher() to spawn 2 processes running init(). This test historically failed because PEFT eagerly imported bitsandbytes, which initialized CUDA in the parent process, causing "Cannot re-initialize CUDA in forked subprocess" errors when using fork-based multiprocessing.

**Significance:** Critical regression test ensuring PEFT maintains compatibility with Accelerate's multiprocessing features. Prevents reintroduction of eager CUDA initialization bugs that would break distributed training workflows. Essential for users running PEFT on multi-GPU systems or in Jupyter notebooks with parallel execution.
