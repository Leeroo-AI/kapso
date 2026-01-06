# File: `scripts/launch_notebook_mp.py`

**Category:** test script

| Property | Value |
|----------|-------|
| Lines | 48 |
| Classes | `MyModule` |
| Functions | `init`, `main` |
| Imports | accelerate.notebook_launcher, peft, peft.utils.infer_device, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Regression test script that ensures PEFT can be launched with Accelerate's multiprocessing without encountering CUDA re-initialization errors. This addresses a historical issue where PEFT would eagerly import bitsandbytes, causing CUDA initialization in the parent process.

**Mechanism:**
- `init()`: Worker function that runs in each subprocess:
  - Creates a minimal PyTorch model (`MyModule` with single linear layer)
  - Moves model to appropriate device (inferred automatically)
  - Wraps model with PEFT using LoraConfig
  - Target modules set to ["linear"] to enable LoRA on the linear layer

- `main()`: Entry point that:
  - Launches `init()` function with 2 parallel processes using `notebook_launcher`
  - notebook_launcher handles multiprocessing with proper CUDA context management

Historical context (from comments):
- Previously caused: "RuntimeError: Cannot re-initialize CUDA in forked subprocess"
- Root cause: PEFT's eager bitsandbytes import initialized CUDA in parent process
- This script exists as a guard against regression of that bug

**Significance:** Critical regression test for multiprocessing compatibility. Ensures PEFT maintains compatibility with Accelerate's distributed training capabilities. The script validates that PEFT's imports don't interfere with CUDA initialization in multiprocessing contexts, which is essential for distributed training workflows.
