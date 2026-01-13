# Environment: Python_3

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|micrograd|https://github.com/karpathy/micrograd]]
* [[source::Doc|setup.py|https://github.com/karpathy/micrograd/blob/master/setup.py]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Educational]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Minimal Python 3.6+ environment with zero external dependencies for core autograd functionality.

=== Description ===

Micrograd is designed as a zero-dependency autograd engine for educational purposes. The core library (`micrograd/engine.py` and `micrograd/nn.py`) requires only Python 3.6+ and uses exclusively the Python standard library (specifically the `random` module for weight initialization). This makes it extremely portable and easy to understand without external framework complexity.

Optional dependencies are required only for testing (PyTorch) and running the demo notebook (numpy, matplotlib, scikit-learn).

=== Usage ===

Use this environment for any work with the micrograd library, including building and training neural networks using the `Value` autograd engine and the `nn` module (Neuron, Layer, MLP classes). The zero-dependency nature makes this ideal for learning environments where package management may be limited.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Any || Cross-platform (Linux, macOS, Windows)
|-
| Hardware || CPU only || No GPU required; scalar-based computation
|-
| Disk || < 1MB || Minimal footprint
|}

== Dependencies ==

=== System Packages ===

* None required for core functionality

=== Python Packages ===

**Core (Required):**
* Python >= 3.6 (uses f-strings, type hints not required)
* `random` (standard library)

**Testing (Optional):**
* `torch` - PyTorch for gradient validation tests
* `pytest` - Test runner

**Demo Notebook (Optional):**
* `numpy` - Array operations for dataset handling
* `matplotlib` - Visualization
* `scikit-learn` - Dataset generation (make_moons)

== Credentials ==

No credentials or environment variables required.

== Quick Install ==

<syntaxhighlight lang="bash">
# Core installation (no dependencies needed)
pip install micrograd

# Or from source
git clone https://github.com/karpathy/micrograd.git
cd micrograd
pip install -e .

# Optional: For running tests
pip install torch pytest

# Optional: For running demo notebook
pip install numpy matplotlib scikit-learn
</syntaxhighlight>

== Code Evidence ==

Python version requirement from `setup.py:21`:
<syntaxhighlight lang="python">
python_requires='>=3.6',
</syntaxhighlight>

Standard library imports only from `nn.py:1`:
<syntaxhighlight lang="python">
import random
from micrograd.engine import Value
</syntaxhighlight>

Engine module has no imports at all from `engine.py` (entire file uses only built-in Python):
<syntaxhighlight lang="python">
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # ...
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `SyntaxError: invalid syntax` (f-strings) || Python < 3.6 || Upgrade to Python 3.6+
|-
|| `ModuleNotFoundError: No module named 'torch'` || Missing PyTorch || `pip install torch` (only needed for tests)
|-
|| `ModuleNotFoundError: No module named 'sklearn'` || Missing scikit-learn || `pip install scikit-learn` (only for demo notebook)
|}

== Compatibility Notes ==

* **Pure Python:** Works on any platform that supports Python 3.6+
* **No GPU:** The library operates entirely on scalar values using CPU; no CUDA/GPU support
* **Educational Focus:** Designed for understanding autograd mechanics, not production use
* **PyTorch Comparison:** Tests validate gradient computation against PyTorch to ensure correctness

== Related Pages ==

* [[required_by::Implementation:Karpathy_Micrograd_Data_Preparation_Pattern]]
* [[required_by::Implementation:Karpathy_Micrograd_MLP_Init]]
* [[required_by::Implementation:Karpathy_Micrograd_MLP_Call]]
* [[required_by::Implementation:Karpathy_Micrograd_Loss_Operations]]
* [[required_by::Implementation:Karpathy_Micrograd_Value_Backward]]
* [[required_by::Implementation:Karpathy_Micrograd_Module_Parameters]]
* [[required_by::Implementation:Karpathy_Micrograd_Module_Zero_Grad]]
