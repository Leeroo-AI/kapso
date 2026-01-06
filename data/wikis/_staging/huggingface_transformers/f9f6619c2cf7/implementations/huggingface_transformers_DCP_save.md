{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
|-
! Domains
| [[domain::Distributed_Computing]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Usage pattern for torch.distributed.checkpoint (DCP) save/load within HuggingFace Transformers distributed training.

=== Description ===
Transformers uses PyTorch's Distributed Checkpoint (DCP) API to save and load model and optimizer states in 3D parallelism setups. The implementation in 3d_parallel_checks.py demonstrates creating a Stateful wrapper (AppState) that encapsulates both model and optimizer, then using dcp.save() and dcp.load() for concurrent checkpointing across all ranks. Each rank saves only its local parameter and optimizer state shards, dramatically reducing memory requirements and I/O time compared to centralized checkpointing.

The Stateful protocol requires implementing state_dict() and load_state_dict() methods, using get_state_dict() and set_state_dict() helpers to properly extract and apply distributed state with DTensor metadata.

=== Usage ===
Create a Stateful wrapper for your model and optimizer, then use dcp.save() to checkpoint and dcp.load() to restore. All ranks call these functions simultaneously, with the checkpoint_id specifying the directory path.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.distributed.checkpoint)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L40-41, L450-482, L613-627

=== Signature ===
<syntaxhighlight lang="python">
# Save
dcp.save(
    state_dict: dict[str, Any],
    checkpoint_id: str | os.PathLike,
    storage_writer: StorageWriter = None,
    planner: SavePlanner = None,
    process_group: ProcessGroup = None
)

# Load
dcp.load(
    state_dict: dict[str, Any],
    checkpoint_id: str | os.PathLike,
    storage_reader: StorageReader = None,
    planner: LoadPlanner = None,
    process_group: ProcessGroup = None
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| state_dict || dict || Yes || Dictionary containing Stateful objects or tensors to save/load
|-
| checkpoint_id || str or PathLike || Yes || Directory path for checkpoint files
|-
| storage_writer || StorageWriter || No || Custom storage backend (default: FileSystemWriter)
|-
| storage_reader || StorageReader || No || Custom storage backend for loading
|-
| planner || SavePlanner/LoadPlanner || No || Custom planning strategy for sharding
|-
| process_group || ProcessGroup || No || Process group for coordination (default: global)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Checkpoint saved to/loaded from filesystem, state_dict modified in-place for load
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForCausalLM
import torch.optim as optim

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()

# Create 3D device mesh
tp_size = 2
dp_size = 4
cp_size = 1
world_size = tp_size * dp_size * cp_size

mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(
    device_type="cuda",
    mesh=mesh,
    mesh_dim_names=("dp", "tp", "cp")
)
tp_mesh = world_mesh["tp"]

# Load model and optimizer
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Define Stateful wrapper (from 3d_parallel_checks.py)
class AppState(Stateful):
    """Wrapper for checkpointing the Application State including model and optimizer."""

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # Extract distributed state with DTensor metadata
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # Apply loaded state to model and optimizer
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

# Training for some steps...
for step in range(100):
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save checkpoint with DCP
CHECKPOINT_DIR = f"checkpoint_tp{tp_size}_dp{dp_size}_cp{cp_size}"

if dist.is_initialized():
    # Create state dictionary
    state_dict = {"app": AppState(model, optimizer)}

    # All ranks call save simultaneously
    dcp.save(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR
    )

    if rank == 0:
        print(f"Checkpoint saved to {CHECKPOINT_DIR}")

# Load checkpoint with DCP
if dist.is_initialized():
    # Create new model and optimizer instances
    new_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B",
        device_mesh=tp_mesh,
        tp_plan="auto",
        dtype=torch.bfloat16
    )
    new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-5)

    # Create state dictionary with new instances
    state_dict = {"app": AppState(new_model, new_optimizer)}

    # All ranks call load simultaneously
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR
    )

    if rank == 0:
        print(f"Checkpoint loaded from {CHECKPOINT_DIR}")

    # Verify loaded weights match original
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(),
        new_model.named_parameters()
    ):
        torch.testing.assert_close(
            param1.to_local() if hasattr(param1, 'to_local') else param1,
            param2.to_local() if hasattr(param2, 'to_local') else param2,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Parameter mismatch: {name1}"
        )

    print(f"Rank {rank}: Checkpoint verification successful")

# Alternative: Direct state_dict usage without Stateful wrapper
model_state_dict, optim_state_dict = get_state_dict(model, optimizer)

state_to_save = {
    "model": model_state_dict,
    "optim": optim_state_dict,
    "epoch": current_epoch,
    "step": current_step
}

dcp.save(state_dict=state_to_save, checkpoint_id=CHECKPOINT_DIR)

# Load back
state_to_load = {
    "model": {},
    "optim": {},
    "epoch": 0,
    "step": 0
}

dcp.load(state_dict=state_to_load, checkpoint_id=CHECKPOINT_DIR)

set_state_dict(
    model, optimizer,
    model_state_dict=state_to_load["model"],
    optim_state_dict=state_to_load["optim"]
)

resumed_epoch = state_to_load["epoch"]
resumed_step = state_to_load["step"]

# Checkpoint directory structure
# checkpoint_tp2_dp4_cp1/
#   ├── __metadata__           # Coordination metadata (rank 0)
#   ├── __0_0.pt              # Rank 0's shards
#   ├── __1_0.pt              # Rank 1's shards
#   ├── __2_0.pt              # Rank 2's shards
#   └── ...                   # Other ranks' shards

# Loading with different parallelism configuration
# Save: TP=2, DP=4 → Load: TP=4, DP=2
# DCP automatically reshards during load!

# Custom storage writer (example)
from torch.distributed.checkpoint import FileSystemWriter

writer = FileSystemWriter(CHECKPOINT_DIR)
dcp.save(
    state_dict=state_dict,
    checkpoint_id=CHECKPOINT_DIR,
    storage_writer=writer
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Distributed_Checkpointing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
