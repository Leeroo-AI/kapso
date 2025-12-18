{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::PEFT]], [[domain::Deprecation]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==
Deprecation warning for Bone (Block Affine) adapter - will be removed in PEFT v0.19.0, migrate to MiSS.

=== Description ===
The Bone (Block Affine) adapter implementation is deprecated and scheduled for removal in PEFT version 0.19.0. The Bone architecture has been superseded by MiSS (Mixed-rank Sparse Subspace), which provides equivalent functionality with improved performance characteristics.

Bone uses block-affine transformations based on Householder reflections. MiSS is the evolution of this approach with better rank handling and sparsity patterns.

=== Usage ===
Apply this heuristic when you encounter:
* Existing codebases using `BoneConfig` or `BoneModel`
* New projects considering Bone as an adapter choice
* Migration planning for PEFT version upgrades

== The Insight (Rule of Thumb) ==
* **Action:** Replace `BoneConfig` with `MissConfig` in new code
* **Migration:** Use the provided script `/scripts/convert-bone-to-miss.py` to convert existing Bone checkpoints to MiSS format
* **Timeline:** Bone will be removed in PEFT v0.19.0
* **Alternative:** `MissConfig` provides equivalent functionality with the same hyperparameters

== Reasoning ==
MiSS (Mixed-rank Sparse Subspace) is a generalization of Bone that:
1. Supports mixed rank configurations across different layers
2. Provides better memory efficiency
3. Has improved compatibility with quantization backends
4. Is actively maintained while Bone is in deprecation

The conversion script handles:
* Weight tensor format conversion
* Config parameter mapping
* Checkpoint metadata updates

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_BoneConfig]]
* [[uses_heuristic::Implementation:huggingface_peft_BoneModel]]
* [[uses_heuristic::Implementation:huggingface_peft_BoneLayer]]
* [[uses_heuristic::Implementation:huggingface_peft_MissConfig]]
