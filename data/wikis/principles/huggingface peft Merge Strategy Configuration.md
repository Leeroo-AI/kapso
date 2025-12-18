{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|TIES|https://arxiv.org/abs/2306.01708]]
* [[source::Paper|DARE|https://arxiv.org/abs/2311.03099]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Multi_Task]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for selecting and configuring adapter merging algorithms, including strategy selection (TIES, DARE, linear) and parameter tuning.

=== Description ===

Merge Strategy Configuration involves selecting the appropriate merging algorithm and its hyperparameters. Different algorithms suit different scenarios:

* **Linear**: Simple weighted average, fast but may average out conflicts
* **TIES**: Trims, elects sign, merges - good for adapters with conflicting updates
* **DARE**: Random pruning with rescaling - robust to parameter redundancy
* **SVD**: Reduces output rank for memory efficiency

=== Usage ===

Apply this principle before executing adapter merge:
* Choose algorithm based on adapter characteristics
* Set weights reflecting adapter importance
* Tune density for TIES/DARE (0.5-0.9 typical)

== Theoretical Basis ==

'''Algorithm Selection Guide:'''

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Similar tasks, no conflicts | Linear |
| Diverse tasks, potential sign conflicts | TIES |
| Need robustness to specific weights | DARE Linear |
| Diverse + robust | DARE TIES |
| Memory constrained output | SVD variants |

'''Key Parameters:'''

* `weights`: Per-adapter importance (sum to 1.0 typical)
* `density`: Fraction of weights to keep (0.0-1.0)
* `majority_sign_method`: "total" (sum) or "frequency" (count)
* `svd_rank`: Target rank for SVD methods

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_merge_strategy_selection]]
