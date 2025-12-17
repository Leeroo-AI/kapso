{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Checkpoint discovery is the process of locating and cataloging all weight files comprising a distributed model checkpoint before loading them into memory.

=== Description ===

Modern large language models often exceed single-file size limits or storage constraints, necessitating split storage across multiple checkpoint files (shards). Checkpoint discovery solves the problem of transparently handling both monolithic and sharded checkpoints through a consistent interface.

The discovery process uses an index file that acts as a manifest, mapping individual model parameters to their physical storage locations. This indirection provides several benefits:

* '''Lazy Loading:''' Load only necessary shards for specific layers or operations
* '''Parallel Downloads:''' Fetch multiple shards concurrently from remote storage
* '''Memory Efficiency:''' Load and process one shard at a time rather than all weights simultaneously
* '''Fault Tolerance:''' Retry individual shard downloads without restarting entire process
* '''Storage Flexibility:''' Rearrange parameter-to-shard mapping without changing model code

The index file typically contains a weight_map (parameter_name -> shard_filename) and metadata (total parameters, shard sizes). Discovery systems must handle both local filesystems and remote object storage, managing caching to avoid redundant downloads.

=== Usage ===

Use checkpoint discovery when:
* Implementing model loading for architectures that may be sharded or monolithic
* Building distributed inference systems that load model partitions across machines
* Creating memory-efficient loading pipelines for resource-constrained environments
* Developing model conversion tools that need to process checkpoints shard-by-shard
* Designing fault-tolerant download systems for large models from cloud storage

== Theoretical Basis ==

Checkpoint discovery implements a two-phase loading protocol:

'''Phase 1: Index Resolution'''
<pre>
INPUT: model_identifier, storage_backend
OUTPUT: index_structure

IF is_sharded_checkpoint(model_identifier) THEN
    index_file = locate_file(model_identifier, "*.index.json")
    index_structure = parse_json(index_file)

    VERIFY index_structure contains:
        - weight_map: dict[param_name, shard_filename]
        - metadata: dict containing checkpoint statistics
ELSE
    # Monolithic checkpoint - create virtual index
    checkpoint_file = locate_file(model_identifier, "*.safetensors" | "*.bin")
    index_structure = create_single_shard_index(checkpoint_file)
END IF

RETURN index_structure
</pre>

'''Phase 2: Shard Enumeration'''
<pre>
INPUT: index_structure, storage_backend
OUTPUT: shard_locations[]

shard_filenames = extract_unique_values(index_structure.weight_map)
shard_locations = []

FOR EACH shard_filename IN shard_filenames:
    IF storage_backend == "local" THEN
        location = resolve_local_path(shard_filename)
    ELSE IF storage_backend == "remote" THEN
        location = download_and_cache(shard_filename)
        # Caching layer handles deduplication
    END IF

    VERIFY file_exists(location) AND is_readable(location)
    shard_locations.append(location)
END FOR

RETURN shard_locations, index_structure.metadata
</pre>

'''Optimization Considerations:'''
* '''Concurrent Downloads:''' Fetch N shards in parallel (typically N=4-8 for I/O-bound workloads)
* '''Cache Coherence:''' Use content-addressable storage (commit hashes) to invalidate stale caches
* '''Partial Loading:''' Allow filtering weight_map to discover only required shards for specific layers
* '''Bandwidth Management:''' Prioritize shards containing frequently-accessed parameters (embeddings, first layers)

'''Invariants:'''
* All parameters in weight_map must map to exactly one shard
* Shard files must be immutable after creation (identified by content hash)
* Discovery must succeed even if some shards are temporarily unavailable (for fault-tolerant systems)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_get_checkpoint_shard_files]]
