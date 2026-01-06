{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Token Probabilities]], [[domain::Output Analysis]], [[domain::Inference]], [[domain::OpenAI Compatibility]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Logprobs provides efficient data structures for tracking and storing token probabilities and vocabulary ranks during LLM generation, with minimal garbage collection overhead.

=== Description ===
The logprobs module implements memory-efficient data structures for storing token probability information across generation sequences. Key features include:

* '''FlatLogprobs:''' Flattened storage using primitive type lists to reduce GC overhead
* '''Logprob dataclass:''' Stores individual token logprob, rank, and decoded text
* '''Backwards compatibility:''' Acts as list[dict[int, Logprob]] while being more efficient
* '''Constant object count:''' Regardless of sequence length or top_logprobs setting
* '''OpenAI API compatible:''' Supports standard logprobs output format

Instead of creating nested dictionaries for every position and rank, FlatLogprobs flattens all information into parallel arrays with index ranges, significantly reducing memory allocations and GC pressure during long generations.

=== Usage ===
Use this module when you need to:
* Track token probabilities during generation
* Implement OpenAI-compatible logprobs output
* Minimize memory overhead for long sequences
* Store vocabulary ranks for top-k tokens
* Decode token IDs to strings for analysis

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/logprobs.py vllm/logprobs.py]

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class Logprob:
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None

@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition]):
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]
    ranks: list[int | None]
    decoded_tokens: list[str | None]

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None
    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None]
    ) -> None
    def __getitem__(self, index: int | slice) -> LogprobsOnePosition | FlatLogprobs
    def __iter__(self) -> Iterator[LogprobsOnePosition]

# Type aliases
LogprobsOnePosition = dict[int, Logprob]
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.logprobs import (
    Logprob,
    FlatLogprobs,
    LogprobsOnePosition,
    PromptLogprobs,
    SampleLogprobs,
    create_prompt_logprobs,
    create_sample_logprobs,
    append_logprobs_for_next_position,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| logprobs_one_position || dict[int, Logprob] || Logprobs for a single generation position
|-
| token_ids || list[int] || Token IDs for the position
|-
| logprobs || list[float] || Log probabilities for each token
|-
| ranks || list[int] || Vocabulary ranks for each token (1-indexed)
|-
| decoded_tokens || list[str &#124; None] || Decoded string for each token
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| FlatLogprobs || FlatLogprobs || Efficiently stored logprobs across all positions
|-
| LogprobsOnePosition || dict[int, Logprob] || Logprobs for a specific position (via indexing)
|}

== Usage Examples ==

=== Create and Populate Logprobs ===
<syntaxhighlight lang="python">
from vllm.logprobs import FlatLogprobs, Logprob

# Create empty logprobs container
logprobs = FlatLogprobs()

# Add logprobs for first position
position_0 = {
    1234: Logprob(logprob=-0.5, rank=1, decoded_token="Hello"),
    5678: Logprob(logprob=-2.1, rank=2, decoded_token="Hi"),
}
logprobs.append(position_0)

# Add logprobs for second position
position_1 = {
    9012: Logprob(logprob=-0.3, rank=1, decoded_token=" world"),
    3456: Logprob(logprob=-1.8, rank=2, decoded_token=" there"),
}
logprobs.append(position_1)

# Access by position
first_position = logprobs[0]  # Returns dict[int, Logprob]
print(first_position[1234].decoded_token)  # "Hello"

# Iterate over all positions
for pos_logprobs in logprobs:
    for token_id, lp in pos_logprobs.items():
        print(f"Token {token_id}: {lp.decoded_token} (logprob={lp.logprob}, rank={lp.rank})")
</syntaxhighlight>

=== Fast Append Without Intermediate Dictionary ===
<syntaxhighlight lang="python">
import itertools
from vllm.logprobs import FlatLogprobs

logprobs = FlatLogprobs()

# Prepare data for a position
token_ids = [100, 200, 300]
logprob_values = [-0.5, -1.2, -2.3]
ranks = itertools.chain((1,), [2, 3, 4])  # Sampled token rank first
decoded = ["the", "a", "an"]

# Fast append avoids creating intermediate dict
logprobs.append_fast(token_ids, logprob_values, ranks, decoded)

# Result is same as using append() but more efficient
print(len(logprobs))  # 1 position
print(logprobs[0][100])  # Logprob(logprob=-0.5, rank=1, decoded_token="the")
</syntaxhighlight>

=== Create Prompt vs Sample Logprobs ===
<syntaxhighlight lang="python">
from vllm.logprobs import create_prompt_logprobs, create_sample_logprobs

# Prompt logprobs (first token is None)
prompt_logprobs = create_prompt_logprobs(flat_logprobs=True)
print(len(prompt_logprobs))  # 1 (first position is None)

# Sample logprobs (empty initially)
sample_logprobs = create_sample_logprobs(flat_logprobs=True)
print(len(sample_logprobs))  # 0

# Can also create non-flat versions
prompt_logprobs_list = create_prompt_logprobs(flat_logprobs=False)
# Returns: [None]

sample_logprobs_list = create_sample_logprobs(flat_logprobs=False)
# Returns: []
</syntaxhighlight>

=== Append Logprobs for Next Position ===
<syntaxhighlight lang="python">
from vllm.logprobs import (
    create_sample_logprobs,
    append_logprobs_for_next_position
)

logprobs = create_sample_logprobs(flat_logprobs=True)

# Sampled token
token_ids = [42]
logprob_values = [-0.1]
decoded_tokens = ["Hello"]
sampled_rank = 1
num_logprobs = 5

# This will add the sampled token plus top-k alternatives
append_logprobs_for_next_position(
    request_logprobs=logprobs,
    token_ids=token_ids,
    logprobs=logprob_values,
    decoded_tokens=decoded_tokens,
    rank=sampled_rank,
    num_logprobs=num_logprobs
)
</syntaxhighlight>

=== Slice Logprobs ===
<syntaxhighlight lang="python">
from vllm.logprobs import FlatLogprobs

# Create logprobs with multiple positions
logprobs = FlatLogprobs()
for i in range(10):
    logprobs.append({i: Logprob(logprob=-i, rank=1, decoded_token=f"token_{i}")})

# Slice to get subset
subset = logprobs[2:5]
print(len(subset))  # 3 positions
print(type(subset))  # FlatLogprobs

# The slice maintains efficient flat structure
for pos_logprobs in subset:
    print(pos_logprobs)
</syntaxhighlight>

=== Access Internal Flat Arrays ===
<syntaxhighlight lang="python">
from vllm.logprobs import FlatLogprobs

logprobs = FlatLogprobs()
logprobs.append({10: Logprob(-0.5, 1, "a"), 20: Logprob(-1.0, 2, "b")})
logprobs.append({30: Logprob(-0.3, 1, "c")})

# Inspect flat storage
print(logprobs.token_ids)        # [10, 20, 30]
print(logprobs.logprobs)         # [-0.5, -1.0, -0.3]
print(logprobs.ranks)            # [1, 2, 1]
print(logprobs.decoded_tokens)   # ["a", "b", "c"]
print(logprobs.start_indices)    # [0, 2]
print(logprobs.end_indices)      # [2, 3]

# Position 0 spans indices [0:2] (tokens 10, 20)
# Position 1 spans indices [2:3] (token 30)
</syntaxhighlight>

== Memory Efficiency ==

=== Comparison ===
{| class="wikitable"
|-
! Approach !! Objects Created !! GC Pressure
|-
| list[dict[int, Logprob]] || O(seq_len * top_k) || High
|-
| FlatLogprobs || O(1) || Minimal
|}

For a sequence with 1000 tokens and top_logprobs=5:
* Traditional: ~5000 dict + 5000 Logprob objects = 10,000 objects
* FlatLogprobs: 1 FlatLogprobs object + 6 list objects = ~7 objects

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[SamplingParams]]
* [[RequestOutput]]
* [[OpenAI API Compatibility]]
* [[Token Decoding]]
