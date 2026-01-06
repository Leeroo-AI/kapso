{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Decoding]], [[domain::Search_Algorithms]], [[domain::Generation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Beam search decoding data structures and utilities for generating multiple sequence candidates during text generation.

=== Description ===
The beam_search.py module is a compact 88-line implementation of data structures and scoring functions for beam search decoding in vLLM. Beam search is a heuristic search algorithm that explores multiple generation paths simultaneously by keeping the top-k most promising candidates at each step, where k is the beam width.

The module defines three key components: (1) BeamSearchSequence - a dataclass representing a single candidate sequence in the beam, containing tokens, per-token logprobs, cumulative log probability, optional text, finish reason, stop reason, LoRA request, and multimodal data; (2) BeamSearchOutput - a wrapper containing the final list of best beam search sequences; (3) BeamSearchInstance - manages active beams and completed sequences during generation, initialized with prompt tokens and maintaining separate lists for in-progress beams and finished sequences.

The get_beam_search_score function implements length-normalized scoring following the formula from Hugging Face transformers: score = cumulative_logprob / (sequence_length ** length_penalty). This prevents bias toward shorter sequences. The create_sort_beams_key_function factory creates comparison functions for ranking beams. The module integrates with vLLM's sampling and scheduling systems to enable beam search as an alternative to standard sampling-based generation.

=== Usage ===
Used internally by vLLM's beam search scheduler and output processors. Not typically instantiated directly by users.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/beam_search.py vllm/beam_search.py]
* '''Lines:''' 1-88

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class BeamSearchSequence:
    tokens: list[int]  # Token IDs including prompt
    logprobs: list[dict[int, Logprob]]  # Per-token log probabilities
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: dict[str, Any] | None = None

@dataclass
class BeamSearchOutput:
    sequences: list[BeamSearchSequence]

class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    )
    beams: list[BeamSearchSequence]
    completed: list[BeamSearchSequence]

def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float

def create_sort_beams_key_function(
    eos_token_id: int,
    length_penalty: float
) -> Callable[[BeamSearchSequence], float]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.beam_search import (
    BeamSearchSequence,
    BeamSearchOutput,
    BeamSearchInstance,
    get_beam_search_score,
    create_sort_beams_key_function,
)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BeamSearchSequence || Dataclass || Single candidate sequence in beam
|-
| BeamSearchOutput || Dataclass || Final beam search results
|-
| BeamSearchInstance || Class || Manages beams during generation
|-
| get_beam_search_score || Function || Compute length-normalized beam score
|-
| create_sort_beams_key_function || Function || Create beam sorting key function
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from vllm.beam_search import (
    BeamSearchSequence,
    BeamSearchInstance,
    get_beam_search_score,
    create_sort_beams_key_function,
)

# Example 1: Creating beam search instance
prompt_tokens = [1, 2, 3, 4, 5]  # Tokenized prompt
beam_instance = BeamSearchInstance(
    prompt_tokens=prompt_tokens,
    lora_request=None,
)

print(f"Initial beams: {len(beam_instance.beams)}")  # 1
print(f"Completed: {len(beam_instance.completed)}")  # 0

# Example 2: Creating and scoring sequences
seq1 = BeamSearchSequence(
    tokens=[1, 2, 3, 4, 5, 10, 20, 30],
    logprobs=[],
    cum_logprob=-5.2,
)

seq2 = BeamSearchSequence(
    tokens=[1, 2, 3, 4, 5, 11, 21, 31, 41],
    logprobs=[],
    cum_logprob=-5.5,
)

eos_token_id = 2
length_penalty = 1.0

score1 = get_beam_search_score(
    seq1.tokens, seq1.cum_logprob, eos_token_id, length_penalty
)
score2 = get_beam_search_score(
    seq2.tokens, seq2.cum_logprob, eos_token_id, length_penalty
)

print(f"Seq1 score: {score1}")  # -5.2 / 8 = -0.65
print(f"Seq2 score: {score2}")  # -5.5 / 9 = -0.611

# Example 3: Sorting beams
beams = [seq1, seq2]
sort_key = create_sort_beams_key_function(eos_token_id, length_penalty)

# Sort in descending order (best scores first)
sorted_beams = sorted(beams, key=sort_key, reverse=True)
print(f"Best beam: {sorted_beams[0].tokens}")

# Example 4: Length penalty effect
# Without penalty (length_penalty = 1.0)
score_no_penalty = get_beam_search_score(
    tokens=[1, 2, 3, 4, 5, 10, 20, 30],
    cumulative_logprob=-8.0,
    eos_token_id=2,
    length_penalty=1.0,
)
print(f"No penalty: {score_no_penalty}")  # -8.0 / 8 = -1.0

# With penalty favoring longer sequences (length_penalty = 0.8)
score_with_penalty = get_beam_search_score(
    tokens=[1, 2, 3, 4, 5, 10, 20, 30],
    cumulative_logprob=-8.0,
    eos_token_id=2,
    length_penalty=0.8,
)
print(f"With penalty: {score_with_penalty}")  # -8.0 / (8^0.8) = -1.246

# Example 5: Managing beams during generation
beam_width = 3
beam_instance = BeamSearchInstance(prompt_tokens=[1, 2, 3])

# Simulate beam expansion
# At step 1, generate beam_width new candidates
new_beams = [
    BeamSearchSequence(
        tokens=[1, 2, 3, 10],
        logprobs=[],
        cum_logprob=-1.2,
    ),
    BeamSearchSequence(
        tokens=[1, 2, 3, 11],
        logprobs=[],
        cum_logprob=-1.5,
    ),
    BeamSearchSequence(
        tokens=[1, 2, 3, 12],
        logprobs=[],
        cum_logprob=-1.8,
    ),
]

beam_instance.beams = new_beams

# At step 2, some beams hit EOS
beam_instance.completed.append(new_beams[0])
beam_instance.beams = new_beams[1:]

print(f"Active beams: {len(beam_instance.beams)}")
print(f"Completed: {len(beam_instance.completed)}")

# Example 6: Using with vLLM API
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B")

# Configure beam search via SamplingParams
sampling_params = SamplingParams(
    n=3,  # Return 3 best sequences
    use_beam_search=True,
    best_of=5,  # Beam width
    temperature=0.0,  # Greedy within beam
    length_penalty=1.0,
)

outputs = llm.generate(
    "Once upon a time,",
    sampling_params=sampling_params
)

# Access beam search results
for output in outputs:
    for seq in output.outputs:
        print(f"Text: {seq.text}")
        print(f"Score: {seq.cumulative_logprob / len(seq.token_ids)}")
</syntaxhighlight>

== Related Pages ==
* [[implements::Algorithm:Beam_Search]]
* [[used_by::Module:vllm-project_vllm_Sequence_Scheduler]]
* [[alternative_to::Module:vllm-project_vllm_Sampling]]
* [[related::Module:vllm-project_vllm_Logprobs]]
