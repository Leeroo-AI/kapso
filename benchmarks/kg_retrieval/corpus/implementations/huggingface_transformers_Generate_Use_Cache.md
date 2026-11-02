# Implementation: Generate_Use_Cache

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Text generation|https://huggingface.co/docs/transformers/main_classes/text_generation]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

The use_cache argument of generate in transformers controls whether past attention keys and values are kept between decoding steps; model generation settings enable it by default.

=== Description ===

With the flag on, each step feeds only the newest token and the returned past_key_values; with it off, the whole sequence is processed again every step, which is much slower but needs less memory.

=== Usage ===

Leave it on for normal generation; turn it off only for debugging or when memory is more constrained than time.

== Code Reference ==

<syntaxhighlight lang="python">
model.generate(input_ids, max_new_tokens=128, use_cache=True)
</syntaxhighlight>

== I/O Contract ==

Returns the generated token ids. When called with return_dict_in_generate=True, the output also exposes past_key_values.
