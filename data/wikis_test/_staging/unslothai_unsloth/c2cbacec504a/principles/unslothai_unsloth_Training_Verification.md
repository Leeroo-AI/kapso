# Principle: unslothai_unsloth_Training_Verification

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Evaluation]], [[domain::Quality_Assurance]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for validating trained model quality through inference testing before export.

=== Description ===

Training Verification ensures the model produces expected outputs by:
1. **Generating test responses** on held-out prompts
2. **Comparing to baselines** or expected outputs
3. **Checking for degradation** (repetition, gibberish, truncation)

This is a critical quality gate before investing time in export.

=== Usage ===

Always verify model quality before export to catch training issues early.

== Practical Guide ==

=== Verification Checklist ===

<syntaxhighlight lang="python">
def verify_model_quality(model, tokenizer, test_prompts):
    FastLanguageModel.for_inference(model)

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check for issues
        assert len(response) > len(prompt), "Empty response"
        assert response.count(response[-20:]) < 3, "Repetition detected"
        print(f"✓ {prompt[:30]}... → {response[-50:]}")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_model_generate]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
