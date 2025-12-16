# Principle: Synthetic Data Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Self-Instruct: Aligning LLMs with Self-Generated Instructions|https://arxiv.org/abs/2212.10560]]
* [[source::Paper|Textbooks Are All You Need|https://arxiv.org/abs/2306.11644]]
* [[source::Blog|Data Generation with LLMs|https://huggingface.co/blog/synthetic-data-generation]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::NLP]], [[domain::Training_Data]], [[domain::Domain_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
A data augmentation methodology that uses large language models to generate training examples from source documents, enabling domain adaptation when labeled data is scarce.

=== Description ===
Synthetic Data Generation leverages the knowledge encoded in pre-trained LLMs to create high-quality training data from unlabeled source documents. This approach is particularly valuable for domain-specific fine-tuning where obtaining human-labeled data is expensive or impractical.

The methodology follows a pipeline:

1. **Document Ingestion** - Convert source materials (PDFs, web pages, documents) to text
2. **Chunking** - Split documents into model-appropriate segments with overlap
3. **QA Generation** - Use an LLM to generate question-answer pairs from each chunk
4. **Quality Filtering** - Apply heuristics or model-based filtering to remove low-quality examples
5. **Format Conversion** - Transform into instruction-tuning format (e.g., Alpaca, ShareGPT)

**Problem Solved:**
Training instruction-following models requires large volumes of (instruction, response) pairs. Traditional approaches rely on human annotation which is:
- Expensive at scale
- Slow to produce
- Domain-limited by annotator expertise

Synthetic generation addresses these by using a capable LLM as an "infinite annotator" that can process any domain given appropriate source material.

**Quality Considerations:**
- Generated data inherits the biases and limitations of the source LLM
- Diversity must be actively managed to avoid mode collapse
- Domain-specific validation is essential for high-stakes applications

=== Usage ===
Use synthetic data generation when:
- Fine-tuning for a specialized domain (legal, medical, technical)
- Labeled data is scarce or expensive to obtain
- You have access to large volumes of unlabeled domain text
- The base LLM has reasonable capability in the target domain

Best Practices:
- Use a stronger model for generation than your fine-tuning target
- Implement diversity controls (temperature, sampling strategies)
- Validate generated data with domain experts on a subset
- Combine synthetic with human-labeled data when available

Do NOT rely solely on synthetic data when:
- Safety-critical applications require verified accuracy
- Domain requires factual precision (medical diagnosis, legal advice)
- Source documents may contain errors or biases

== Theoretical Basis ==
'''Self-Instruct Methodology:'''
<syntaxhighlight lang="python">
# Core generation loop
def generate_qa_pairs(document_chunk, llm, num_pairs=25):
    """
    Generate instruction-response pairs from a document chunk.

    Prompt template guides the LLM to:
    1. Identify key concepts in the chunk
    2. Generate diverse questions (factual, analytical, creative)
    3. Produce accurate answers grounded in the source
    """
    prompt = f"""
    Given the following text, generate {num_pairs} diverse question-answer pairs.

    Text:
    {document_chunk}

    For each pair:
    - Question should be answerable from the text
    - Answer should be accurate and grounded
    - Vary question types (what, how, why, compare, explain)

    Output as JSON: [{{"question": ..., "answer": ...}}, ...]
    """
    return llm.generate(prompt, temperature=0.7)
</syntaxhighlight>

'''Chunking Strategy:'''
<math>
\text{chunk\_size} = \text{max\_seq\_len} - 2 \times \text{generation\_tokens} - \text{buffer}
</math>

Overlap between chunks ensures context continuity:
<syntaxhighlight lang="python">
def chunk_with_overlap(text, max_tokens, overlap_tokens=64):
    """
    Split text into chunks with token-level overlap.

    The overlap ensures:
    - Questions spanning chunk boundaries can be answered
    - Context is preserved for entity/concept references
    """
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokens[start:end])
        start = end - overlap_tokens
    return chunks
</syntaxhighlight>

'''Data Quality Metrics:'''
- **Groundedness**: Can the answer be verified from the source?
- **Diversity**: Are questions varied in type and complexity?
- **Difficulty**: Mix of simple and complex reasoning required
- **Coherence**: Is the Q-A pair self-contained and clear?

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_SyntheticDataKit]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Management]]
