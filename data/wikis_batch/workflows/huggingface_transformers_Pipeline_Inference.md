{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Pipelines|https://huggingface.co/docs/transformers/main_classes/pipelines]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::NLP]], [[domain::Computer_Vision]], [[domain::Audio]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for running inference on transformer models using the high-level pipeline() API that abstracts preprocessing, model execution, and postprocessing.

=== Description ===
This workflow covers the complete inference pipeline in the Transformers library using the `pipeline()` function. It handles:

1. **Task Resolution**: Identifying the task type and selecting appropriate models
2. **Component Loading**: Loading model, tokenizer, and feature extractors
3. **Input Preprocessing**: Converting raw inputs to model-ready tensors
4. **Model Inference**: Running the forward pass with proper device handling
5. **Output Postprocessing**: Converting model outputs to human-readable results

The pipeline API supports 30+ task types across text, vision, audio, and multimodal domains with automatic batching and device placement.

=== Usage ===
Execute this workflow when you need to:
* Quickly run inference without manual preprocessing/postprocessing
* Use models for standard NLP tasks (classification, NER, QA, generation)
* Process images (classification, segmentation, object detection)
* Transcribe audio or generate speech
* Build prototypes or production inference endpoints

== Execution Steps ==

=== Step 1: Task Resolution ===
[[step::Principle:huggingface_transformers_Task_Resolution]]

Determine the task type from the task string or model's pipeline_tag. The task defines which pipeline class to use and what preprocessing/postprocessing steps are required. Task aliases map common names to canonical task identifiers.

'''Supported task categories:'''
* Text: text-generation, text-classification, token-classification, fill-mask, question-answering
* Vision: image-classification, object-detection, image-segmentation, depth-estimation
* Audio: automatic-speech-recognition, audio-classification, text-to-audio
* Multimodal: image-to-text, visual-question-answering, document-question-answering
* Zero-shot: zero-shot-classification, zero-shot-image-classification

=== Step 2: Component Loading ===
[[step::Principle:huggingface_transformers_Pipeline_Component_Loading]]

Load all required components for the pipeline: the model, tokenizer (for text), image processor (for vision), or feature extractor (for audio). Components are loaded from the same checkpoint or can be specified separately.

'''Components by task type:'''
* Text tasks: Model + Tokenizer
* Vision tasks: Model + Image Processor
* Audio tasks: Model + Feature Extractor
* Multimodal tasks: Model + Processor (combines multiple)

=== Step 3: Pipeline Instantiation ===
[[step::Principle:huggingface_transformers_Pipeline_Instantiation]]

Create the pipeline instance with all components and configuration. The pipeline class is selected based on the task type. Device placement and batch size are configured at this stage.

'''Configuration options:'''
* Device placement (CPU, CUDA, MPS)
* Batch size for throughput optimization
* Framework selection (PyTorch required)
* Custom preprocessing/postprocessing

=== Step 4: Input Preprocessing ===
[[step::Principle:huggingface_transformers_Pipeline_Preprocessing]]

Transform raw inputs into model-ready tensors. Preprocessing varies by task: tokenization for text, image transforms for vision, audio feature extraction for speech. Batching is handled transparently.

'''Preprocessing by modality:'''
* Text: Tokenization with padding and truncation
* Images: Resize, normalize, convert to tensors
* Audio: Extract mel spectrograms or waveform features
* Documents: OCR + text extraction + layout encoding

=== Step 5: Model Inference ===
[[step::Principle:huggingface_transformers_Pipeline_Model_Forward]]

Execute the model's forward pass on the preprocessed inputs. The pipeline handles batch splitting, device transfers, and inference mode context. For generative tasks, this includes the full generation loop.

'''Inference details:'''
* Run with torch.no_grad() for efficiency
* Handle batch sizes larger than memory via chunking
* Support streaming for text generation
* Manage KV cache for efficient generation

=== Step 6: Output Postprocessing ===
[[step::Principle:huggingface_transformers_Pipeline_Postprocessing]]

Convert model outputs to human-readable results. Postprocessing is task-specific: decode token IDs to text, apply NMS for object detection, threshold masks for segmentation, etc.

'''Output formats:'''
* Classification: Labels with scores
* Generation: Generated text sequences
* NER/Tagging: Entities with spans and types
* Object Detection: Bounding boxes with labels and scores
* Segmentation: Masks with class assignments

== Execution Diagram ==
{{#mermaid:graph TD
    A[Task Resolution] --> B[Component Loading]
    B --> C[Pipeline Instantiation]
    C --> D[Input Preprocessing]
    D --> E[Model Inference]
    E --> F[Output Postprocessing]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Task_Resolution]]
* [[step::Principle:huggingface_transformers_Pipeline_Component_Loading]]
* [[step::Principle:huggingface_transformers_Pipeline_Instantiation]]
* [[step::Principle:huggingface_transformers_Pipeline_Preprocessing]]
* [[step::Principle:huggingface_transformers_Pipeline_Model_Forward]]
* [[step::Principle:huggingface_transformers_Pipeline_Postprocessing]]
