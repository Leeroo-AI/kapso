{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Pipeline Tutorial|https://huggingface.co/docs/transformers/pipeline_tutorial]]
* [[source::Doc|Pipelines Reference|https://huggingface.co/docs/transformers/main_classes/pipelines]]
|-
! Domains
| [[domain::Inference]], [[domain::NLP]], [[domain::Computer_Vision]], [[domain::Audio]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
End-to-end high-level inference workflow that abstracts preprocessing, model inference, and postprocessing for diverse ML tasks including text, audio, vision, and multimodal inputs.

=== Description ===
The Pipeline system is the primary inference API in HuggingFace Transformers. It provides a unified interface for running inference across 30+ task types including text generation, classification, question answering, speech recognition, image classification, and multimodal understanding.

The Pipeline workflow follows a three-stage architecture:
1. **Preprocessing**: Convert raw inputs (text, images, audio) into model-ready tensors
2. **Forward Pass**: Execute model inference with automatic device management
3. **Postprocessing**: Transform model outputs into user-friendly formats

Key features include automatic model selection based on task, batching support, GPU acceleration, and compatibility with the HuggingFace Hub's 1M+ model checkpoints.

=== Usage ===
Execute this workflow when you need to:
* Run inference on pretrained models without manual preprocessing
* Process text, images, audio, or video through ML models
* Quickly prototype or deploy models for NLP/CV/Audio tasks
* Leverage the HuggingFace Hub's model ecosystem for immediate use

Typical input: Raw data (text strings, image paths/URLs, audio files)
Typical output: Structured predictions (labels, generated text, bounding boxes, etc.)

== Execution Steps ==

=== Step 1: Task and Model Resolution ===
[[step::Principle:huggingface_transformers_Task_Model_Resolution]]

Determine the appropriate pipeline class and model based on the specified task. The pipeline registry maps task names to implementation classes and default models. If a model is provided as a string, resolve its configuration and architecture.

'''Key considerations:'''
* Task aliases are resolved (e.g., "sentiment-analysis" → "text-classification")
* Model architecture is inferred from config if not explicitly specified
* Default models are selected per task when none is provided

=== Step 2: Tokenizer/Processor Loading ===
[[step::Principle:huggingface_transformers_Processor_Loading]]

Load the appropriate preprocessing components based on the task modality. This may include tokenizers for text, image processors for vision tasks, feature extractors for audio, or composite processors for multimodal inputs.

'''Component selection logic:'''
* Text tasks load PreTrainedTokenizer
* Vision tasks load BaseImageProcessor
* Audio tasks load SequenceFeatureExtractor
* Multimodal tasks load ProcessorMixin composites

=== Step 3: Model Loading and Device Placement ===
[[step::Principle:huggingface_transformers_Pipeline_Model_Loading]]

Load the pretrained model weights and place the model on the appropriate compute device. Handle quantization, dtype selection, and accelerate device maps for efficient memory usage.

'''Device placement strategy:'''
* Respect user-specified device parameter
* Use accelerate device_map if model was loaded with it
* Default to GPU 0 if available, else CPU
* Support for MLU, NPU, XPU, HPU, MPS backends

=== Step 4: Input Preprocessing ===
[[step::Principle:huggingface_transformers_Pipeline_Preprocessing]]

Transform raw inputs into model-ready tensor dictionaries. Each pipeline subclass implements task-specific preprocessing that handles input normalization, tokenization, and tensor creation.

'''Preprocessing responsibilities:'''
* Validate and normalize input format
* Apply tokenization or feature extraction
* Create attention masks and other auxiliary tensors
* Handle batching and padding for multiple inputs

=== Step 5: Model Forward Pass ===
[[step::Principle:huggingface_transformers_Pipeline_Forward]]

Execute the model's forward pass within an inference context. Tensors are moved to the model's device, inference is run under torch.no_grad(), and outputs are moved back to CPU for postprocessing.

'''Forward pass guarantees:'''
* Automatic device placement for input tensors
* Gradient computation disabled for efficiency
* Output tensors returned on CPU for postprocessing

=== Step 6: Output Postprocessing ===
[[step::Principle:huggingface_transformers_Pipeline_Postprocessing]]

Transform model outputs into user-friendly formats. Each pipeline subclass implements task-specific postprocessing that converts raw logits/hidden states into structured predictions.

'''Postprocessing outputs:'''
* Text generation: Generated strings with metadata
* Classification: Labels with confidence scores
* Detection: Bounding boxes with class labels
* QA: Answer spans with confidence scores

== Execution Diagram ==
{{#mermaid:graph TD
    A[Task & Model Resolution] --> B[Processor Loading]
    B --> C[Model Loading & Device Placement]
    C --> D[Input Preprocessing]
    D --> E[Model Forward Pass]
    E --> F[Output Postprocessing]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Task_Model_Resolution]]
* [[step::Principle:huggingface_transformers_Processor_Loading]]
* [[step::Principle:huggingface_transformers_Pipeline_Model_Loading]]
* [[step::Principle:huggingface_transformers_Pipeline_Preprocessing]]
* [[step::Principle:huggingface_transformers_Pipeline_Forward]]
* [[step::Principle:huggingface_transformers_Pipeline_Postprocessing]]
