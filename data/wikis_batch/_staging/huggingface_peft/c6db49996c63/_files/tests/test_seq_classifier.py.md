# tests/test_seq_classifier.py

## Understanding

### Purpose
Validates PEFT adapters work correctly with sequence classification models across various architectures (BERT, RoBERTa, Llama) and ensures classification heads are properly handled as modules_to_save.

### Mechanism
- Tests ALL_CONFIGS (18+ adapter types: AdaLoRA, BOFT, Bone, C3A, DeloRA, FourierFT, Gralora, HRA, IA3, LoRA, OFT, PrefixTuning, PromptEncoder, PromptTuning, Road, Shira, VBLoRA, Vera, WaveFT) against PEFT_SEQ_CLS_MODELS_TO_TEST (tiny BERT, RoBERTa, Llama sequence classifiers)
- Parameterized tests call PeftCommonTester methods (_test_model_attr, _test_prepare_for_training, _test_save_pretrained, etc.) with task_type="SEQ_CLS"
- Specifically tests that classification head (classifier or score layer) is wrapped in ModulesToSaveWrapper to prevent adapter targeting of the classification head
- Includes special test for PromptTuning with TEXT initialization using tokenizer_name_or_path

### Significance
Critical for validating that PEFT correctly handles sequence classification task requirements where the classification head must be trainable but not adapter-modified, preventing issue #2027 where classification heads were incorrectly targeted by adapters with target_modules="all-linear".
