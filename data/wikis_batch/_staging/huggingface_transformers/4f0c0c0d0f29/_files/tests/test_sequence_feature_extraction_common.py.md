**Status:** ✅ Explored

**Purpose:** Tests common functionality for sequence feature extractors used in audio/speech processing models.

**Mechanism:** SequenceFeatureExtractionTestMixin extends FeatureExtractionSavingTestMixin to test audio feature extraction properties (feature_size, sampling_rate, padding_value), batch feature creation, padding operations, and PyTorch tensor conversion for sequence-based inputs.

**Significance:** Ensures audio feature extractors consistently handle variable-length sequences with proper padding and maintain compatibility across different tensor formats.
