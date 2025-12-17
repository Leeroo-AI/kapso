**Status:** ✅ Explored

**Purpose:** Implements AMD's Quark quantization framework integration for loading prequantized models. Handles QParamsLinear modules with separate quantizers for weights, inputs, biases, and outputs.

**Mechanism:** The QuarkHfQuantizer class (requires_calibration=True) validates quark library availability and maps models to Quark format through _map_to_quark() before weight loading, using json_export_config for pack_method and custom_mode settings. It implements WeightConverter operations with QuarkDeserialize to handle Quark's state_dict key renaming convention, where keys like "weight_quantizer.scale" are saved as "weight_scale". The CHECKPOINT_KEYS mapping defines 8 quantizer parameters (scale/zero_point for weight/bias/input/output), and converters assign these to appropriate quantizer attributes during loading.

**Significance:** Quark provides AMD's optimized quantization solution tailored for their hardware ecosystem. The explicit handling of Quark's state_dict serialization format ensures seamless model loading, while the support for multiple quantizer types (weight, bias, input, output) enables comprehensive quantization strategies. The calibration requirement indicates this is focused on post-training quantization with pre-calibrated models, making it suitable for deployment scenarios with AMD accelerators.
