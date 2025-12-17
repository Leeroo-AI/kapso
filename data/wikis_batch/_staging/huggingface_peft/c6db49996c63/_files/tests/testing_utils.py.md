# tests/testing_utils.py

## Understanding

### Purpose
Provides test utilities and decorators for PEFT tests including hub_online_once caching, hardware requirement decorators (require_torch_gpu, require_bitsandbytes), and helper functions (get_state_dict, load_cat_image).

### Mechanism
- hub_online_once context manager uses _HubOnlineOnce singleton to cache hub downloads: first call downloads and caches to cache_dir, subsequent calls use cached model avoiding hub connection
- Hardware decorators: require_non_cpu checks torch.cuda.is_available() or xpu, require_torch_gpu checks xpu or mps or cuda with compute_capability >= 7.0 (Volta+), require_torch_multi_gpu checks device_count >= 2, require_bitsandbytes checks importlib.util.find_spec("bitsandbytes") and compute_capability >= 7.5 or cpu
- skip_non_gpu decorator requires xpu/cuda, skip_non_multi_gpu requires device_count >= 2
- temp_seed context manager sets random seed temporarily: torch.manual_seed(seed), np.random.seed(seed), random.seed(seed), restores old_state on exit
- get_state_dict helper extracts state dict from model or PeftModel.base_model.model
- set_init_weights_false helper sets init_lora_weights/init_weights=False for config_cls in config_kwargs
- load_dataset_english_quotes loads "Abirate/english_quotes" dataset
- load_cat_image loads "ybelkada/random-images" cat image with PIL.Image.open
- create_tests_from_file generates parameterized tests from data files

### Significance
Critical infrastructure enabling tests to run efficiently (caching hub downloads), skip appropriately on unsupported hardware (CPU-only, single GPU, non-Volta GPUs), and maintain reproducibility (temp_seed), reducing test flakiness and CI costs.
