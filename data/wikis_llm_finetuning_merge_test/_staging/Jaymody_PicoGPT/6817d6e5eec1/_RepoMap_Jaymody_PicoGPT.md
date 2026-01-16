# Repository Map: Jaymody_PicoGPT

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/jaymody/picoGPT |
| Branch | main |
| Generated | 2026-01-15 19:00 |
| Python Files | 4 |
| Total Lines | 385 |
| Explored | 4/4 |

## Structure


📖 README: `README.md`

---

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `encoder.py` | 120 | BPE tokenizer encode/decode | Impl: Jaymody_PicoGPT_Encoder, Jaymody_PicoGPT_Encoder_Encode, Jaymody_PicoGPT_Encoder_Decode; Principle: Jaymody_PicoGPT_BPE_Tokenization, Jaymody_PicoGPT_Text_Encoding, Jaymody_PicoGPT_Text_Decoding; Heur: Jaymody_PicoGPT_BPE_Caching_LRU | [→](./_files/encoder_py.md) |
| ✅ | `gpt2.py` | 121 | Main GPT-2 implementation | Impl: Jaymody_PicoGPT_Gpt2, Jaymody_PicoGPT_Generate; Principle: Jaymody_PicoGPT_Transformer_Architecture, Jaymody_PicoGPT_Autoregressive_Generation; Heur: Jaymody_PicoGPT_Causal_Masking_Large_Negative, Jaymody_PicoGPT_Pre_Norm_Architecture, Jaymody_PicoGPT_Weight_Tying_Embeddings, Jaymody_PicoGPT_Stable_Softmax, Jaymody_PicoGPT_Sequence_Length_Validation | [→](./_files/gpt2_py.md) |
| ✅ | `gpt2_pico.py` | 62 | Minimal pico GPT-2 version | (Alternative implementation - same as gpt2.py) | [→](./_files/gpt2_pico_py.md) |
| ✅ | `utils.py` | 82 | Model loading utilities | Impl: Jaymody_PicoGPT_Download_Gpt2_Files, Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt; Principle: Jaymody_PicoGPT_Model_Download, Jaymody_PicoGPT_Weight_Conversion; Heur: Jaymody_PicoGPT_Streaming_Download_Large_Files; Env: Jaymody_PicoGPT_Python_Dependencies | [→](./_files/utils_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description | Count |
|-------|-------------|-------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections | 1 |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations | 7 |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations | 7 |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages | 1 |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages | 7 |
