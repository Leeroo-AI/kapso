# Repository Map: Jaymody_PicoGPT

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/jaymody/picoGPT |
| Branch | main |
| Generated | 2026-01-14 09:53 |
| Python Files | 4 |
| Total Lines | 385 |
| Explored | 4/4 |

## Structure


📖 README: `README.md`

---

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `encoder.py` | 120 | BPE tokenizer for GPT-2 | Impl: Jaymody_PicoGPT_Encoder_Encode, Jaymody_PicoGPT_Encoder_Decode; Principle: Jaymody_PicoGPT_Input_Tokenization, Jaymody_PicoGPT_Output_Decoding; Env: Jaymody_PicoGPT_Python_Dependencies | [→](./_files/encoder_py.md) |
| ✅ | `gpt2.py` | 121 | Full GPT-2 NumPy inference | Impl: Jaymody_PicoGPT_Gpt2, Jaymody_PicoGPT_Generate; Principle: Jaymody_PicoGPT_Transformer_Forward_Pass, Jaymody_PicoGPT_Autoregressive_Generation; Env: Jaymody_PicoGPT_Python_Dependencies; Heur: Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs, Jaymody_PicoGPT_No_KV_Cache_Performance, Jaymody_PicoGPT_Context_Length_Limits | [→](./_files/gpt2_py.md) |
| ✅ | `gpt2_pico.py` | 62 | Minimal ~60 line GPT-2 | Workflow: Jaymody_PicoGPT_Text_Generation | [→](./_files/gpt2_pico_py.md) |
| ✅ | `utils.py` | 82 | Model download and loading | Impl: Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params; Principle: Jaymody_PicoGPT_Model_Loading; Env: Jaymody_PicoGPT_Python_Dependencies; Heur: Jaymody_PicoGPT_Model_Size_Memory_Requirements | [→](./_files/utils_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description |
|-------|-------------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |
