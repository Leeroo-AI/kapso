# Implementation Index: Jaymody_PicoGPT

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params | [→](./implementations/Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params.md) | ✅Principle:Jaymody_PicoGPT_Model_Loading, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_Model_Size_Memory_Requirements | utils.py:L68-82 | Model loading orchestrator |
| Jaymody_PicoGPT_Encoder_Encode | [→](./implementations/Jaymody_PicoGPT_Encoder_Encode.md) | ✅Principle:Jaymody_PicoGPT_Input_Tokenization, ✅Env:Jaymody_PicoGPT_Python_Dependencies | encoder.py:L101-106 | BPE text→IDs |
| Jaymody_PicoGPT_Gpt2 | [→](./implementations/Jaymody_PicoGPT_Gpt2.md) | ✅Principle:Jaymody_PicoGPT_Transformer_Forward_Pass, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_No_KV_Cache_Performance, ✅Heuristic:Jaymody_PicoGPT_Context_Length_Limits | gpt2.py:L73-83 | Full forward pass |
| Jaymody_PicoGPT_Generate | [→](./implementations/Jaymody_PicoGPT_Generate.md) | ✅Principle:Jaymody_PicoGPT_Autoregressive_Generation, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs, ✅Heuristic:Jaymody_PicoGPT_No_KV_Cache_Performance, ✅Heuristic:Jaymody_PicoGPT_Context_Length_Limits | gpt2.py:L86-94 | Greedy generation loop |
| Jaymody_PicoGPT_Encoder_Decode | [→](./implementations/Jaymody_PicoGPT_Encoder_Decode.md) | ✅Principle:Jaymody_PicoGPT_Output_Decoding, ✅Env:Jaymody_PicoGPT_Python_Dependencies | encoder.py:L108-111 | BPE IDs→text |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
