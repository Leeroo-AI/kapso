# Principle Index: Jaymody_PicoGPT

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Jaymody_PicoGPT_Model_Loading | [→](./principles/Jaymody_PicoGPT_Model_Loading.md) | ✅Impl:Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params, ✅Heuristic:Jaymody_PicoGPT_Model_Size_Memory_Requirements | Loading GPT-2 weights from TF checkpoints |
| Jaymody_PicoGPT_Input_Tokenization | [→](./principles/Jaymody_PicoGPT_Input_Tokenization.md) | ✅Impl:Jaymody_PicoGPT_Encoder_Encode | BPE tokenization text→IDs |
| Jaymody_PicoGPT_Transformer_Forward_Pass | [→](./principles/Jaymody_PicoGPT_Transformer_Forward_Pass.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Heuristic:Jaymody_PicoGPT_Context_Length_Limits | GPT-2 forward pass in NumPy |
| Jaymody_PicoGPT_Autoregressive_Generation | [→](./principles/Jaymody_PicoGPT_Autoregressive_Generation.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Heuristic:Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs | Greedy decoding generation loop |
| Jaymody_PicoGPT_Output_Decoding | [→](./principles/Jaymody_PicoGPT_Output_Decoding.md) | ✅Impl:Jaymody_PicoGPT_Encoder_Decode | BPE decoding IDs→text |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
