# Principle Index: Jaymody_PicoGPT

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Jaymody_PicoGPT_Model_Download | [→](./principles/Jaymody_PicoGPT_Model_Download.md) | ✅Impl:Jaymody_PicoGPT_Download_Gpt2_Files, ✅Heuristic:Jaymody_PicoGPT_Streaming_Download_Large_Files | Downloads GPT-2 checkpoint files from OpenAI Azure blob storage |
| Jaymody_PicoGPT_Weight_Conversion | [→](./principles/Jaymody_PicoGPT_Weight_Conversion.md) | ✅Impl:Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt | Converts TensorFlow checkpoints to NumPy arrays |
| Jaymody_PicoGPT_BPE_Tokenization | [→](./principles/Jaymody_PicoGPT_BPE_Tokenization.md) | ✅Impl:Jaymody_PicoGPT_Encoder, ✅Heuristic:Jaymody_PicoGPT_BPE_Caching_LRU | Byte-pair encoding tokenization algorithm |
| Jaymody_PicoGPT_Text_Encoding | [→](./principles/Jaymody_PicoGPT_Text_Encoding.md) | ✅Impl:Jaymody_PicoGPT_Encoder_Encode | Converts text to token IDs |
| Jaymody_PicoGPT_Autoregressive_Generation | [→](./principles/Jaymody_PicoGPT_Autoregressive_Generation.md) | ✅Impl:Jaymody_PicoGPT_Generate, ✅Heuristic:Jaymody_PicoGPT_Sequence_Length_Validation | Token-by-token text generation loop |
| Jaymody_PicoGPT_Transformer_Architecture | [→](./principles/Jaymody_PicoGPT_Transformer_Architecture.md) | ✅Impl:Jaymody_PicoGPT_Gpt2, ✅Heuristic:Jaymody_PicoGPT_Causal_Masking_Large_Negative, ✅Heuristic:Jaymody_PicoGPT_Pre_Norm_Architecture, ✅Heuristic:Jaymody_PicoGPT_Weight_Tying_Embeddings, ✅Heuristic:Jaymody_PicoGPT_Stable_Softmax | GPT-2 decoder-only transformer architecture |
| Jaymody_PicoGPT_Text_Decoding | [→](./principles/Jaymody_PicoGPT_Text_Decoding.md) | ✅Impl:Jaymody_PicoGPT_Encoder_Decode | Converts token IDs back to text |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
