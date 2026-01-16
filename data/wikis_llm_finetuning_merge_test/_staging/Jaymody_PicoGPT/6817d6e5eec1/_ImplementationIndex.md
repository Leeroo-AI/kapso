# Implementation Index: Jaymody_PicoGPT

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| Jaymody_PicoGPT_Download_Gpt2_Files | [→](./implementations/Jaymody_PicoGPT_Download_Gpt2_Files.md) | ✅Principle:Jaymody_PicoGPT_Model_Download, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_Streaming_Download_Large_Files | utils.py:L13-41 | Downloads model files from Azure blob storage |
| Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt | [→](./implementations/Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt.md) | ✅Principle:Jaymody_PicoGPT_Weight_Conversion, ✅Env:Jaymody_PicoGPT_Python_Dependencies | utils.py:L44-65 | Converts TF checkpoint to NumPy dict |
| Jaymody_PicoGPT_Encoder | [→](./implementations/Jaymody_PicoGPT_Encoder.md) | ✅Principle:Jaymody_PicoGPT_BPE_Tokenization, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_BPE_Caching_LRU | encoder.py:L47-120 | BPE tokenizer class (encode/decode/bpe) |
| Jaymody_PicoGPT_Encoder_Encode | [→](./implementations/Jaymody_PicoGPT_Encoder_Encode.md) | ✅Principle:Jaymody_PicoGPT_Text_Encoding, ✅Env:Jaymody_PicoGPT_Python_Dependencies | encoder.py:L101-106 | Text to token ID conversion |
| Jaymody_PicoGPT_Generate | [→](./implementations/Jaymody_PicoGPT_Generate.md) | ✅Principle:Jaymody_PicoGPT_Autoregressive_Generation, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_Sequence_Length_Validation | gpt2.py:L86-94 | Autoregressive generation loop |
| Jaymody_PicoGPT_Gpt2 | [→](./implementations/Jaymody_PicoGPT_Gpt2.md) | ✅Principle:Jaymody_PicoGPT_Transformer_Architecture, ✅Env:Jaymody_PicoGPT_Python_Dependencies, ✅Heuristic:Jaymody_PicoGPT_Causal_Masking_Large_Negative, ✅Heuristic:Jaymody_PicoGPT_Pre_Norm_Architecture, ✅Heuristic:Jaymody_PicoGPT_Weight_Tying_Embeddings, ✅Heuristic:Jaymody_PicoGPT_Stable_Softmax | gpt2.py:L73-83 | GPT-2 forward pass in NumPy |
| Jaymody_PicoGPT_Encoder_Decode | [→](./implementations/Jaymody_PicoGPT_Encoder_Decode.md) | ✅Principle:Jaymody_PicoGPT_Text_Decoding, ✅Env:Jaymody_PicoGPT_Python_Dependencies | encoder.py:L108-111 | Token ID to text conversion |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
