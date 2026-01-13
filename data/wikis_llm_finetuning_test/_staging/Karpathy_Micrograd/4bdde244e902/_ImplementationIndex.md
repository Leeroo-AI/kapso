# Implementation Index: Karpathy_Micrograd

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| Karpathy_Micrograd_Data_Preparation_Pattern | [→](./implementations/Karpathy_Micrograd_Data_Preparation_Pattern.md) | ✅Principle:Karpathy_Micrograd_Data_Preparation, ⬜Env:Karpathy_Micrograd_Python_3 | (user code) | Pattern Doc: Python lists |
| Karpathy_Micrograd_MLP_Init | [→](./implementations/Karpathy_Micrograd_MLP_Init.md) | ✅Principle:Karpathy_Micrograd_Network_Architecture_Definition, ⬜Env:Karpathy_Micrograd_Python_3 | nn.py:L45-60 | Network construction |
| Karpathy_Micrograd_MLP_Call | [→](./implementations/Karpathy_Micrograd_MLP_Call.md) | ✅Principle:Karpathy_Micrograd_Forward_Pass_Computation, ⬜Env:Karpathy_Micrograd_Python_3 | nn.py:L51-54 | Forward pass |
| Karpathy_Micrograd_Loss_Operations | [→](./implementations/Karpathy_Micrograd_Loss_Operations.md) | ✅Principle:Karpathy_Micrograd_Loss_Computation, ⬜Env:Karpathy_Micrograd_Python_3 | engine.py:L35-79 | Value arithmetic for loss |
| Karpathy_Micrograd_Value_Backward | [→](./implementations/Karpathy_Micrograd_Value_Backward.md) | ✅Principle:Karpathy_Micrograd_Backward_Pass, ⬜Env:Karpathy_Micrograd_Python_3 | engine.py:L54-70 | Reverse autodiff |
| Karpathy_Micrograd_Module_Parameters | [→](./implementations/Karpathy_Micrograd_Module_Parameters.md) | ✅Principle:Karpathy_Micrograd_Parameter_Update, ⬜Env:Karpathy_Micrograd_Python_3 | nn.py:L10-57 | Parameter access |
| Karpathy_Micrograd_Module_Zero_Grad | [→](./implementations/Karpathy_Micrograd_Module_Zero_Grad.md) | ✅Principle:Karpathy_Micrograd_Training_Loop, ⬜Env:Karpathy_Micrograd_Python_3 | nn.py:L6-8 | Gradient reset |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
