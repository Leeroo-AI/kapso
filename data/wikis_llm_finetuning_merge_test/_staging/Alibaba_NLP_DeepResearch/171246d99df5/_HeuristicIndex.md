# Heuristic Index: Alibaba_NLP_DeepResearch

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Alibaba_NLP_DeepResearch_Token_Limit_Management | [→](./heuristics/Alibaba_NLP_DeepResearch_Token_Limit_Management.md) | ✅Impl:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run, ✅Impl:Alibaba_NLP_DeepResearch_count_tokens, ✅Principle:Alibaba_NLP_DeepResearch_Context_Management | 110K token limit before forced answer |
| Alibaba_NLP_DeepResearch_Image_Resizing_Constraints | [→](./heuristics/Alibaba_NLP_DeepResearch_Image_Resizing_Constraints.md) | ✅Impl:Alibaba_NLP_DeepResearch_OmniSearch_process_image, ✅Impl:Alibaba_NLP_DeepResearch_OmniSearch_run_main, ✅Principle:Alibaba_NLP_DeepResearch_Image_Processing | max_pixels=1024*28*28, min_pixels=256*28*28 |
| Alibaba_NLP_DeepResearch_Locale_Detection_Search | [→](./heuristics/Alibaba_NLP_DeepResearch_Locale_Detection_Search.md) | ✅Impl:Alibaba_NLP_DeepResearch_Search_call, ✅Impl:Alibaba_NLP_DeepResearch_WebSearch_call, ✅Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution | Chinese character detection for locale |
| Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry | [→](./heuristics/Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry.md) | ✅Impl:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run, ✅Impl:Alibaba_NLP_DeepResearch_Visit_call, ✅Impl:Alibaba_NLP_DeepResearch_Call_Llm_Judge, ✅Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution | Exponential backoff with jitter for API calls |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
