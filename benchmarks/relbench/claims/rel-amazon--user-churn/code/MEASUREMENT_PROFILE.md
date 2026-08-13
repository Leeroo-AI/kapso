# Evaluation and input profile

## Evaluator mechanics

The immutable registered evaluator runs `main.py` in an isolated child, loads both NumPy vectors, validates length and finiteness, and calls the official RelBench task evaluator on all 409,792 validation rows. `--fraction` and `--seed` are manifest metadata for this suite and do not subsample scoring rows. The selection metric is validation ROC AUC; average precision, accuracy, and F1 are also reported. Full runs are archived with code and both prediction vectors. Validation predictions must come from a fit that never consumes validation labels; the test fit may add validation outcomes.

## Origins and label distribution

Train contains 4,708,383 rows and 1,486,748 customers over 31 quarterly origins from 2008-01-10 through 2015-07-02. The last eight origins begin 2013-10-03. The two prescribed forward origins have 416,981 rows at 2015-04-02 with positive rate 0.59635 and 415,013 rows at 2015-07-02 with positive rate 0.59867. Validation has 409,792 unique customers at 2015-10-01 and positive rate 0.64203. Test has 351,885 unique customers at 2016-01-01; its outcomes were not accessed.

## Review and history distribution

The review table has 12,644,508 rows, 1,584,084 observed customers, 416,125 observed products, and 11,824,697 distinct summary/body content hashes. There are 3,366,645 customer-time duplicate follow-up rows. Within-customer gap quantiles at 0/10/25/50/75/90/95/99 percent are 0/0/0/11/77/264/466/1157 days.

Validation lifetime depth quantiles at 10/25/50/75/90/95/99 percent are 2/3/6/11/22/38/124; 14.74%, 6.13%, and 2.48% exceed lengths 16, 32, and 64. Test quantiles are 2/4/6/12/25/44/143; 16.93%, 7.30%, and 3.02% exceed those lengths. Maximum depths are 4,503 and 4,655. Validation recency quantiles at 10/25/50/75/90 percent are 6/18/37/62/79 days, versus 8/27/46/66/80 for test. Nearly all eligible histories contain review text.

Review summary length quantiles at 0/25/50/75/90/95/99 percent are 0/11/19/34/51/63/94 characters; body quantiles are 0/123/239/628/1509/2267/4206. Product title quantiles at 25/50/75/90/99 percent are 25/43/65/90/155 characters and description quantiles are 0/279/880/1469/4231. Product category and description have 18,854 and 36,395 null rows, respectively.

## Coverage axes and critical path

Reported slices are lifetime depth 1-5/6-32/33+, recency 0-30/31-60/61-91 days, any text/no text, truncated/not truncated, and last-rating extreme/non-extreme. The measured recency and depth shifts support explicit absolute lags, within-origin ranks, 32 ordered events, and summary tokens. The critical path is the 11.8-million-document ModernBERT cache: a 4,096-document probe reached 883 documents/s at length 128 and an 8,192-document probe reached 2,094 documents/s at the planned 96-token fallback, so the one-million-document confirmation controls the final truncation and schedule.

## Conditional-survival iteration extension

Exact timestamp profiling confirms that all 12,644,508 reviews fall on 2,923 midnight timestamps, so customer-day and customer-timestamp coalescing are identical. The 12,644,508 rows reduce to 9,277,863 customer-time events, removing 3,366,645 same-time follow-up rows. This makes positive completed gaps the correct self-supervised unit and avoids assigning density to zero-length intervals.

The registered evaluation still scores every validation row and does not use `fraction` to subsample. The additional coverage axes are coalesced multiplicity, distinct products at an event, already-survived recency, and completed-versus-terminal likelihood contribution. The critical path for this iteration is measured coalesced-transition throughput rather than text extraction because the full ModernBERT review and product caches already exist and are content-addressed.

The coalesced artifact built at 167,303 events/s. The initial 100,000-transition GPU probe measured 18,965-26,470 transitions/s under changing concurrent load, triggering the prescribed one-pass, 64-event fallback whenever it fell below 25,000; steady chronological passes reached 109,000-157,000 completed transitions/s. Four forward residual origins scored 0.679184/0.681468/0.697560/0.688168 for the selected latent/community stage. Final validation slices were depth 1-2/3-5/6+ AUC 0.588595/0.622064/0.737758 and recency 0-30/31-60/61-91 AUC 0.732709/0.660744/0.634731.
