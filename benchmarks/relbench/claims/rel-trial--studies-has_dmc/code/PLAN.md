TIME ALLOCATION: Critical path = three-pass MedCPT embeddings at measured 540 short documents/s; target sustained >=150 documents/s on long passages and completion by T+85 minutes.
TIME ALLOCATION: Confirm after the first 2,048 real documents, after each cached passage, and after causal retrieval reaches 50,000/200,000 rows.
TIME ALLOCATION: Freeze model/features at T+185 minutes, reserving the final 35 minutes for contract checks and the required foreground full evaluation.

# Plan

1. Persist the metadata-only evaluation profile and campaign feature register.
2. Build aligned documents, structured relational features, causal empirical-Bayes priors, MedCPT embeddings, and historical retrieval features.
3. Fit forward-validated compact boosted trees with separate train-only validation and train-plus-validation test chains.
4. Exercise debug mode, validate prediction contracts, run the registered full evaluation, and record stratum diagnostics.

Revision after the first full run: the measured top-512 candidate support was constant at thresholds 0.30/0.50/0.70, making those features uninformative. With all three expensive passages already cached, spend one deliberate refinement run on thresholds 0.85/0.90/0.95 selected strictly by the predeclared 2017–2019 folds, then freeze.

Revision after threshold run: forward mean increased from 0.817093 to 0.817268, so retain the change despite validation noise. Use the remaining pre-freeze window for one label-free widening targeted at unseen-program losses: raw source identity and field-specific monitoring/DMC phrase evidence, admitted only if the same forward-fold mean improves.
