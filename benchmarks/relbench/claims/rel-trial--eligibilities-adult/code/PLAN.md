TIME ALLOCATION: Critical-path artifact is the full BioClinical ModernBERT Model A/B checkpoint chain; the first full attempt measured 1,864 examples/minute at micro-batch 4, and the revised target is at least 2,500 examples/minute at micro-batch 16 with the same effective batch of 32.
TIME ALLOCATION: Confirm after tokenization, the first 250 optimizer steps, every forward-year score, and the frozen Model A validation vector.
TIME ALLOCATION: Freeze optimization 210 minutes after the full candidate starts, reserving the remaining 30-minute harness window for Model A/B inference, CatBoost inference, validation, and serialization.

Revision: the first 1,500 optimizer steps showed only 34.4% mean assigned-GPU utilization and 3.5 GB memory. The attempt was stopped before checkpoint/archive creation and micro-batch was increased from 4 to 16 to raise the bounded checkpoint-growth rate without changing effective batch or optimizer hyperparameters.

# Plan

1. Profile immutable scoring, temporal split geometry, text lengths, relation availability, and structured categories.
2. Build censored relational documents, structured features, historical priors, and persistent token/feature caches.
3. Train the sequential 2017–2019 encoder curriculum and structured forward models, then select the fixed internal blend.
4. Freeze Model A validation predictions, adapt Model B using validation labels only afterward, and produce test predictions.
5. Run the fast contract gate followed by the registered full evaluation and preserve its complete manifest output.
