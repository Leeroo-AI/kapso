# Evaluation profile

## Mechanics

- The immutable grader launches `main.py` in an isolated child. Full fidelity runs `python main.py`; fast fidelity only adds `--debug`. Both score all 11,983 official validation rows, and `fraction`/`seed` are manifest metadata rather than subsampling controls.
- The headline objective is official validation ROC-AUC, so only probability ordering matters. All official outputs also include average precision, accuracy, and F1.
- Validation predictions must come from a train-only chain. Test predictions may use a separately rebuilt train-plus-validation chain. The grader requires aligned float arrays of shapes `(11983,)` and `(18944,)`.

## Input distribution

- Train has 202,840 rows from 2000-01-03 through 2019-12-31 at 35.17% positive. Validation has 11,983 rows in 2020 at 30.53% positive. Test has 18,944 rows across 2021/2022/2023 with 10,768/6,459/1,717 rows.
- Train date batching is material: 4,984 distinct dates, median 6 rows, 90th percentile 47, 99th percentile 943, maximum 1,420; 80.76% of dates contain multiple rows. Causal features must query each complete date batch before updating with its labels.
- The train positive rate trends from 42.79% in 2000 to 28.30% in 2019. Calendar time and changing cohort composition are coverage axes.
- Validation/test studies with any sponsor seen during train are 92.48%/89.31%; any condition seen are 79.95%/73.42%; any intervention seen are 22.44%/18.67%; any facility seen are 70.70%/61.85%. This confirms the proposed sparse/new-cohort assumption and makes global semantic fallback important.
- Mean relation degrees for train/validation/test are sponsor 1.61/1.56/1.47, condition 1.64/1.57/1.39, intervention 0.69/0.42/0.33, and facility 7.07/4.71/3.07. Missing relation rates are especially high for interventions and rise from 63.89% train to 80.97% test.
- Interventional share changes from 80.57% train to 73.15% validation and 77.33% test. `Not Applicable` phase rises from 36.13% to 42.29% and 51.55%. Source-class NIH declines from 0.87% to 0.31% and 0.14%.
- Text is dense: brief title and summary are complete; official title is 1.69% missing in train and complete later. Median summary length is 392/446/460 characters, detailed description 686/784/782, and eligibility criteria 764/689/686. Detailed descriptions are missing for roughly 29–32%.

## Coverage axes and reporting

- Time: report validation ROC-AUC by half-year and month where both classes occur.
- Cohort novelty: report seen/unseen lead sponsor, any known condition, and any known facility slices.
- Protocol: report interventional versus observational and major phase groups.
- Retrieval support: report by top-neighbor similarity and matching-candidate count.
- Text availability/length: report detailed-description presence and document-length bins.

## Critical path

The frozen MedCPT passage matrix bounds semantic retrieval quality and cannot be recovered by downstream tree tuning. The pinned encoder loaded successfully on the assigned A100; a 1,024-document short-text benchmark achieved 540.5 documents/s. Real 512-token passage throughput is checked during extraction, with three passage matrices cached incrementally before retrieval/model consumers are built.

The system declares `kapso_evaluation/` immutable and treats any edit there as tampering, so this profile is persisted at repository root instead of inside that protected directory.
