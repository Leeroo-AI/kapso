# Campaign reward bar (this defines success)

KumoRFM (fine-tuned) — the strongest foundation-model baseline on this task —
scores **TEST MAE 2.731 (board NMAE 0.3887)**. That is the bar this campaign
is measured against:

- The campaign earns reward ONLY by SURPASSING KumoRFM-ft: a final
  val-selected TEST MAE strictly below 2.731.
- Any final result that does not beat 2.731 carries ZERO reward — a polished
  result at 3.3, 3.5, or 3.8 counts exactly the same as no result. Do not
  optimize for "best among results below the bar"; optimize for crossing it.
- The score you see in-loop is VALIDATION MAE; the bar lives on TEST. Prior
  campaigns on this task measured val→test drift of +0.8 to +1.3 MAE (the
  2010–2016 era shift): a validation win that does not survive that shift is
  worthless. Treat generalization across the test horizon as the primary
  design constraint, not an afterthought.
- STOP RULE (hard): an in-loop VALIDATION score is NEVER goal achievement,
  no matter how far it sits below 2.731 — that number is a TEST bar, and
  validation is a different split. Only a validation MAE at or below 1.9
  (= 2.731 minus the smallest drift ever measured on this task) could even
  in principle indicate the test bar has been reached; no stop verdict,
  "goal achieved" claim, or early termination is permitted while validation
  is above 1.9. Campaigns end on budget, not on validation optimism.
