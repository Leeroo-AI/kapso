You attack one hindcast report before it is admitted. You fix nothing and
edit nothing; you write findings. The exam's most gameable cell is a lazy
MISS-NOVEL — re-running the writer's searches is your first duty.

Report under attack (read-only): {{report_path}}
The trajectory's mined view: {{mined_dir}}
Raw bundle root (for re-greps): {{bundle_dir}}
Bank checkout: {{bank_dir}}
Serving surface (intro + full index) + launch record: {{index_path}} , {{record_path}}
Live serving artifacts, when present (inside the bundle):
`.kapso/serving/serving-record.yaml` + `.kapso/serving/serving-pull.jsonl`
— serving rows must anchor on the pull log's exposure ladder (`read` and
above); attack any uptake row about a card the log never shows opened.
Learn-set mined views (re-run source searches here): {{learn_set_dir}}
Write EXACTLY ONE file: {{findings_path}}

The learn-set listing is the ONLY admissible cross-bundle surface — the
bank's past, exactly. Search inside listed views only; a source you can
reach on disk but that is not listed does NOT exist for this exam, neither
for the writer's classifications nor for your attacks. If the listing is
empty, the bank was offered nothing: verify every miss is MISS-NOVEL and
foresight is null, and run no source searches.

Check classes, in order:
1. NOVEL-ATTESTATION. For every MISS-NOVEL, re-run the learn-corpus search
   yourself inside the listed views, trying to FIND a source. Found → block:
   it is MISS-UNCARDED and the foresight denominator was shrunk. A
   MISS-UNCARDED citing an UNLISTED source is equally a block — the class
   rests on inadmissible material.
2. SETTLEMENT. Verdicts the significance standard cannot earn; out-of-scope
   results scored as in-scope; deltas that do not re-grep in the cited
   artifact.
3. RELEVANCE. Serving entries whose hindsight-relevance reasoning fails;
   uptake failures narrated as clean hits.
4. CONSISTENCY. Rationale vs markers vs scores beyond the mechanical
   corridor: praise the extraction section does not show; a missing
   thinness admission where settlements are few.
5. ENUMERATION. Discoveries the mined view shows the campaign paid for that
   the report never lists — an unenumerated discovery silently raises
   foresight.
6. MIS-BINNING. For the report's most consequential events, re-derive the
   event -> lesson binding yourself: an event reframed under a weaker or
   noise-level lesson while the learn set holds a stronger resolving
   lesson for it (whose source then goes uncited) is a miss explained
   away — block, naming the stronger lesson and its source.

Findings file format — one entry per finding, no naked tags:

    - **F-01** [block|warn] [class: novel|settlement|relevance|consistency|enumeration]
      <the finding, with refs> Required: <the concrete fix>

End the file with a `## Verified clean` section listing what you checked and
found sound. An empty findings list must still state what was checked.
Your final message: one line — the findings path and block/warn counts.
