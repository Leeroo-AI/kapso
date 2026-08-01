# Evaluation profile — rel-ratebeer/user-count

## Measurement mechanics

The immutable registered evaluator invokes `main.py` in a child process, requires finite original-order validation and test arrays of shapes `(19908,)` and `(9392,)`, and computes official RelBench `r2`, `mae`, and `rmse` on all validation rows. The manifest score is validation `r2`; `fraction` and `seed` do not subsample scoring. Test labels are absent. Full runs are archived. The protected `kapso_evaluation/` directory was read but not modified, so this required profile is stored at repository root.

## Input profile

- Official train has 373,709 rows, 154,071 users, and 74 origins from 2000-06-07 through 2018-06-03. Origin sizes range from 33 to 16,763.
- Validation is one origin at 2018-09-01 with 19,908 users. Test is one origin at 2020-01-01 with 9,392 users.
- Only 9,359 validation users and 3,307 test users occur anywhere in official train; 3,055 test users occur in validation. Historical event features therefore carry most user identity generalization.
- Train targets have 187,067 zeros (50.1%); 3,708 rows above 332 contribute 2,051,976 target counts. Validation has 12,483 zeros (62.7%); 113 rows above 332 contribute 67,475 counts. Raw-scale tail accuracy controls R2.
- Monthly origins from 2012-01-01 through 2019-10-01 plus 2019-10-03 yield 1,068,347 eligible episode rows before official-row deduplication. The model-A portion through 2018-06-01 has 797,801 rows.
- Beer-rating volume peaks at 1,177,568 events in 2015 and falls to 853,546 in 2019; annual active raters change sharply, including 51,636 in 2018 and 31,984 in 2019. Site-regime features are required.
- Beer ratings span 2000-04-12 through 2019-12-31; place ratings span 2004-01-03 through 2019-12-31; favorites begin only on 2018-05-02. Availability `created_at` values span 2024-01-01 through 2025-02-03, so strict `created_at <= seed` removes every availability row for all official and auxiliary seeds.
- Rating language is concentrated: English is 10,637,872 of 11,847,969 rows, null is 469,093, then Polish 189,045 and French 132,001. Concentration features must retain an other-language bucket.

## Coverage axes and strata

Coverage axes are forecast origin/regime, recent 90-day intensity, zero versus nonzero target, heavy-tail target band, user tenure/recency, auxiliary-table presence, relational breadth/concentration, and platform activity/share. Reported target strata are `0`, `1-2`, `3-15`, `16-70`, `71-332`, and `333+`; internal evaluation additionally aggregates by forecast origin before averaging.

The solution assumption that availability supplies historical signal is contradicted by the sanitized timestamps and is excluded after the mandatory temporal filter. Relational utility remains an internally tested assumption. Mutable snapshot aggregates and untimed UPC rows remain excluded.
