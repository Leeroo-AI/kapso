Your previous response had evidence quotes that don't exist in the files.

ERRORS (quotes not found):
{{error_feedback}}

Please regenerate the repo model with CORRECT quotes.
- Quotes must be EXACT VERBATIM substrings from the file content
- Copy character-for-character, do NOT paraphrase
- Use shorter quotes (function/class names) if unsure
- Ensure evidence `path` matches the file the quote comes from (use the `=== FILE: ... ===` headers).
- IMPORTANT: Do NOT "rewrite" a quote to make it shorter.
  - Instead, select a shorter substring that already exists in the file.
- IMPORTANT: Do NOT invent pseudo-code as evidence.
  - Bad (usually not present verbatim): `if not data: return []`
  - Good: quote an actual line like `if not os.path.exists(filepath):` or `return []` that exists exactly.
- If you cannot find an exact quote that supports a claim, REMOVE the claim (do not invent).
- For multi-line code (configs / function calls), do NOT collapse it into a fake one-liner quote.
  - Instead, include MULTIPLE evidence items, each quoting a real single line.

Your previous model (fix the evidence):
{{previous_model_json}}

FILE CONTENTS (copy quotes EXACTLY from here):
{{files_payload}}

Return ONLY valid JSON with the same schema, but with corrected evidence quotes.

