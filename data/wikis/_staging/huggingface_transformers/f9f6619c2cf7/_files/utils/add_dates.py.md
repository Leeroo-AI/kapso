# utils/add_dates.py

## Understanding

### Purpose
Manages model documentation dates

### Mechanism
The script automates the addition and validation of release dates and HuggingFace commit dates in model documentation cards. It fetches paper publication dates from HuggingFace Papers API, determines first commit dates via git history, and updates model documentation with structured date metadata. The tool supports checking for missing/incorrect dates, replacing arxiv links with HuggingFace paper links, and bulk processing of all models or specific modified models.

### Significance
Ensures consistent and accurate temporal metadata across all model documentation, making it easier for users to understand when models were released and added to the library. The automated validation in CI prevents documentation from becoming outdated or inconsistent.
