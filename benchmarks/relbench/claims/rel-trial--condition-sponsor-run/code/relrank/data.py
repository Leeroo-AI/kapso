import re
import time

import numpy as np
import pandas as pd


def _number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    if not match:
        return np.nan
    number = float(match.group(1))
    text = str(value).lower()
    if "month" in text:
        number /= 12.0
    elif "week" in text:
        number /= 52.0
    elif "day" in text:
        number /= 365.25
    return number


def _contains(series: pd.Series, value: str) -> np.ndarray:
    return series.fillna("").astype(str).str.upper().str.contains(value, regex=False).to_numpy(dtype=np.float32)


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)


def _top_code(values: pd.Series):
    if len(values) == 0:
        return -1
    counts = values.dropna().astype(str).value_counts()
    return counts.index[0] if len(counts) else -1


class DataAssets:
    def __init__(self, context):
        started = time.time()
        self.context = context
        self.tables = {name: table.df for name, table in context.db.table_dict.items()}
        self.n_conditions = len(self.tables["conditions"])
        self.n_sponsors = len(self.tables["sponsors"])
        self.n_studies = len(self.tables["studies"])
        self.train = context.train.df.copy()
        self.val = context.val.df.copy()
        self.test = context.test.df.copy()
        self.condition_rel = self.tables["conditions_studies"][["nct_id", "condition_id", "date"]].copy()
        self.sponsor_rel = self.tables["sponsors_studies"][
            ["nct_id", "sponsor_id", "lead_or_collaborator", "date"]
        ].copy()
        self.events = self.condition_rel.merge(self.sponsor_rel, on="nct_id", suffixes=("_condition", "_sponsor"))
        self.events["visible"] = self.events[["date_condition", "date_sponsor"]].max(axis=1)
        self.study_features, self.study_feature_names, self.study_top_country = self._base_study_features()
        sponsors = self.tables["sponsors"].sort_values("sponsor_id")
        classes = sponsors["agency_class"].fillna("UNKNOWN").astype(str)
        vocabulary = {value: index for index, value in enumerate(sorted(classes.unique()))}
        self.agency_code = classes.map(vocabulary).to_numpy(dtype=np.int16)
        self.agency_vocabulary = vocabulary
        print(f"[data] loaded 15 tables and {len(self.events)} causal pair events in {time.time() - started:.1f}s")

    def _base_study_features(self):
        studies = self.tables["studies"].sort_values("nct_id")
        designs = self.tables["designs"].drop_duplicates("nct_id").set_index("nct_id").reindex(studies["nct_id"])
        eligibility = self.tables["eligibilities"].drop_duplicates("nct_id").set_index("nct_id").reindex(studies["nct_id"])
        features = []
        names = []

        def add(name, values):
            features.append(np.asarray(values, dtype=np.float32))
            names.append(name)

        enrollment = _numeric(studies["enrollment"])
        arms = _numeric(studies["number_of_arms"])
        groups = _numeric(studies["number_of_groups"])
        add("study_log_enrollment", np.log1p(np.maximum(enrollment, 0)))
        add("study_enrollment_missing", np.isnan(enrollment))
        add("study_arms", arms)
        add("study_arms_missing", np.isnan(arms))
        add("study_groups", groups)
        add("study_groups_missing", np.isnan(groups))
        for value in ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]:
            add(f"study_type_{value.lower()}", _contains(studies["study_type"], value))
        for value in ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "EARLY_PHASE1", "NA"]:
            add(f"phase_{value.lower()}", _contains(studies["phase"], value))
        for value in ["INDUSTRY", "NIH", "FED", "NETWORK", "OTHER", "UNKNOWN"]:
            add(f"source_class_{value.lower()}", _contains(studies["source_class"], value))

        for column, values in {
            "allocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
            "masking": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
            "intervention_model": ["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL", "SEQUENTIAL"],
            "primary_purpose": ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "BASIC_SCIENCE", "SCREENING", "DEVICE_FEASIBILITY"],
        }.items():
            for value in values:
                add(f"design_{column}_{value.lower()}", _contains(designs[column], value))
        add("design_present", designs["id"].notna().to_numpy(dtype=np.float32))

        minimum_age = eligibility["minimum_age"].map(_number).to_numpy(dtype=np.float32)
        maximum_age = eligibility["maximum_age"].map(_number).to_numpy(dtype=np.float32)
        add("eligibility_minimum_age", minimum_age)
        add("eligibility_minimum_age_missing", np.isnan(minimum_age))
        add("eligibility_maximum_age", maximum_age)
        add("eligibility_maximum_age_missing", np.isnan(maximum_age))
        for column, values in {
            "gender": ["ALL", "FEMALE", "MALE"],
            "healthy_volunteers": ["YES", "NO"],
            "adult": ["TRUE"],
            "child": ["TRUE"],
            "older_adult": ["TRUE"],
        }.items():
            for value in values:
                add(f"eligibility_{column}_{value.lower()}", _contains(eligibility[column], value))
        add("eligibility_criteria_length", eligibility["criteria"].fillna("").astype(str).str.len().to_numpy(dtype=np.float32))
        add("eligibility_population_length", eligibility["population"].fillna("").astype(str).str.len().to_numpy(dtype=np.float32))
        add("eligibility_present", eligibility["id"].notna().to_numpy(dtype=np.float32))

        facilities = self.tables["facilities_studies"][["nct_id", "facility_id"]].merge(
            self.tables["facilities"][["facility_id", "country", "state"]], on="facility_id", how="left"
        )
        facility_group = facilities.groupby("nct_id", sort=False)
        facility_count = facility_group.size().reindex(studies["nct_id"], fill_value=0)
        country_breadth = facility_group["country"].nunique().reindex(studies["nct_id"], fill_value=0)
        state_breadth = facility_group["state"].nunique().reindex(studies["nct_id"], fill_value=0)
        country_counts = (
            facilities.dropna(subset=["country"]).groupby(["nct_id", "country"], sort=False).size().rename("count").reset_index()
        )
        top_country_rows = country_counts.sort_values(
            ["nct_id", "count"], ascending=[True, False], kind="stable"
        ).drop_duplicates("nct_id")
        top_country_count = top_country_rows.set_index("nct_id")["count"].reindex(studies["nct_id"], fill_value=0)
        top_country = top_country_rows.set_index("nct_id")["country"].reindex(studies["nct_id"], fill_value=-1)
        countries = sorted(x for x in self.tables["facilities"]["country"].dropna().astype(str).unique())
        country_codes = {value: index for index, value in enumerate(countries)}
        study_top_country = top_country.map(country_codes).fillna(-1).to_numpy(dtype=np.int16)
        add("facility_count", facility_count.to_numpy(dtype=np.float32))
        add("facility_country_breadth", country_breadth.to_numpy(dtype=np.float32))
        add("facility_state_breadth", state_breadth.to_numpy(dtype=np.float32))
        add("facility_top_country_share", top_country_count.to_numpy(dtype=np.float32) / np.maximum(facility_count.to_numpy(dtype=np.float32), 1))
        add("facility_present", (facility_count.to_numpy() > 0).astype(np.float32))

        intervention_count = (
            self.tables["interventions_studies"].groupby("nct_id", sort=False)["intervention_id"].nunique().reindex(studies["nct_id"], fill_value=0)
        )
        add("intervention_breadth", intervention_count.to_numpy(dtype=np.float32))
        add("intervention_present", (intervention_count.to_numpy() > 0).astype(np.float32))

        matrix = np.column_stack(features).astype(np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix, names, study_top_country

    def result_features(self, cutoff: pd.Timestamp):
        features = []
        names = []

        def add(name, series):
            values = np.zeros(self.n_studies, dtype=np.float32)
            if len(series):
                indices = series.index.to_numpy(dtype=np.int64)
                values[indices] = series.to_numpy(dtype=np.float32)
            features.append(values)
            names.append(name)

        outcomes = self.tables["outcomes"]
        outcomes = outcomes[outcomes["date"] <= cutoff]
        outcomes = outcomes.assign(
            primary_flag=outcomes["outcome_type"].fillna("").astype(str).str.upper().str.contains("PRIMARY").astype(np.float32),
            text_length=outcomes["description"].fillna("").astype(str).str.len().astype(np.float32),
        )
        outcome_group = outcomes.groupby("nct_id", sort=False)
        outcome_count = outcome_group.size()
        add("result_outcome_count", outcome_count)
        add("result_primary_outcome_rate", outcome_group["primary_flag"].mean())
        add("result_outcome_text_length", outcome_group["text_length"].mean())
        add("result_outcome_present", (outcome_count > 0).astype(np.float32))

        analyses = self.tables["outcome_analyses"]
        analyses = analyses[analyses["date"] <= cutoff]
        analyses = analyses.assign(p_value_present=analyses["p_value"].notna().astype(np.float32))
        analysis_group = analyses.groupby("nct_id", sort=False)
        analysis_count = analysis_group.size()
        add("result_analysis_count", analysis_count)
        add("result_p_value_rate", analysis_group["p_value_present"].mean())
        add("result_analysis_present", (analysis_count > 0).astype(np.float32))

        events = self.tables["reported_event_totals"]
        events = events[events["date"] <= cutoff]
        event_group = events.groupby("nct_id", sort=False)
        event_count = event_group.size()
        affected = event_group["subjects_affected"].sum(min_count=1)
        risk = event_group["subjects_at_risk"].sum(min_count=1)
        add("result_adverse_event_count", event_count)
        add("result_adverse_event_rate", (affected / risk.clip(lower=1)).fillna(0))
        add("result_adverse_event_present", (event_count > 0).astype(np.float32))

        withdrawals = self.tables["drop_withdrawals"]
        withdrawals = withdrawals[withdrawals["date"] <= cutoff]
        withdrawal_group = withdrawals.groupby("nct_id", sort=False)
        withdrawal_count = withdrawal_group.size()
        add("result_withdrawal_rows", withdrawal_count)
        add("result_withdrawal_count", np.log1p(withdrawal_group["count"].sum(min_count=1).fillna(0)))
        add("result_withdrawal_present", (withdrawal_count > 0).astype(np.float32))
        return np.column_stack(features).astype(np.float32), names
