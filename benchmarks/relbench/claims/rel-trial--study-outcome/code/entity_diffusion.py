from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_mart import MODEL_NAME, MODEL_REVISION, SEED, register_artifact, within_timestamp_features


ROUTES = {
    "condition": ("conditions_studies", "condition_id", "conditions", "mesh_term", 16, 0.76),
    "intervention": ("interventions_studies", "intervention_id", "interventions", "mesh_term", 16, 0.76),
    "sponsor": ("sponsors_studies", "sponsor_id", "sponsors", "name", 8, 0.82),
    "facility": ("facilities_studies", "facility_id", "facilities", "name", 4, 0.88),
}
RESTART = 0.25
HALF_LIFE_DAYS = 1825.0
VERSION = "lane1_causal_entity_diffusion_v1"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def dense_rows(frame: pd.DataFrame, id_column: str, values: np.ndarray) -> np.ndarray:
    size = int(frame[id_column].max()) + 1
    output = np.zeros((size, values.shape[1]), dtype=np.float32)
    output[frame[id_column].to_numpy(np.int64)] = values.astype(np.float32)
    return output


def category_vectors(values: pd.Series, dimensions: int, seed: int) -> np.ndarray:
    text = values.fillna("<missing>").astype(str)
    codes, names = pd.factorize(text, sort=True)
    rng = np.random.default_rng(seed)
    lookup = normalize_rows(rng.standard_normal((len(names), dimensions)).astype(np.float32))
    return lookup[codes]


def cached_lane3_semantics(shared: Path, route: str, frame: pd.DataFrame, id_column: str) -> np.ndarray:
    components = {"condition": 10, "intervention": 10, "sponsor": 12, "facility": 8}[route]
    path = shared / "lane3_temporal_heterosage_v4" / f"entity_semantics_{route}_{components}.npy"
    values = np.load(path)
    if len(values) != len(frame):
        raise RuntimeError(f"lane-3 semantic row mismatch for {route}: {values.shape}, {len(frame)}")
    return dense_rows(frame, id_column, normalize_rows(values))


def encode_biomedical_entities(db, shared: Path, debug: bool) -> dict[str, np.ndarray]:
    root = shared / VERSION / "entity_vectors"
    root.mkdir(parents=True, exist_ok=True)
    frames = {}
    base = {}
    missing = []
    for route in ["condition", "intervention", "sponsor"]:
        _, id_column, table, text_column, _, _ = ROUTES[route]
        frame = db.table_dict[table].df[[id_column, text_column] + (["agency_class"] if route == "sponsor" else [])].copy()
        frames[route] = frame
        path = root / f"bioclinical_{route}_{MODEL_REVISION[:12]}.npy"
        if path.exists() and not debug:
            values = np.load(path)
            expected = int(frame[id_column].max()) + 1
            if values.shape != (expected, 768):
                raise RuntimeError(f"entity embedding cache mismatch for {route}: {values.shape}")
            base[route] = values.astype(np.float32)
        elif debug:
            base[route] = cached_lane3_semantics(shared, route, frame, id_column)
        else:
            missing.append(route)
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True, revision=MODEL_REVISION)
        model.max_seq_length = 128
        for route in missing:
            _, id_column, _, text_column, _, _ = ROUTES[route]
            frame = frames[route]
            started = time.time()
            encoded = model.encode(frame[text_column].fillna("").astype(str).tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            values = dense_rows(frame, id_column, encoded)
            path = root / f"bioclinical_{route}_{MODEL_REVISION[:12]}.npy"
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, values)
            os.replace(temporary, path)
            base[route] = values
            print(f"[diffusion] encoded {route} entities={len(frame)} rate={len(frame) / max(time.time() - started, 1e-6):.1f}/s", flush=True)
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    output = {}
    for route in ["condition", "intervention"]:
        output[route] = normalize_rows(base[route])
    sponsors = frames["sponsor"]
    sponsor_id = ROUTES["sponsor"][1]
    lexical = cached_lane3_semantics(shared, "sponsor", sponsors, sponsor_id)
    agency = dense_rows(sponsors, sponsor_id, category_vectors(sponsors["agency_class"], 8, SEED + 31))
    output["sponsor"] = normalize_rows(np.column_stack([math.sqrt(0.82) * normalize_rows(base["sponsor"]), math.sqrt(0.13) * lexical, math.sqrt(0.05) * agency]))
    facilities = db.table_dict["facilities"].df[["facility_id", "name", "city", "state", "country"]].copy()
    facility_name = cached_lane3_semantics(shared, "facility", facilities, "facility_id")
    city = dense_rows(facilities, "facility_id", category_vectors(facilities["city"], 8, SEED + 41))
    state = dense_rows(facilities, "facility_id", category_vectors(facilities["state"], 8, SEED + 43))
    country = dense_rows(facilities, "facility_id", category_vectors(facilities["country"], 8, SEED + 47))
    output["facility"] = normalize_rows(np.column_stack([math.sqrt(0.72) * facility_name, math.sqrt(0.14) * city, math.sqrt(0.08) * state, math.sqrt(0.06) * country]))
    if not debug:
        register_artifact(shared, "Lane 1 typed entity representations", root, "Pinned BioClinical condition/intervention/sponsor vectors and cached facility name/locality representations for exact same-type bridges.", f"{VERSION}-vectors-{MODEL_REVISION}", "Run main.py after deleting the entity_vectors directory; the pinned model and lane-3 semantics are reused.")
    return output


def entity_script_profile(db) -> dict:
    output = {}
    for route, (_, _, table, text_column, _, _) in ROUTES.items():
        values = db.table_dict[table].df[text_column].fillna("").astype(str)
        non_ascii = values.map(lambda value: any(ord(char) > 127 for char in value))
        non_latin = values.map(lambda value: any(char.isalpha() and ord(char) > 591 for char in value))
        output[route] = {"count": int(len(values)), "non_ascii_rate": float(non_ascii.mean()), "non_latin_rate": float(non_latin.mean()), "empty_rate": float(values.str.strip().eq("").mean())}
    facilities = db.table_dict["facilities"].df
    script = facilities["name"].fillna("").astype(str).map(lambda value: any(ord(char) > 127 for char in value))
    country = facilities.assign(non_ascii=script).groupby(facilities["country"].fillna("<missing>"), dropna=False).agg(count=("facility_id", "size"), non_ascii_rate=("non_ascii", "mean")).sort_values("count", ascending=False).head(20)
    output["facility_top_countries"] = [{"country": str(index), "count": int(row["count"]), "non_ascii_rate": float(row["non_ascii_rate"])} for index, row in country.iterrows()]
    return output


def relation_events(db, episodes: pd.DataFrame, route: str) -> pd.DataFrame:
    relation_table, entity_column, _, _, _, _ = ROUTES[route]
    relation = db.table_dict[relation_table].df[["nct_id", entity_column, "date"]].dropna(subset=["nct_id", entity_column, "date"]).copy()
    relation = relation.drop_duplicates(["nct_id", entity_column], keep="first")
    joined = relation.merge(episodes[["nct_id", "report_date", "success"]], on="nct_id", how="inner", validate="many_to_one")
    joined = joined[joined["date"].le(joined["report_date"])].copy()
    joined = joined.rename(columns={entity_column: "entity"})
    joined["entity"] = joined["entity"].astype(np.int64)
    return joined[["nct_id", "entity", "report_date", "success"]]


def seed_memberships(db, seeds: pd.DataFrame, route: str) -> pd.DataFrame:
    relation_table, entity_column, _, _, _, _ = ROUTES[route]
    relation = db.table_dict[relation_table].df[["nct_id", entity_column, "date"]].dropna(subset=["nct_id", entity_column, "date"]).copy()
    relation = relation.drop_duplicates(["nct_id", entity_column], keep="first")
    joined = seeds[["row_id", "nct_id", "timestamp"]].merge(relation, on="nct_id", how="inner")
    joined = joined[joined["date"].le(joined["timestamp"])].drop_duplicates(["row_id", entity_column])
    joined = joined.rename(columns={entity_column: "entity"})
    joined["entity"] = joined["entity"].astype(np.int64)
    return joined[["row_id", "nct_id", "timestamp", "entity"]]


def facility_countries(db, size: int) -> np.ndarray:
    facilities = db.table_dict["facilities"].df
    codes, _ = pd.factorize(facilities["country"].fillna("<missing>").astype(str), sort=True)
    output = np.full(size, -1, dtype=np.int32)
    output[facilities["facility_id"].to_numpy(np.int64)] = codes.astype(np.int32)
    return output


def exact_neighbors(vectors: np.ndarray, query_ids: np.ndarray, candidate_ids: np.ndarray, k: int, threshold: float, countries: np.ndarray | None, debug: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch

    size = len(vectors)
    neighbor_ids = np.full((size, k), -1, dtype=np.int64)
    neighbor_scores = np.zeros((size, k), dtype=np.float32)
    query_ids = np.unique(query_ids.astype(np.int64))
    candidate_ids = np.unique(candidate_ids.astype(np.int64))
    if debug and len(candidate_ids) > 5000:
        positions = np.linspace(0, len(candidate_ids) - 1, 5000).astype(np.int64)
        candidate_ids = candidate_ids[positions]
    valid_candidates = np.linalg.norm(vectors[candidate_ids], axis=1) > 0
    candidate_ids = candidate_ids[valid_candidates]
    valid_queries = np.linalg.norm(vectors[query_ids], axis=1) > 0
    query_ids = query_ids[valid_queries]
    if not len(query_ids) or not len(candidate_ids):
        return neighbor_ids, neighbor_scores
    groups = [None] if countries is None else np.unique(countries[query_ids])
    device = torch.device("cuda")
    for group in groups:
        queries = query_ids if group is None else query_ids[countries[query_ids] == group]
        candidates = candidate_ids if group is None else candidate_ids[countries[candidate_ids] == group]
        if not len(queries) or not len(candidates):
            continue
        candidate_tensor = torch.as_tensor(vectors[candidates], dtype=torch.float32, device=device)
        take = min(k, len(candidates))
        for start in range(0, len(queries), 2048):
            rows = queries[start:start + 2048]
            query_tensor = torch.as_tensor(vectors[rows], dtype=torch.float32, device=device)
            similarity = query_tensor @ candidate_tensor.T
            self_mask = rows[:, None] == candidates[None, :]
            if self_mask.any():
                similarity[torch.as_tensor(self_mask, device=device)] = -2.0
            scores, positions = torch.topk(similarity, k=take, dim=1)
            scores = scores.cpu().numpy()
            ids = candidates[positions.cpu().numpy()]
            valid = scores >= threshold
            neighbor_ids[rows, :take] = np.where(valid, ids, -1)
            neighbor_scores[rows, :take] = np.where(valid, scores, 0.0)
    return neighbor_ids, neighbor_scores


def source_statistics(events: pd.DataFrame, timestamp: pd.Timestamp, size: int) -> dict[str, dict[str, np.ndarray]]:
    current = events[events["report_date"].le(timestamp)].copy()
    entity = current["entity"].to_numpy(np.int64)
    success = current["success"].to_numpy(np.float64)
    lag = (timestamp - current["report_date"]).dt.days.to_numpy(np.float64)
    all_weight = np.ones(len(current), dtype=np.float64)
    half_weight = np.exp(-math.log(2.0) * lag / HALF_LIFE_DAYS)
    quantiles = current.assign(lag=lag).groupby("entity")["lag"].quantile([0.25, 0.5, 0.75]).unstack()
    output = {}
    for view, weight in [("all", all_weight), ("half5y", half_weight)]:
        mass = np.bincount(entity, weights=weight, minlength=size).astype(np.float64)
        number = np.bincount(entity, weights=weight * success, minlength=size).astype(np.float64)
        lag_number = np.bincount(entity, weights=weight * lag, minlength=size).astype(np.float64)
        fields = {"mass": mass, "number": number, "lag_number": lag_number}
        for value, name in [(0.25, "lag_q25"), (0.5, "lag_q50"), (0.75, "lag_q75")]:
            array = np.zeros(size, dtype=np.float64)
            if value in quantiles:
                ids = quantiles.index.to_numpy(np.int64)
                array[ids] = quantiles[value].to_numpy(np.float64)
            fields[name] = array
        output[view] = fields
    output["support"] = np.bincount(entity, minlength=size).astype(np.float64)
    return output


def propagated_state(stats: dict[str, np.ndarray], neighbor_ids: np.ndarray, neighbor_scores: np.ndarray, threshold: float, iterations: int) -> dict[str, np.ndarray]:
    base_mass = stats["mass"]
    base_number = stats["number"]
    base_lag = stats["lag_number"]
    valid = (neighbor_ids >= 0) & (neighbor_scores >= threshold)
    safe_ids = np.where(valid, neighbor_ids, 0)
    scaled = np.where(valid, np.exp((neighbor_scores - np.max(np.where(valid, neighbor_scores, -20.0), axis=1, keepdims=True)) / 0.05), 0.0)
    weight_sum = scaled.sum(axis=1, keepdims=True)
    weights = np.divide(scaled, weight_sum, out=np.zeros_like(scaled), where=weight_sum > 0)
    mass = base_mass.copy()
    number = base_number.copy()
    lag_number = base_lag.copy()
    quantile_values = {name: stats[name].copy() for name in ["lag_q25", "lag_q50", "lag_q75"]}
    effective = base_mass.copy()
    for _ in range(iterations):
        has_neighbor = weight_sum[:, 0] > 0
        neighbor_mass = np.sum(weights * mass[safe_ids], axis=1)
        neighbor_number = np.sum(weights * number[safe_ids], axis=1)
        neighbor_lag = np.sum(weights * lag_number[safe_ids], axis=1)
        neighbor_effective = np.sum(weights * effective[safe_ids], axis=1)
        mass = RESTART * base_mass + (1.0 - RESTART) * np.where(has_neighbor, neighbor_mass, 0.0)
        number = RESTART * base_number + (1.0 - RESTART) * np.where(has_neighbor, neighbor_number, 0.0)
        lag_number = RESTART * base_lag + (1.0 - RESTART) * np.where(has_neighbor, neighbor_lag, 0.0)
        effective = RESTART * base_mass + (1.0 - RESTART) * np.where(has_neighbor, neighbor_effective, 0.0)
        for name in quantile_values:
            neighbor_value = np.sum(weights * quantile_values[name][safe_ids], axis=1)
            quantile_values[name] = RESTART * stats[name] + (1.0 - RESTART) * np.where(has_neighbor, neighbor_value, 0.0)
    output = {"mass": mass, "number": number, "lag_number": lag_number, "effective": effective, "exact_mass": RESTART * base_mass, "bridge_mass": np.maximum(mass - RESTART * base_mass, 0.0), "base_mass": base_mass}
    output.update(quantile_values)
    return output


def aggregate_route(rows: np.ndarray, members: pd.DataFrame, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    positions = {int(row): index for index, row in enumerate(rows)}
    result = {name: np.zeros(len(rows), dtype=np.float64) for name in ["success", "total_mass", "effective_support", "exact_mass", "bridge_mass", "lag_mean", "lag_q25", "lag_q50", "lag_q75", "entropy", "top_source_gap", "exact_cold_start", "cold_start", "bridge_only", "entity_count"]}
    if members.empty:
        result["success"].fill(0.5)
        result["exact_cold_start"].fill(1.0)
        result["cold_start"].fill(1.0)
        return result
    frame = members[["row_id", "entity"]].copy()
    entity = frame["entity"].to_numpy(np.int64)
    for name in ["mass", "number", "effective", "exact_mass", "bridge_mass", "lag_number", "lag_q25", "lag_q50", "lag_q75", "base_mass"]:
        frame[name] = state[name][entity]
    grouped = frame.groupby("row_id", sort=False)
    count = grouped.size()
    mass = grouped["mass"].sum()
    number = grouped["number"].sum()
    exact = grouped["exact_mass"].sum()
    bridge = grouped["bridge_mass"].sum()
    base = grouped["base_mass"].sum()
    index = np.asarray([positions[int(value)] for value in count.index], dtype=np.int64)
    denominator = count.to_numpy(np.float64)
    mass_sum = mass.to_numpy(np.float64)
    result["entity_count"][index] = denominator
    result["total_mass"][index] = mass_sum / denominator
    result["effective_support"][index] = grouped["effective"].sum().to_numpy(np.float64) / denominator
    result["exact_mass"][index] = exact.to_numpy(np.float64) / denominator
    result["bridge_mass"][index] = bridge.to_numpy(np.float64) / denominator
    result["success"][index] = np.divide(number.to_numpy(np.float64), mass_sum, out=np.full(len(index), 0.5), where=mass_sum > 0)
    result["lag_mean"][index] = np.divide(grouped["lag_number"].sum().to_numpy(np.float64), mass_sum, out=np.zeros(len(index)), where=mass_sum > 0)
    for name in ["lag_q25", "lag_q50", "lag_q75"]:
        weighted = (frame[name] * frame["mass"]).groupby(frame["row_id"]).sum().reindex(count.index).to_numpy(np.float64)
        result[name][index] = np.divide(weighted, mass_sum, out=np.zeros(len(index)), where=mass_sum > 0)
    shares = np.divide(frame["mass"].to_numpy(np.float64), frame["row_id"].map(mass).to_numpy(np.float64), out=np.zeros(len(frame)), where=frame["row_id"].map(mass).to_numpy(np.float64) > 0)
    entropy_terms = np.where(shares > 0, -shares * np.log(np.clip(shares, 1e-12, 1.0)), 0.0)
    entropy = pd.Series(entropy_terms, index=frame.index).groupby(frame["row_id"]).sum().reindex(count.index).to_numpy(np.float64)
    normalizer = np.log(np.maximum(denominator, 2.0))
    result["entropy"][index] = entropy / normalizer
    ordered = frame[["row_id", "mass"]].sort_values(["row_id", "mass"], ascending=[True, False])
    ordered["rank"] = ordered.groupby("row_id").cumcount()
    top = ordered[ordered["rank"].lt(2)].pivot(index="row_id", columns="rank", values="mass").reindex(count.index).fillna(0.0)
    first = top.get(0, pd.Series(0.0, index=count.index)).to_numpy(np.float64)
    second = top.get(1, pd.Series(0.0, index=count.index)).to_numpy(np.float64)
    result["top_source_gap"][index] = np.divide(first - second, mass_sum, out=np.zeros(len(index)), where=mass_sum > 0)
    result["exact_cold_start"][index] = (base.to_numpy(np.float64) == 0).astype(np.float64)
    result["cold_start"][index] = (mass_sum == 0).astype(np.float64)
    result["bridge_only"][index] = ((base.to_numpy(np.float64) == 0) & (mass_sum > 0)).astype(np.float64)
    missing = np.setdiff1d(np.arange(len(rows)), index, assume_unique=False)
    result["success"][missing] = 0.5
    result["exact_cold_start"][missing] = 1.0
    result["cold_start"][missing] = 1.0
    return result


def add_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    variants = sorted({match.group(1) for column in frame for match in [re.match(r"diff_(loose|strict)_", column)] if match})
    depths = sorted({int(match.group(1)) for column in frame for match in [re.search(r"_d([0-9]+)_success$", column)] if match})
    for variant in variants:
        for view in ["all", "half5y"]:
            for depth in depths:
                prefix = f"diff_{variant}"
                success_columns = [f"{prefix}_{route}_{view}_d{depth}_success" for route in ROUTES]
                mass_columns = [f"{prefix}_{route}_{view}_d{depth}_total_mass" for route in ROUTES]
                cold_columns = [f"{prefix}_{route}_{view}_d{depth}_cold_start" for route in ROUTES]
                bridge_columns = [f"{prefix}_{route}_{view}_d{depth}_bridge_only" for route in ROUTES]
                if not all(column in frame for column in success_columns):
                    continue
                success = frame[success_columns].to_numpy(np.float64)
                mass = frame[mass_columns].to_numpy(np.float64)
                available = 1.0 - frame[cold_columns].to_numpy(np.float64)
                key = f"{prefix}_routes_{view}_d{depth}"
                output[f"{key}_success_mean"] = np.divide((success * available).sum(axis=1), available.sum(axis=1), out=np.full(len(frame), 0.5), where=available.sum(axis=1) > 0)
                output[f"{key}_success_std"] = np.nanstd(np.where(available > 0, success, np.nan), axis=1)
                output[f"{key}_success_range"] = np.nanmax(np.where(available > 0, success, np.nan), axis=1) - np.nanmin(np.where(available > 0, success, np.nan), axis=1)
                output[f"{key}_available_count"] = available.sum(axis=1)
                output[f"{key}_bridge_only_count"] = frame[bridge_columns].sum(axis=1)
                shares = np.divide(mass, mass.sum(axis=1, keepdims=True), out=np.zeros_like(mass), where=mass.sum(axis=1, keepdims=True) > 0)
                output[f"{key}_mass_entropy"] = np.where(shares > 0, -shares * np.log(np.clip(shares, 1e-12, 1.0)), 0.0).sum(axis=1) / math.log(len(ROUTES))
    return output.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def bridge_audit(db, memberships: dict[str, pd.DataFrame], neighbors: dict[str, tuple[np.ndarray, np.ndarray]], timestamp: pd.Timestamp) -> dict:
    rng = np.random.default_rng(SEED)
    output = {}
    for route, member in memberships.items():
        _, entity_column, table, text_column, _, threshold = ROUTES[route]
        frame = db.table_dict[table].df.set_index(entity_column)
        eligible = member[(member["timestamp"].dt.year.le(2019)) & member["timestamp"].le(timestamp)]["entity"].drop_duplicates().to_numpy(np.int64)
        ids, scores = neighbors[route]
        pairs = []
        for source in eligible:
            valid = np.flatnonzero((ids[source] >= 0) & (scores[source] >= threshold))
            if len(valid):
                pairs.append((int(source), int(ids[source, valid[0]]), float(scores[source, valid[0]])))
        if len(pairs) > 64:
            pairs = [pairs[index] for index in rng.choice(len(pairs), 64, replace=False)]
        lexical = []
        locality = []
        agreement = []
        for source, target, score in pairs:
            left = str(frame.at[source, text_column]) if source in frame.index else ""
            right = str(frame.at[target, text_column]) if target in frame.index else ""
            a = set(re.findall(r"[a-z0-9]+", left.lower()))
            b = set(re.findall(r"[a-z0-9]+", right.lower()))
            lexical.append(len(a & b) / max(1, len(a | b)))
            agreement.append(any(ord(char) > 127 for char in left) == any(ord(char) > 127 for char in right))
            if route == "facility":
                locality.append(float(str(frame.at[source, "city"]).lower() == str(frame.at[target, "city"]).lower() or str(frame.at[source, "state"]).lower() == str(frame.at[target, "state"]).lower()))
            elif route == "sponsor":
                locality.append(float(str(frame.at[source, "agency_class"]) == str(frame.at[target, "agency_class"])))
        output[route] = {"sample_count": len(pairs), "mean_cosine": float(np.mean([pair[2] for pair in pairs])) if pairs else 0.0, "mean_token_jaccard": float(np.mean(lexical)) if lexical else 0.0, "script_agreement_rate": float(np.mean(agreement)) if agreement else 0.0, "agency_or_locality_agreement_rate": float(np.mean(locality)) if locality else None, "enabled": bool(len(pairs) >= 8)}
    return output


def build_diffusion_features(db, seeds: pd.DataFrame, episodes: pd.DataFrame, shared: Path, debug: bool) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    vectors = encode_biomedical_entities(db, shared, debug)
    events = {route: relation_events(db, episodes, route) for route in ROUTES}
    memberships = {route: seed_memberships(db, seeds, route) for route in ROUTES}
    profile = entity_script_profile(db)
    feature_values: dict[str, np.ndarray] = {}
    audit_neighbors = {}
    depths = [1] if debug else [2, 3]
    variants = {"loose": 0.0} if debug else {"loose": 0.0, "strict": 0.04}
    facility_country = facility_countries(db, len(vectors["facility"]))
    timestamps = sorted(seeds["timestamp"].unique())
    for timestamp_value in timestamps:
        timestamp = pd.Timestamp(timestamp_value)
        rows = seeds.loc[seeds["timestamp"].eq(timestamp), "row_id"].to_numpy(np.int64)
        for route, (_, _, _, _, k, threshold) in ROUTES.items():
            current_members = memberships[route][memberships[route]["timestamp"].eq(timestamp)]
            stats = source_statistics(events[route], timestamp, len(vectors[route]))
            destinations = np.flatnonzero(stats["support"] >= 5)
            query_ids = np.unique(current_members["entity"].to_numpy(np.int64)) if len(current_members) else np.empty(0, dtype=np.int64)
            search_ids = np.union1d(query_ids, destinations)
            countries = facility_country if route == "facility" else None
            neighbor_ids, neighbor_scores = exact_neighbors(vectors[route], search_ids, destinations, k, threshold, countries, debug)
            if timestamp.year == 2019:
                audit_neighbors[route] = (neighbor_ids, neighbor_scores)
            for variant, increment in variants.items():
                active_threshold = min(0.99, threshold + increment)
                for view in ["all", "half5y"]:
                    for depth in depths:
                        state = propagated_state(stats[view], neighbor_ids, neighbor_scores, active_threshold, depth)
                        aggregate = aggregate_route(rows, current_members, state)
                        for name, values in aggregate.items():
                            column = f"diff_{variant}_{route}_{view}_d{depth}_{name}"
                            if column not in feature_values:
                                feature_values[column] = np.full(len(seeds), np.nan, dtype=np.float32)
                            feature_values[column][rows] = values.astype(np.float32)
            print(f"[diffusion] origin={timestamp.date()} route={route} queries={len(query_ids)} destinations={len(destinations)}", flush=True)
    frame = add_disagreement(pd.DataFrame(feature_values, index=seeds["row_id"]))
    preferred = [column for column in frame if column.endswith("_success") or column.endswith("_total_mass") or column.endswith("_effective_support")]
    normalized = within_timestamp_features(seeds, frame, preferred)
    normalized = normalized.drop(columns=["within_small_cohort"], errors="ignore")
    frame = frame.join(normalized).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    diagnostics = {"version": VERSION, "debug": debug, "elapsed_seconds": time.time() - started, "shape": list(frame.shape), "script_profile": profile, "bridge_audit": bridge_audit(db, memberships, audit_neighbors, pd.Timestamp("2019-01-01")) if audit_neighbors else {}}
    return frame, diagnostics


def load_or_build_diffusion(db, seeds: pd.DataFrame, episodes: pd.DataFrame, shared: Path, debug: bool) -> tuple[pd.DataFrame, dict]:
    root = shared / VERSION
    root.mkdir(parents=True, exist_ok=True)
    suffix = "debug" if debug else "full"
    feature_path = root / f"diffusion_features_{suffix}.pkl"
    diagnostics_path = root / f"diagnostics_{suffix}.json"
    if feature_path.exists():
        frame = pd.read_pickle(feature_path)
        diagnostics = json.loads(diagnostics_path.read_text())
        if len(frame) != len(seeds):
            raise RuntimeError(f"diffusion cache row mismatch: {frame.shape}, {len(seeds)}")
        print(f"[diffusion] loaded cache {frame.shape}", flush=True)
        return frame, diagnostics
    frame, diagnostics = build_diffusion_features(db, seeds, episodes, shared, debug)
    temporary = feature_path.with_suffix(".tmp.pkl")
    frame.to_pickle(temporary)
    os.replace(temporary, feature_path)
    temporary_json = diagnostics_path.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(diagnostics, indent=2) + "\n")
    os.replace(temporary_json, diagnostics_path)
    register_artifact(shared, f"Lane 1 causal entity diffusion {suffix}", root, "Annual same-type semantic bridge diffusion features with exact destination support, causal report censoring, restart, route availability, and cohort transforms.", f"{VERSION}-{suffix}", "Run main.py after deleting the matching diffusion feature cache; entity vectors are reused.")
    return frame, diagnostics
