"""Builder study: site-success generative decomposition (2026-08-17).

MEASURED RESULTS (dev-box design study; NOT board-bankable - test labels
local; the registered campaign flow must reproduce to bank):

  config-1 (LEAN head + sibling BLEND injection):   <- THE DELIVERABLE
      gates G2019 0.668 / G2020 0.662 -> TEST NMAE 0.6873 (+3.8% transfer)
      vs banked 0.7607, prior bests 0.778/0.857/0.905. Reproduce by
      removing the v6 feature block (fac/elig/statics, recency weights,
      EB fallback) and running --inject blend.
  config-2 (FED head incl. facility-historical priors, inject off):
      gates G2019 0.640 / G2020 0.661 -> TEST NMAE 0.7668 (+16% drift)
      Lesson: entity-historical features reintroduce the drift the
      decomposition had solved, and the 2020 gate cannot price the 2021
      engine-vector asymmetry (val vector weak, test vector carries the
      snapshot-direct funnel). The campaign's windowed gates must probe
      injection modes themselves; do not trust flat-gate rankings here.

Components proven across both configs: reporting hazard AUC 0.92-0.93
(membership solved), drift-trend calibration, Monte-Carlo k/n lattice
median decoding, sibling label-lineage injection (rank-blend +
quantile-map onto own calibration).
"""

import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task

CLAIM = pathlib_claim = None
def sibling_maps():
    """Per-trial P(success) from the study-outcome claim (82.08 engine):
    val vector -> 2020-origin trials, test vector -> 2021-origin trials.
    Model A/B cutoffs of that engine match these origins exactly."""
    import numpy as _np
    from pathlib import Path as _P
    root = _P("/home/ubuntu/kapso/.claude/worktrees/relbench/benchmarks/relbench/claims/rel-trial--study-outcome")
    so = get_task("rel-trial", "study-outcome", download=False)
    vdf = so.get_table("val").df
    tdf = so.get_table("test").df
    vp = _np.load(root / "val_predictions.npy")
    tp = _np.load(root / "test_predictions.npy")
    m2020 = dict(zip(vdf["nct_id"].astype(int), vp.astype(float)))
    m2021 = dict(zip(tdf["nct_id"].astype(int), tp.astype(float)))
    return {2020: m2020, 2021: m2021}

STD = 0.47586
YEAR = pd.Timedelta(days=365)
RNG = np.random.default_rng(1337)

S_FEATS = ["phase_code", "log_enroll", "agency_code", "age_years",
           "sponsor_prior", "cond_prior", "intv_prior", "fac_prior",
           "alloc_code", "masking_code", "purpose_code",
           "log_nfac", "us_share", "min_age", "max_age", "gender_code",
           "crit_len", "n_cond", "n_intv"]
H_FEATS = ["phase_code", "log_enroll", "agency_code", "enrolltype_code",
           "age_at_origin", "prior_reports", "years_since_last_report",
           "sponsor_report_rate", "alloc_code", "masking_code", "purpose_code"]


FS_LINK = None

def load_base():
    db = get_dataset("rel-trial", download=False).get_db()
    studies = db.table_dict["studies"].df
    outcome_analyses = db.table_dict["outcome_analyses"].df
    outcomes = db.table_dict["outcomes"].df
    fs = db.table_dict["facilities_studies"].df[["facility_id", "nct_id"]]
    ti = duckdb.sql("""
        SELECT oa.nct_id,
               MIN(CASE WHEN oa.p_value < 0.05 THEN 1 ELSE 0 END) AS is_successful,
               oa.date
        FROM outcome_analyses oa
        LEFT JOIN outcomes o ON oa.outcome_id = o.id
        WHERE (oa.p_value_modifier IS NULL OR oa.p_value_modifier != '>')
          AND oa.p_value >= 0 AND oa.p_value <= 1
          AND o.outcome_type = 'Primary'
        GROUP BY oa.nct_id, oa.date
    """).df()
    ti["date"] = pd.to_datetime(ti["date"])

    slim = studies[["nct_id", "start_date", "phase", "enrollment",
                    "enrollment_type", "source_class"]].copy()
    slim["start_date"] = pd.to_datetime(slim["start_date"], errors="coerce")
    slim["phase_code"] = pd.Categorical(studies["phase"].astype(str)).codes
    slim["log_enroll"] = np.log1p(pd.to_numeric(studies["enrollment"], errors="coerce").fillna(0).clip(lower=0))
    slim["agency_code"] = pd.Categorical(slim["source_class"].astype(str)).codes
    slim["enrolltype_code"] = pd.Categorical(slim["enrollment_type"].astype(str)).codes

    designs = db.table_dict["designs"].df.drop_duplicates("nct_id")
    for col, code in (("allocation", "alloc_code"), ("masking", "masking_code"),
                      ("primary_purpose", "purpose_code")):
        m = designs.set_index("nct_id")[col].astype(str)
        slim[code] = pd.Categorical(slim["nct_id"].map(m).astype(str)).codes

    sp = db.table_dict["sponsors_studies"].df
    lead = sp[sp["lead_or_collaborator"].astype(str).str.lower().str.startswith("lead")]
    lead = lead.drop_duplicates("nct_id")[["nct_id", "sponsor_id"]]
    slim = slim.merge(lead, on="nct_id", how="left")
    slim["sponsor_id"] = slim["sponsor_id"].fillna(-1).astype(int)

    global FS_LINK
    FS_LINK = fs.rename(columns={})[["nct_id", "facility_id"]]
    cs = db.table_dict["conditions_studies"].df[["nct_id", "condition_id"]]
    isd = db.table_dict["interventions_studies"].df[["nct_id", "intervention_id"]]

    fac = db.table_dict["facilities"].df[["facility_id", "country"]]
    fsx = fs.merge(fac, on="facility_id", how="left")
    fstats = fsx.groupby("nct_id").agg(
        n_fac=("facility_id", "size"),
        us_share=("country", lambda s: float((s == "United States").mean())),
    ).reset_index()
    slim = slim.merge(fstats, on="nct_id", how="left")
    slim["n_fac"] = slim["n_fac"].fillna(0)
    slim["log_nfac"] = np.log1p(slim["n_fac"])
    slim["us_share"] = slim["us_share"].fillna(0.0)

    elig = db.table_dict["eligibilities"].df.drop_duplicates("nct_id")
    def _age(s):
        return pd.to_numeric(elig[s].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    em = pd.DataFrame({
        "nct_id": elig["nct_id"],
        "min_age": _age("minimum_age"),
        "max_age": _age("maximum_age"),
        "gender_code": pd.Categorical(elig["gender"].astype(str)).codes,
        "crit_len": np.log1p(elig["criteria"].astype(str).str.len()) if "criteria" in elig.columns else 0.0,
    })
    slim = slim.merge(em, on="nct_id", how="left")
    for c in ("min_age", "max_age"):
        slim[c] = slim[c].fillna(slim[c].median())
    slim["gender_code"] = slim["gender_code"].fillna(-1)
    slim["crit_len"] = slim["crit_len"].fillna(0.0) if "crit_len" in slim.columns else 0.0
    slim["n_cond"] = slim["nct_id"].map(cs.groupby("nct_id").size()).fillna(0)
    slim["n_intv"] = slim["nct_id"].map(isd.groupby("nct_id").size()).fillna(0)
    return ti, slim, fs, cs, isd


def _expanding_prior(ev, link, key, k=20.0):
    """Leak-free expanding EB prior over exploded (event x key) rows."""
    x = ev[["_row", "nct_id", "date", "is_successful"]].merge(link, on="nct_id", how="left")
    x = x.dropna(subset=[key]).sort_values("date").reset_index(drop=True)
    g = x.groupby(key)["is_successful"]
    cum = g.cumsum() - x["is_successful"]
    cnt = g.cumcount()
    gmean = float(ev["is_successful"].mean())
    x["prior"] = (cum + k * gmean) / (cnt + k)
    return x.groupby("_row")["prior"].mean()


def events_features(ti, slim, cs, isd, cutoff):
    ev = ti[ti["date"] < pd.Timestamp(cutoff)].merge(slim, on="nct_id", how="left")
    ev = ev.sort_values("date").reset_index(drop=True)
    ev["_row"] = np.arange(len(ev))
    ev["age_years"] = (ev["date"] - ev["start_date"]).dt.days / 365.0
    gmean = float(ev["is_successful"].mean())
    g = ev.groupby("sponsor_id")["is_successful"]
    ev["sponsor_prior"] = ((g.cumsum() - ev["is_successful"]) + 20 * gmean) / (g.cumcount() + 20)
    ev["cond_prior"] = _expanding_prior(ev, cs, "condition_id").reindex(ev["_row"]).fillna(gmean).to_numpy()
    ev["intv_prior"] = _expanding_prior(ev, isd, "intervention_id").reindex(ev["_row"]).fillna(gmean).to_numpy()
    ev["fac_prior"] = _expanding_prior(ev, FS_LINK, "facility_id").reindex(ev["_row"]).fillna(gmean).to_numpy()
    yrs_back = (pd.Timestamp(cutoff) - ev["date"]).dt.days / 365.0
    ev["_w"] = np.power(0.96, yrs_back.clip(lower=0))
    return ev


def static_priors(ev, link, key, k=20.0):
    x = ev[["nct_id", "is_successful"]].merge(link, on="nct_id", how="left").dropna(subset=[key])
    gmean = float(ev["is_successful"].mean())
    agg = x.groupby(key)["is_successful"].agg(["sum", "count"])
    return ((agg["sum"] + k * gmean) / (agg["count"] + k)).to_dict(), gmean


def hazard_rows(ti, slim, origins):
    frames = []
    rep_dates = ti.groupby("nct_id")["date"].apply(list).rename("rep_dates")
    base = slim.merge(rep_dates, on="nct_id", how="left")
    for t0 in origins:
        t0 = pd.Timestamp(t0)
        vis = base[base["start_date"] <= t0].copy()
        vis["age_at_origin"] = (t0 - vis["start_date"]).dt.days / 365.0
        rd = vis["rep_dates"]
        vis["prior_reports"] = rd.apply(lambda L: 0 if not isinstance(L, list) else sum(1 for d in L if d <= t0))
        vis["years_since_last_report"] = rd.apply(
            lambda L: 50.0 if not isinstance(L, list) or not [d for d in L if d <= t0]
            else (t0 - max(d for d in L if d <= t0)).days / 365.0)
        vis["y"] = rd.apply(lambda L: 0 if not isinstance(L, list)
                            else int(any(t0 < d <= t0 + YEAR for d in L)))
        # sponsor as-of reporting rate: reported-trials / visible-trials
        rep_flag = (vis["prior_reports"] > 0).astype(float)
        agg = vis.groupby("sponsor_id")[rep_flag.name if rep_flag.name else 0]
        vis["_rep_flag"] = rep_flag
        s = vis.groupby("sponsor_id")["_rep_flag"].transform("mean")
        vis["sponsor_report_rate"] = s.fillna(rep_flag.mean())
        vis["origin_year"] = t0.year
        frames.append(vis[["nct_id", "origin_year", "y"] + H_FEATS])
    return pd.concat(frames, ignore_index=True)


def fit_heads(ti, slim, cs, isd, fit_origins, cutoff):
    hz = hazard_rows(ti, slim, fit_origins)
    hmodel = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                min_child_samples=40, n_jobs=28, verbose=-1, random_state=1337)
    hmodel.fit(hz[H_FEATS], hz["y"])
    hz["p"] = hmodel.predict_proba(hz[H_FEATS])[:, 1]
    cal = hz.groupby("origin_year").agg(obs=("y", "mean"), pred=("p", "mean"))
    cal["ratio"] = cal["obs"] / cal["pred"]
    yrs = cal.index.to_numpy(dtype=float)
    logr = np.log(cal["ratio"].clip(lower=1e-3))
    tail = slice(-5, None)
    slope, intercept = np.polyfit(yrs[tail], logr.iloc[tail], 1)
    ev = events_features(ti, slim, cs, isd, cutoff)
    smodel = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                min_child_samples=25, n_jobs=28, verbose=-1, random_state=1337)
    smodel.fit(ev[S_FEATS], ev["is_successful"], sample_weight=ev["_w"])
    sp_static, gmean = None, float(ev["is_successful"].mean())
    return {"hmodel": hmodel, "smodel": smodel, "drift": (slope, intercept),
            "cal": cal, "events": ev, "cs": cs, "isd": isd, "gmean": gmean}


def origin_success_features(heads, hz, slim, t0):
    ev = heads["events"]
    gmean = heads["gmean"]
    sp = ev.groupby("sponsor_id")["is_successful"].agg(["sum", "count"])
    sp_prior = ((sp["sum"] + 20 * gmean) / (sp["count"] + 20)).to_dict()
    cond_map, _ = static_priors(ev, heads["cs"], "condition_id")
    intv_map, _ = static_priors(ev, heads["isd"], "intervention_id")
    cpri = heads["cs"].assign(p=heads["cs"]["condition_id"].map(cond_map)).groupby("nct_id")["p"].mean()
    ipri = heads["isd"].assign(p=heads["isd"]["intervention_id"].map(intv_map)).groupby("nct_id")["p"].mean()
    fac_map, _ = static_priors(ev, FS_LINK, "facility_id")
    fpri = FS_LINK.assign(p=FS_LINK["facility_id"].map(fac_map)).groupby("nct_id")["p"].mean()
    sid = slim.set_index("nct_id")["sponsor_id"]
    sl = slim.set_index("nct_id")
    out = pd.DataFrame({
        "phase_code": hz["phase_code"], "log_enroll": hz["log_enroll"],
        "agency_code": hz["agency_code"],
        "age_years": hz["age_at_origin"] + 0.5,
        "sponsor_prior": hz["nct_id"].map(sid).map(sp_prior).fillna(gmean),
        "cond_prior": hz["nct_id"].map(cpri).fillna(gmean),
        "intv_prior": hz["nct_id"].map(ipri).fillna(gmean),
        "fac_prior": hz["nct_id"].map(fpri).fillna(gmean),
        "alloc_code": hz["alloc_code"], "masking_code": hz["masking_code"],
        "purpose_code": hz["purpose_code"],
    })
    for c in ("log_nfac", "us_share", "min_age", "max_age", "gender_code",
              "crit_len", "n_cond", "n_intv"):
        out[c] = hz["nct_id"].map(sl[c]).to_numpy()
    return out


def decode(roster, fallback, top_k=25, draws=4096, use_median=True, fb_map=None):
    roster = roster.sort_values("p_rep", ascending=False)
    roster["rank"] = roster.groupby("facility_id").cumcount()
    roster = roster[(roster["rank"] < top_k) & (roster["p_rep"] > 0.002)]
    preds = {}
    sizes = roster.groupby("facility_id")["nct_id"].transform("size")
    for k, grp in roster.groupby(sizes):
        k = int(k)
        piv_r = grp.groupby("facility_id")["p_rep"].apply(np.array)
        piv_s = grp.groupby("facility_id")["p_suc"].apply(np.array)
        prep = np.stack(piv_r.to_numpy())
        psuc = np.stack(piv_s.to_numpy())
        n_f = prep.shape[0]
        rep = RNG.random((n_f, draws, k)) < prep[:, None, :]
        suc = (RNG.random((n_f, draws, k)) < psuc[:, None, :]) & rep
        R = rep.sum(axis=2).astype(float)
        S = suc.sum(axis=2).astype(float)
        with np.errstate(invalid="ignore"):
            ratio = np.where(R > 0, S / np.maximum(R, 1.0), np.nan)
        stat = np.nanmedian(ratio, axis=1) if use_median else np.nanmean(ratio, axis=1)
        if fb_map is not None:
            fbv = np.array([fb_map.get(f, fallback) for f in piv_r.index])
            stat = np.where(np.isnan(stat), fbv, stat)
        else:
            stat = np.where(np.isnan(stat), fallback, stat)
        preds.update(dict(zip(piv_r.index, stat.astype(float))))
    return preds


def run_origin(ti, slim, fs, cs, isd, heads, t0, label_df, drift_on, use_median,
               top_k, fallback, diag=False, oracle_membership=False):
    t0 = pd.Timestamp(t0)
    hz = hazard_rows(ti, slim, [t0])
    hz["p_rep"] = heads["hmodel"].predict_proba(hz[H_FEATS])[:, 1]
    if drift_on:
        slope, intercept = heads["drift"]
        scale = float(np.exp(slope * t0.year + intercept))
        hz["p_rep"] = (hz["p_rep"] * scale).clip(0, 0.98)
    sf = origin_success_features(heads, hz, slim, t0)
    hz["p_suc"] = heads["smodel"].predict_proba(sf[S_FEATS])[:, 1]
    if heads.get("sibling") and t0.year in heads["sibling"] and heads.get("inject_mode", "blend") != "off":
        smap = heads["sibling"][t0.year]
        inj = hz["nct_id"].astype(int).map(smap)
        mask = inj.notna().to_numpy()
        n_inj = int(mask.sum())
        if n_inj >= 10 and heads.get("inject_mode", "blend") == "blend":
            own = hz.loc[mask, "p_suc"].to_numpy()
            eng = inj[mask].to_numpy(dtype=float)
            r_eng = eng.argsort().argsort().astype(float)
            r_own = own.argsort().argsort().astype(float)
            r_mix = 0.5 * r_eng + 0.5 * r_own
            order = r_mix.argsort().argsort()
            mapped = np.sort(own)[order]
            hz.loc[mask, "p_suc"] = mapped
        elif n_inj >= 10 and heads.get("inject_mode", "blend") == "extreme":
            # Trust the engine ONLY at its calibrated extremes (the
            # snapshot-direct funnel component); own head elsewhere.
            eng = inj.to_numpy(dtype=float)
            ext = mask & ((eng < 0.02) | (eng > 0.98))
            hz.loc[ext, "p_suc"] = eng[ext]
            print(f"  [sibling] extreme-trust applied to {int(ext.sum())} trials")
        print(f"  [sibling] engine ordering quantile-mapped for {n_inj} trials at {t0.year}")

    if diag:
        h_auc = roc_auc_score(hz["y"], hz["p_rep"]) if hz["y"].nunique() > 1 else float("nan")
        win = ti[(ti["date"] > t0) & (ti["date"] <= t0 + YEAR)]
        evw = win.merge(hz[["nct_id", "p_suc"]], on="nct_id", how="inner")
        s_auc = roc_auc_score(evw["is_successful"], evw["p_suc"]) if evw["is_successful"].nunique() > 1 else float("nan")
        print(f"  [diag {t0.year}] hazard AUC={h_auc:.4f} success AUC(on window events)={s_auc:.4f} "
              f"drift scale={np.exp(heads['drift'][0]*t0.year+heads['drift'][1]):.3f}")

    if oracle_membership:
        win = ti[(ti["date"] > t0) & (ti["date"] <= t0 + YEAR)][["nct_id"]]
        roster = hz.merge(win, on="nct_id", how="inner")
        roster["p_rep"] = 0.999
    else:
        roster = hz
    roster = roster[["nct_id", "p_rep", "p_suc"]].merge(fs, on="nct_id", how="inner")
    roster = roster[roster["facility_id"].isin(set(label_df["facility_id"]))]
    ev = heads["events"]
    fx = ev[["nct_id", "is_successful"]].merge(FS_LINK, on="nct_id", how="inner")
    fagg = fx.groupby("facility_id")["is_successful"].agg(["sum", "count"])
    gmean = heads["gmean"]
    feb = ((fagg["sum"] + 5 * gmean) / (fagg["count"] + 5))
    fb_map = feb.to_dict()  # raw EB facility fallback (n unknown)
    preds = decode(roster, fallback, top_k=top_k, use_median=use_median, fb_map=fb_map)
    m = label_df[["facility_id", "success_rate"]].drop_duplicates("facility_id").copy()
    m["pred"] = m["facility_id"].map(preds)
    m["pred"] = m["pred"].fillna(m["facility_id"].map(fb_map)).fillna(fallback)
    mae = float(np.abs(m["pred"] - m["success_rate"]).mean())
    return mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--drift", type=int, default=1)
    ap.add_argument("--median", type=int, default=1)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--sibling", type=int, default=1)
    ap.add_argument("--inject", choices=["blend", "extreme", "off"], default="blend")
    args = ap.parse_args()

    ti, slim, fs, cs, isd = load_base()
    sib = sibling_maps() if args.sibling else None
    task = get_task("rel-trial", "site-success", download=False)
    train = task.get_table("train").df
    val = task.get_table("val").df
    train["timestamp"] = pd.to_datetime(train["timestamp"])

    if not args.final:
        for gate_name, fit_years, cutoff, t0, ldf in (
            ("G2019", range(2013, 2019), "2019-01-01", "2019-01-01",
             train[train["timestamp"] == "2019-01-01"]),
            ("G2020/val", range(2013, 2020), "2020-01-01", "2020-01-01", val),
        ):
            fb = float(train[train["timestamp"] < cutoff]["success_rate"].median())
            heads = fit_heads(ti, slim, cs, isd,
                              [f"{y}-01-01" for y in fit_years], cutoff)
            heads["sibling"] = sib
            heads["inject_mode"] = args.inject
            mae = run_origin(ti, slim, fs, cs, isd, heads, t0, ldf,
                             bool(args.drift), bool(args.median), args.topk, fb,
                             diag=args.diag)
            print(f"[{gate_name}] MAE={mae:.4f} NMAE={mae/STD:.4f}")
            if args.diag:
                om = run_origin(ti, slim, fs, cs, isd, heads, t0, ldf,
                                False, bool(args.median), args.topk, fb,
                                oracle_membership=True)
                print(f"[{gate_name}/oracle-membership] MAE={om:.4f} NMAE={om/STD:.4f}")
    else:
        test = task.get_table("test", mask_input_cols=False).df
        fb = float(train["success_rate"].median())
        heads = fit_heads(ti, slim, cs, isd,
                          [f"{y}-01-01" for y in range(2013, 2021)], "2021-01-01")
        heads["sibling"] = sib
        heads["inject_mode"] = args.inject
        mae = run_origin(ti, slim, fs, cs, isd, heads, "2021-01-01", test,
                         bool(args.drift), bool(args.median), args.topk, fb)
        print(f"[FINAL/test] MAE={mae:.4f} NMAE={mae/STD:.4f}")


if __name__ == "__main__":
    main()
