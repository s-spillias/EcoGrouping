# -*- coding: utf-8 -*-
"""
Model selection + parallelized bootstrapping + plotting for per-model (LLM) batch-size effects.

What it does
------------
1) Builds a tidy DataFrame from your `analysis_results` (one row per run with F1, model, dataset, replicate, batch size).
2) Per model, evaluates four candidates on the logit scale with dataset fixed-effects:
   - FLAT:  F1_logit ~ C(dataset)
   - PRE:   F1_logit ~ BatchSize_sc + C(dataset)
   - HINGE: F1_logit ~ hinge + C(dataset), τ profiled on a grid
   - FULL:  F1_logit ~ BatchSize_sc + hinge + C(dataset), τ profiled on a grid
   Selects winner by BIC (default) or AIC.
3) Computes two bootstrap tests with inner parallelization (processes):
   - Break test (PRE vs FULL): sup-F with wild cluster bootstrap (Rademacher at replicate level).
   - Pre-slope test (HINGE vs FULL): sup-F with wild cluster bootstrap.
4) Plots (facet by model): dataset lines + only the winning curve; draws τ if hinge-type; annotates BIC/AIC,
   Δ to 2nd best, p_break, p_pre, and ΔF1(+1σ post) when hinge-type.

Inputs
------
- analysis_results: Dict[str, Dict[str, Any]]
    Each value contains at least:
    {
      "model": <str>,
      "dataset": <str>,
      "C": <batch size>,
      "M": <replicate id>,
      "analysis": {"metrics": {"macro_f1_score" or "f1_score": float}}
    }
- metrics_by_dataset: Optional[dict] to draw dataset-colored series in the plot.
    metrics_by_dataset[dataset][model][batch_size] = {
        "multiclass": {"avg_metrics": {"macro_f1_score": float}},
        "std_metrics": {"macro_f1_score": float}   # optional
    }

Usage (example)
---------------
best_df = run_model_selection_with_bootstrap_and_plot(
    analysis_results,
    metrics_by_dataset=metrics_by_dataset,   # or None
    out_dir="Artifacts/ModelSelection",
    fig_prefix="best_fits_by_model",
    criterion="bic",
    tau_quantiles=(0.15, 0.85),
    n_taus=31,
    B_break=999,
    B_preslope=499,
    n_jobs_models=1,      # set >1 to parallelize across models
    n_jobs_inner=8,       # set >1 to parallelize inside bootstrap (avoid nested pools!)
    verbose=True,
)

This writes:
- CSV: Artifacts/ModelSelection/per_model_best_fit_with_tests.csv
- PNG: Figures/best_fits_by_model.png
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import statsmodels.formula.api as smf
from numpy.random import default_rng
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.helper import get_model_shorthand, build_dataset_color_map
# Prevent BLAS oversubscription (good hygiene for parallel loops)
import os as _os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# ---------------------------------------------------------------------------
# Data building helpers
# ---------------------------------------------------------------------------

def _pick_f1(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    """Prefer macro_f1_score if present (multiclass), else binary f1_score."""
    if not metrics:
        return None
    if metrics.get("macro_f1_score") is not None:
        return metrics.get("macro_f1_score")
    return metrics.get("f1_score")


def get_clean_dataset_info(metrics_by_dataset, grouping_dir="Datasets/Groupings"):
    dataset_info = []
    for dataset_name in metrics_by_dataset:
        dataset_file = os.path.join(grouping_dir, f"{dataset_name}.json")
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                size_estimate = len(json.load(f))
        except Exception as e:
            print(f"[warn] Could not read {dataset_file}: {e}")
            size_estimate = 0

        clean = re.sub(r'(?<!^)(?=[A-Z])', ' ',
                       dataset_name.replace('.json', '').replace('_', '\n'))
        dataset_info.append({
            "name": dataset_name,
            "clean": clean,
            "size": size_estimate
        })

    dataset_info.sort(key=lambda x: x["size"])
    return dataset_info


def build_model_fit_results_from_summary(summary_df):
    """
    Return:
      { model_name: {
          'winner': 'FULL'|'HINGE'|'PRE'|'FLAT',
          'alpha': float,
          'beta_pre': float,
          'beta_post': float,         # incremental post-break slope (logit scale)
          'tau': float|None,          # on original batch-size scale
          'mu_B': float,
          'sigma_B': float,
          'gamma_mean': float,        # mean dataset FE (logit scale), for level alignment
          'predict': callable         # lambda batch_array -> F1hat array
      }, ... }
    """
    import numpy as np

    out = {}
    for _, r in summary_df.iterrows():
        mdl   = r["model"]
        win   = str(r["winner"])
        alpha = float(r["alpha"])
        b0    = float(r.get("beta0", 0.0))
        bpls  = float(r.get("beta_plus", 0.0))
        tauB  = r.get("tau_batch")
        muB   = float(r.get("mu_B", 0.0))
        sdB   = float(r.get("sigma_B", 1.0)) or 1.0

        # Dataset FE mean (aligns with the averaging used in stats_model.py)
        gamma_mean = 0.0
        try:
            dsfe = r.get("dataset_fe")
            # dataset_fe is a stringified dict if read from CSV; eval carefully if so
            if isinstance(dsfe, str):
                import ast
                dsfe = ast.literal_eval(dsfe)
            if isinstance(dsfe, dict) and len(dsfe):
                gamma_mean = float(np.mean(list(dsfe.values())))
        except Exception:
            gamma_mean = 0.0

        # Map coefficients by winner
        if win == "FULL":
            beta_pre  = b0
            beta_post = bpls
            tau       = float(tauB) if np.isfinite(tauB) else None
        elif win == "HINGE":
            beta_pre  = 0.0
            beta_post = bpls
            tau       = float(tauB) if np.isfinite(tauB) else None
        elif win == "PRE":
            beta_pre  = b0
            beta_post = 0.0
            tau       = None
        else:  # FLAT
            beta_pre  = 0.0
            beta_post = 0.0
            tau       = None

        def _mk_predict(alpha=alpha, beta_pre=beta_pre, beta_post=beta_post,
                        tau=tau, muB=muB, sdB=sdB, gamma_mean=gamma_mean):
            def _expit(z): return 1.0 / (1.0 + np.exp(-z))
            def _pred(batch_sizes):
                x  = np.asarray(batch_sizes, dtype=float)
                xs = (x - muB) / (sdB if sdB != 0 else 1.0)
                if tau is None:
                    hinge = np.zeros_like(xs)
                    z = (alpha + gamma_mean) + beta_pre * xs
                else:
                    tau_sc = (tau - muB) / (sdB if sdB != 0 else 1.0)
                    hinge  = np.clip(xs - tau_sc, 0, None)
                    z = (alpha + gamma_mean) + beta_pre * xs + beta_post * hinge
                return _expit(z)
            return _pred

        out[mdl] = {
            "winner": win,
            "alpha": alpha,
            "beta_pre": beta_pre,
            "beta_post": beta_post,
            "tau": tau,
            "mu_B": muB,
            "sigma_B": sdB,
            "gamma_mean": gamma_mean,
            "predict": _mk_predict()
        }

    return out


def build_mixed_model_df(analysis_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Return tidy DataFrame with columns:
       ['F1','batch_size','model','dataset','replicate','run_id','F1_logit','BatchSize_sc'] (latter computed later)."""
    rows: List[Dict[str, Any]] = []
    for key, res in analysis_results.items():
        ds = res.get("dataset")
        mdl = res.get("model")
        c = res.get("C")    # batch size
        r = res.get("M")    # replicate
        met = (res.get("analysis") or {}).get("metrics") or {}
        f1 = _pick_f1(met)
        if f1 is None:
            continue
        rows.append({
            "F1": float(f1),
            "batch_size": float(c),
            "model": str(mdl),
            "dataset": str(ds),
            "replicate": str(r),
            "run_id": key,
        })
    df = pd.DataFrame(rows)
    return df

def coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("dataset","model","replicate"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ---------------------------------------------------------------------------
# Model specs + OLS utilities
# ---------------------------------------------------------------------------

def _fit_ols(formula: str, df: pd.DataFrame):
    """Simple OLS fit with statsmodels."""
    model = smf.ols(formula, data=df)
    res = model.fit()
    return res

def _ols_null_formula_flat() -> str:
    # FLAT: constant by dataset FE
    return "F1_logit ~ C(dataset)"

def _ols_null_formula_pre_only() -> str:
    # PRE (monotone): one slope in BatchSize_sc + dataset FE
    return "F1_logit ~ BatchSize_sc + C(dataset)"

def _ols_alt_formula_collapse() -> str:
    # HINGE: post-break slope only + dataset FE
    return "F1_logit ~ hinge + C(dataset)"

def _ols_alt_formula_full() -> str:
    # FULL: pre-slope + post-break slope + dataset FE
    return "F1_logit ~ BatchSize_sc + hinge + C(dataset)"

def _supF_stat(res0, res1) -> Tuple[float, int, int]:
    """Classic improvement-in-fit F-stat for nested OLS."""
    rss0, rss1 = float(res0.ssr), float(res1.ssr)
    df0, df1 = int(res0.df_resid), int(res1.df_resid)
    df_num = int(df0 - df1)
    df_den = int(df1)
    if df_num <= 0 or df_den <= 0:
        return np.nan, df_num, df_den
    F = ((rss0 - rss1)/df_num) / (rss1/df_den)
    return float(F), df_num, df_den

def _criterion_value(res, criterion: Literal["bic","aic"]="bic") -> float:
    """Return BIC or AIC; inf on failure."""
    try:
        return float(getattr(res, criterion))
    except Exception:
        return float("inf")

# ---------------------------------------------------------------------------
# τ profiling + model selection (BIC/AIC)
# ---------------------------------------------------------------------------

def _profile_tau_for_model(
    df: pd.DataFrame,
    taus_sc: np.ndarray,
    which: Literal["hinge","full"] = "full",
    criterion: Literal["bic","aic"] = "bic"
):
    """Return best result dict (tau_sc,res,crit) for which ∈ {'hinge','full'}."""
    best = None
    best_val = np.inf  # lower is better for BIC/AIC

    for t in taus_sc:
        w = df.copy()
        w["hinge"] = np.clip(w["BatchSize_sc"] - t, 0, None)
        formula = _ols_alt_formula_collapse() if which == "hinge" else _ols_alt_formula_full()
        res = _fit_ols(formula, w)

        crit_unpen = _criterion_value(res, criterion)
        n = len(df)
        pen = (np.log(max(n, 2)) if criterion == "bic" else 2.0)  # τ penalty
        crit = crit_unpen + pen

        # keep verbose print
        print(f"[{df['model'].iloc[0]}] {which.upper()}@tau_sc={t:.4f} "
              f"{criterion.upper()}_unpen={crit_unpen:.3f} +pen={pen:.3f} => {crit:.3f}")

        if np.isfinite(crit) and crit < best_val:
            best_val = crit
            best = {"tau_sc": float(t), "res": res, "crit": float(crit)}

    return best


def _fit_nonhinge_models(df: pd.DataFrame, criterion: Literal["bic","aic"]="bic"):
    out = {}
    res_flat = _fit_ols(_ols_null_formula_flat(), df)
    out["FLAT"] = {"res": res_flat, "crit": _criterion_value(res_flat, criterion)}
    res_pre = _fit_ols(_ols_null_formula_pre_only(), df)
    out["PRE"] = {"res": res_pre, "crit": _criterion_value(res_pre, criterion)}
    return out

def _select_best_candidate(dfm: pd.DataFrame,
                           mu: float, sd: float,
                           tau_quantiles=(0.15, 0.85),
                           n_taus: int = 31,
                           criterion: Literal["bic","aic"]="bic") -> Optional[dict]:
    """Return dict describing the BIC/AIC winner for this model subset."""
    if int(dfm["BatchSize_sc"].nunique()) < 3:
        return None
    lo = float(dfm["BatchSize_sc"].quantile(tau_quantiles[0]))
    hi = float(dfm["BatchSize_sc"].quantile(tau_quantiles[1]))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    taus_sc = np.linspace(lo, hi, max(7, n_taus))

    cand = _fit_nonhinge_models(dfm, criterion=criterion)
    best_hinge = _profile_tau_for_model(dfm, taus_sc, which="hinge", criterion=criterion)
    if best_hinge is not None:
        cand["HINGE"] = {"res": best_hinge["res"], "crit": best_hinge["crit"], "tau_sc": best_hinge["tau_sc"]}
    best_full = _profile_tau_for_model(dfm, taus_sc, which="full", criterion=criterion)
    if best_full is not None:
        cand["FULL"] = {"res": best_full["res"], "crit": best_full["crit"], "tau_sc": best_full["tau_sc"]}
    # Inside _select_best_candidate before ranking:
    # After building cand = {...} and after profiling best_hinge / best_full
    print(f"[{dfm['model'].iloc[0]}] candidate {criterion.upper()}s:")
    for name, v in cand.items():
        tau_show = v.get("tau_sc")
        if tau_show is None or (isinstance(tau_show, float) and not np.isfinite(tau_show)):
            print(f"  {name:5s}: {v['crit']:.3f}   tau_sc=None")
        else:
            print(f"  {name:5s}: {v['crit']:.3f}   tau_sc={float(tau_show):.3f}")



    ranked = sorted([(k, v["crit"]) for k, v in cand.items()], key=lambda kv: kv[1])
    if not ranked:
        return None
    winner, best_crit = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else float("nan")
    gap = float(second - best_crit) if np.isfinite(second) else np.nan
    print(f"[select-best] {dfm['model'].iloc[0]}  winner={winner}  {criterion.upper()}={best_crit:.3f}  Δ={gap:.3f}")
    res_win = cand[winner]["res"]
    tau_sc  = float(cand[winner]["tau_sc"]) if "tau_sc" in cand[winner] else np.nan
    tau_batch = float(mu + sd * tau_sc) if np.isfinite(tau_sc) else np.nan

    params = res_win.params
    alpha_0   = float(params.get("Intercept", 0.0))
    beta0     = float(params.get("BatchSize_sc", 0.0)) if winner in ("PRE","FULL") else 0.0
    beta_plus = float(params.get("hinge", 0.0))        if winner in ("HINGE","FULL") else 0.0
    ds_levels = sorted(dfm["dataset"].unique())
    dataset_fe = {ds: float(params.get(f"C(dataset)[T.{ds}]", 0.0)) for ds in ds_levels}

    return {
        "winner": winner,
        "crit_name": criterion.upper(),
        "crit_best": float(best_crit),
        "crit_delta_to_2nd": float(gap),
        "tau_sc": float(tau_sc) if np.isfinite(tau_sc) else np.nan,
        "tau_batch": float(tau_batch) if np.isfinite(tau_batch) else np.nan,
        "res": res_win,
        "alpha": alpha_0,
        "beta0": beta0,
        "beta_plus": beta_plus,
        "mu_B": float(mu),
        "sigma_B": float(sd),
        "dataset_fe": dataset_fe,
    }

# ---------------------------------------------------------------------------
# Predictions (average across dataset FE) on original F1 scale
# ---------------------------------------------------------------------------

from typing import Dict, List, Optional, Literal
from collections import OrderedDict

def _avg_predictions_on_F1_scale(
    res,
    dfm: pd.DataFrame,
    mu: float,
    sd: float,
    batch_values: List[float],
    tau_sc: float | None,
    *,
    avg_on: Literal["logit", "prob"] = "prob",  # NEW: choose averaging scale
) -> Dict[float, float]:
    """
    Return {B: F1hat} averaged over datasets present in dfm.

    avg_on="logit": average predicted logit across datasets, then expit.
    avg_on="prob":  average predicted probability (expit(logit)) across datasets.
    """
    def _expit(z): 
        return 1.0 / (1.0 + np.exp(-z))

    xs = np.array(batch_values, dtype=float)
    Xg = pd.DataFrame({
        "batch_size": xs,
        "BatchSize_sc": (xs - mu) / (sd if sd != 0 else 1.0),
    })
    if (tau_sc is not None) and np.isfinite(tau_sc):
        Xg["hinge"] = np.clip(Xg["BatchSize_sc"] - tau_sc, 0, None)

    ds_levels = sorted(dfm["dataset"].unique())
    # Keep duplicated row indices on purpose so we can groupby(level=0)
    X_all = pd.concat([Xg.assign(dataset=ds) for ds in ds_levels], ignore_index=False)

    pred = res.predict(X_all)
    P = pd.Series(np.asarray(pred), index=X_all.index)  # ensure Series with matching index

    if avg_on == "logit":
        # mean(logit) -> expit
        logit_mean = P.groupby(level=0).mean().values
        f1_vals = _expit(logit_mean)
    else:
        # mean(expit(logit)) on probability scale
        f1_vals = P.groupby(level=0).apply(lambda z: float(np.mean(_expit(np.asarray(z, dtype=float))))).values

    # Preserve ascending B order for plotting
    return OrderedDict((float(B), float(v)) for B, v in zip(batch_values, f1_vals))


# ---------------------------------------------------------------------------
# Bootstrap (parallel, wild-cluster) for break & pre-slope tests
# ---------------------------------------------------------------------------

def _cw_bootstrap_break_chunk(df: pd.DataFrame,
                              taus_sc: np.ndarray,
                              B: int,
                              cluster_col: str,
                              seed: int):
    """Chunk worker for break test bootstrap: PRE (H0) vs FULL (H1), sup-F across τ."""
    rng = default_rng(seed)
    # Fit PRE under H0 on original data
    res0 = _fit_ols(_ols_null_formula_pre_only(), df)
    fitted0 = res0.fittedvalues.values
    resid0 = df["F1_logit"].values - fitted0
    clusters = df[cluster_col].astype(str).values
    groups = np.unique(clusters)

    supF_star, tau_star = [], []
    for _ in range(B):
        w = {g: rng.choice([-1.0, 1.0]) for g in groups}
        e_star = np.array([w[g] * r for g, r in zip(clusters, resid0)])
        y_star = fitted0 + e_star
        dfs = df.copy()
        dfs["F1_logit"] = y_star
        res0s = _fit_ols(_ols_null_formula_pre_only(), dfs)

        bestF, bestTau = -np.inf, np.nan
        for t in taus_sc:
            work = dfs.copy()
            work["hinge"] = np.clip(work["BatchSize_sc"] - t, 0, None)
            res1s = _fit_ols(_ols_alt_formula_full(), work)
            F, _, _ = _supF_stat(res0s, res1s)
            if np.isfinite(F) and F > bestF:
                bestF, bestTau = F, t
        supF_star.append(bestF)
        tau_star.append(bestTau)
    return np.asarray(supF_star), np.asarray(tau_star)

def _cw_bootstrap_break_parallel(df: pd.DataFrame,
                                 taus_sc: np.ndarray,
                                 *,
                                 B: int,
                                 cluster_col: str = "replicate",
                                 seed: int = 123,
                                 n_jobs_inner: int = 4):
    """Parallel break-test bootstrap returning (supF_star, tau_star)."""
    n_jobs_inner = max(1, int(n_jobs_inner))
    # Split B into nearly equal chunks
    sizes = [B // n_jobs_inner] * n_jobs_inner
    for i in range(B % n_jobs_inner):
        sizes[i] += 1

    sup_all, tau_all = [], []
    with ProcessPoolExecutor(max_workers=n_jobs_inner) as ex:
        futs = []
        offset = 0
        for bsize in sizes:
            futs.append(ex.submit(_cw_bootstrap_break_chunk, df, taus_sc, bsize, cluster_col, seed + offset))
            offset += 100_000
        for fut in as_completed(futs):
            s, t = fut.result()
            sup_all.append(s); tau_all.append(t)
    return np.concatenate(sup_all), np.concatenate(tau_all)

def _preslope_bootstrap_chunk(df: pd.DataFrame,
                              taus_sc: np.ndarray,
                              B: int,
                              cluster_col: str,
                              seed: int):
    """Chunk worker for pre-slope test bootstrap: HINGE (H0) vs FULL (H1), sup-F across τ."""
    rng = default_rng(seed)
    # Profile τ under H0 (hinge-only) to get best fitted/residuals for DGP
    best0 = None
    for t in taus_sc:
        w0 = df.copy()
        w0["hinge"] = np.clip(w0["BatchSize_sc"] - t, 0, None)
        r0 = _fit_ols(_ols_alt_formula_collapse(), w0)
        llf = float(r0.llf)
        if (best0 is None) or (llf > best0["llf"]):
            best0 = {"tau_sc": float(t), "res0": r0, "llf": llf}
    fitted0 = best0["res0"].fittedvalues.values
    resid0 = df["F1_logit"].values - fitted0
    clusters = df[cluster_col].astype(str).values
    uniq = np.unique(clusters)

    supF_star = []
    for _ in range(B):
        w = {g: rng.choice([-1.0, 1.0]) for g in uniq}
        e_star = np.array([w[g]*r for g, r in zip(clusters, resid0)])
        y_star = fitted0 + e_star
        dfs = df.copy()
        dfs["F1_logit"] = y_star

        bestF = -np.inf
        for t in taus_sc:
            ws = dfs.copy()
            ws["hinge"] = np.clip(ws["BatchSize_sc"] - t, 0, None)
            res0s = _fit_ols(_ols_alt_formula_collapse(), ws)
            ress  = _fit_ols(_ols_alt_formula_full(), ws)
            F, _, _ = _supF_stat(res0s, ress)
            if np.isfinite(F) and F > bestF:
                bestF = F
        supF_star.append(bestF)
    return np.asarray(supF_star)

def _preslope_bootstrap_parallel(df: pd.DataFrame,
                                 taus_sc: np.ndarray,
                                 *,
                                 B: int,
                                 cluster_col: str = "replicate",
                                 seed: int = 123,
                                 n_jobs_inner: int = 4):
    """Parallel pre-slope bootstrap returning supF_star array."""
    n_jobs_inner = max(1, int(n_jobs_inner))
    sizes = [B // n_jobs_inner] * n_jobs_inner
    for i in range(B % n_jobs_inner):
        sizes[i] += 1

    sup_all = []
    with ProcessPoolExecutor(max_workers=n_jobs_inner) as ex:
        futs = []
        offset = 0
        for bsize in sizes:
            futs.append(ex.submit(_preslope_bootstrap_chunk, df, taus_sc, bsize, cluster_col, seed + offset))
            offset += 100_000
        for fut in as_completed(futs):
            s = fut.result()
            sup_all.append(s)
    return np.concatenate(sup_all)

# ---------------------------------------------------------------------------
# Per-model: compute selection, observed F-statistics, parallel bootstraps
# ---------------------------------------------------------------------------

def _analyze_one_model(dfm: pd.DataFrame,
                       *,
                       mu: float, sd: float,
                       criterion: Literal["bic","aic"],
                       tau_quantiles: Tuple[float,float],
                       n_taus: int,
                       B_break: int, B_preslope: int,
                       n_jobs_inner: int,
                       seed: int = 123) -> Optional[dict]:
    """Return a dict containing selection result, bootstrap p-values, and plotting curves."""
    pick = _select_best_candidate(dfm, mu, sd, tau_quantiles=tau_quantiles, n_taus=n_taus, criterion=criterion)
    if pick is None:
        return None

    # τ grid for tests (use same range as selection)
    lo = float(dfm["BatchSize_sc"].quantile(tau_quantiles[0]))
    hi = float(dfm["BatchSize_sc"].quantile(tau_quantiles[1]))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        taus_sc = np.linspace(-1.0, 1.0, max(7, n_taus))  # fallback
    else:
        taus_sc = np.linspace(lo, hi, max(7, n_taus))

    # Observed sup-F for break test: PRE (null) vs FULL (alt)
    res_pre = _fit_ols(_ols_null_formula_pre_only(), dfm)
    F_obs_break = -np.inf
    for t in taus_sc:
        w = dfm.copy()
        w["hinge"] = np.clip(w["BatchSize_sc"] - t, 0, None)
        res_full_t = _fit_ols(_ols_alt_formula_full(), w)
        F_t, df_num, df_den = _supF_stat(res_pre, res_full_t)
        if np.isfinite(F_t) and F_t > F_obs_break:
            F_obs_break = F_t

    # Observed sup-F for pre-slope test: HINGE (null) vs FULL (alt)
    F_obs_pre = -np.inf
    for t in taus_sc:
        w = dfm.copy()
        w["hinge"] = np.clip(w["BatchSize_sc"] - t, 0, None)
        res_h0 = _fit_ols(_ols_alt_formula_collapse(), w)
        res_h1 = _fit_ols(_ols_alt_formula_full(), w)
        F_t, _, _ = _supF_stat(res_h0, res_h1)
        if np.isfinite(F_t) and F_t > F_obs_pre:
            F_obs_pre = F_t

    # Parallel bootstraps
    supF_star_break, tau_star_break = _cw_bootstrap_break_parallel(
        dfm, taus_sc, B=B_break, cluster_col="replicate", seed=seed, n_jobs_inner=n_jobs_inner
    )
    p_break = (np.sum(supF_star_break >= F_obs_break) + 1) / (len(supF_star_break) + 1)

    supF_star_pre = _preslope_bootstrap_parallel(
        dfm, taus_sc, B=B_preslope, cluster_col="replicate", seed=seed, n_jobs_inner=n_jobs_inner
    )
    p_pre = (np.sum(supF_star_pre >= F_obs_pre) + 1) / (len(supF_star_pre) + 1)

    # τ CI / IQR from break bootstrap τ*
    if len(tau_star_break):
        tau_ci_lo_sc = float(np.quantile(tau_star_break, 0.025))
        tau_ci_hi_sc = float(np.quantile(tau_star_break, 0.975))
        tau_iqr_sc = float(np.subtract(*np.quantile(tau_star_break, [0.75, 0.25])))
        tau_ci_lo = mu + sd * tau_ci_lo_sc
        tau_ci_hi = mu + sd * tau_ci_hi_sc
        tau_iqr = sd * tau_iqr_sc
    else:
        tau_ci_lo = tau_ci_hi = tau_iqr = np.nan

    # Build best-fit curve for plotting
    # Use union of observed batch sizes within this model
    B_values = sorted(dfm["batch_size"].unique().tolist())
    tau_sc = pick["tau_sc"] if np.isfinite(pick["tau_sc"]) else None
    # --- Slope-equality guardrail: demote when pre/post slopes are (near) identical ---
    eps_slope = 1e-3  # tolerance on logit-scale slope per 1 SD of batch size
    # You may tune eps_slope; e.g., 5e-4 (strict) or 2e-3 (looser)

    def _demote_to_flat():
        res_flat = _fit_ols(_ols_null_formula_flat(), dfm)
        pick.update({
            "winner": "FLAT",
            "res": res_flat,
            "tau_sc": np.nan,
            "tau_batch": np.nan,
            "beta0": 0.0,
            "beta_plus": 0.0,
        })

    def _demote_to_pre():
        res_pre = _fit_ols(_ols_null_formula_pre_only(), dfm)
        # Pull the PRE slope into beta0 so plotting/annotation remain coherent
        b0_pre = float(res_pre.params.get("BatchSize_sc", 0.0))
        pick.update({
            "winner": "PRE",
            "res": res_pre,
            "tau_sc": np.nan,
            "tau_batch": np.nan,
            "beta0": b0_pre,
            "beta_plus": 0.0,
        })

    # Only meaningful for hinge-type winners where a break is claimed
    if pick["winner"] in ("HINGE", "FULL") and np.isfinite(pick.get("tau_sc", np.nan)):
        t = float(pick["tau_sc"])
        w = dfm.copy()
        w["hinge"] = np.clip(w["BatchSize_sc"] - t, 0, None)
        res_full_at_t = _fit_ols(_ols_alt_formula_full(), w)

        b0_full   = float(res_full_at_t.params.get("BatchSize_sc", 0.0))
        bplus_full= float(res_full_at_t.params.get("hinge", 0.0))
        pre_slope = b0_full
        post_slope = b0_full + bplus_full
        delta = abs(post_slope - pre_slope)

        if delta < eps_slope:
            # No meaningful slope change across τ -> no break
            if abs(pre_slope) < eps_slope:
                _demote_to_flat()   # essentially flat
            else:
                _demote_to_pre()    # one-slope model


    best_curve = _avg_predictions_on_F1_scale(
    pick["res"], dfm, mu, sd, B_values, tau_sc, avg_on="prob"
    )


    # Practical post-break ΔF1 for +1 SD (only meaningful for hinge-type winners)
    ds_levels = sorted(dfm["dataset"].unique())
    params = pick["res"].params
    alpha0 = pick["alpha"]
    beta_pre = float(params.get("BatchSize_sc", 0.0))
    beta_post = float(params.get("hinge", 0.0))
    gamma_vals = [float(params.get(f"C(dataset)[T.{ds}]", 0.0)) for ds in ds_levels]
    gamma_mean = float(np.mean(gamma_vals)) if len(gamma_vals) else 0.0

    def _expit(z): return 1.0 / (1.0 + np.exp(-z))
    deltaF1_post_1sd = None
    if pick["winner"] in ("HINGE","FULL") and (tau_sc is not None):
        z0 = alpha0 + gamma_mean + beta_pre * (tau_sc) + beta_post * 0.0
        z1 = alpha0 + gamma_mean + beta_pre * (tau_sc + 1.0) + beta_post * 1.0
        deltaF1_post_1sd = float(_expit(z1) - _expit(z0))

    return {
        "winner": pick["winner"],
        "crit_name": pick["crit_name"],
        "crit_best": pick["crit_best"],
        "crit_delta_to_2nd": pick["crit_delta_to_2nd"],
        "tau_sc": pick["tau_sc"],
        "tau_batch": pick["tau_batch"],
        "tau_ci_lo": float(tau_ci_lo),
        "tau_ci_hi": float(tau_ci_hi),
        "tau_iqr": float(tau_iqr),
        "p_break": float(p_break),
        "p_pre": float(p_pre),
        "alpha": pick["alpha"],
        "beta0": pick["beta0"],
        "beta_plus": pick["beta_plus"],
        "mu_B": pick["mu_B"],
        "sigma_B": pick["sigma_B"],
        "dataset_fe": pick["dataset_fe"],
        "fitted_best": best_curve,
        "deltaF1_post_1sd": deltaF1_post_1sd,
    }

# ---------------------------------------------------------------------------
# Plotting: facet by model, dataset lines + best fit + τ + annotations
# ---------------------------------------------------------------------------

def _ensure_sorted_numeric(keys) -> List[float]:
    out = []
    for k in keys:
        try:
            out.append(int(k))
        except Exception:
            try:
                out.append(float(k))
            except Exception:
                pass
    return sorted(set(out))

from collections import OrderedDict
from typing import Optional, Dict, List, Literal, Tuple

def _all_chunks_from_metrics(metrics_by_dataset: dict) -> List[float]:
    """Collect and return a sorted list of all batch sizes across datasets/models from metrics_by_dataset."""
    chunks = set()
    for ds, by_model in (metrics_by_dataset or {}).items():
        for m, by_B in (by_model or {}).items():
            for k in (by_B or {}).keys():
                try:
                    chunks.add(int(k))
                except Exception:
                    try:
                        chunks.add(float(k))
                    except Exception:
                        pass
    return sorted(chunks)

def _build_minimal_df_from_metrics(metrics_by_dataset: dict) -> pd.DataFrame:
    """Construct a minimal tidy DataFrame from metrics_by_dataset to support plotting and axis construction."""
    rows = []
    for ds, by_model in (metrics_by_dataset or {}).items():
        for m, by_B in (by_model or {}).items():
            for k, entry in (by_B or {}).items():
                # accept numeric or string keys
                try:
                    B = float(k)
                except Exception:
                    continue
                f1 = None
                if isinstance(entry, dict) and "multiclass" in entry:
                    avg = entry["multiclass"].get("avg_metrics") or {}
                    f1 = avg.get("macro_f1_score")
                if f1 is None:
                    continue
                rows.append({
                    "dataset": str(ds),
                    "model": str(m),
                    "batch_size": float(B),
                    "F1": float(f1),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = coerce_categoricals(df)
    return df

def _make_best_curve_for_model(predict_fn, batch_values: List[float]) -> "OrderedDict[float, float]":
    """Use a provided predict(batch_array)->F1hat to build an ordered {B: F1} mapping."""
    B_sorted = sorted(set(float(b) for b in batch_values))
    yhat = predict_fn(np.array(B_sorted, dtype=float))
    return OrderedDict((float(B), float(v)) for B, v in zip(B_sorted, yhat))


def _xpos_for_val(batch_sorted: List[float], pos: List[int], value: float) -> float:
    if value <= batch_sorted[0]:
        return pos[0]
    if value >= batch_sorted[-1]:
        return pos[-1]
    for i in range(len(batch_sorted)-1):
        L, R = batch_sorted[i], batch_sorted[i+1]
        if L <= value <= R:
            frac = (value - L) / (R - L) if R > L else 0.0
            return pos[i] + frac * (pos[i+1] - pos[i])
    return pos[-1]

def _plot_best_fits_by_model(
    df: Optional[pd.DataFrame],
    results_by_model: Dict[str, dict],
    metrics_by_dataset: Optional[dict],
    output_prefix: str,
    *,
    show_means: bool = False,
    show_scatter: bool = True,
    scatter_alpha: float = 0.45,
    scatter_size: float = 14,
    jitter: float = 0.12,
    curve_color: str = "#d62728",
    curve_lw: float = 2.6,
    tau_line_color: str = "0.35",
    tau_line_ls: str = "--",
    tau_line_lw: float = 1.6,
    tau_ci_alpha: float = 0.12
):
    """
    Faceted (by model) plot showing:
      • Optional dataset mean lines,
      • replicate-level scatter,
      • best-fit curve (selected model),
      • τ line/CI when appropriate.

    NEW:
      • Dataset "size" is derived as the largest numeric key in metrics_by_dataset[dataset][model].
      • Datasets are sorted by that size (ascending) for legend order.
      • Figure-level legend always shows 'Clean Name (size)' and uses colors in that same order.
    """
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    os.makedirs("Figures", exist_ok=True)

    # ---------- Helpers ----------
    def _clean_ds(name: str) -> str:
        # underscore -> newline, split CamelCase, strip .json if present
        base = name.replace(".json", "")
        return re.sub(r'(?<!^)(?=[A-Z])', '', base.replace('_', '\n'))

    def _numeric_keys(d: dict):
        out = []
        for k in (d or {}).keys():
            try:
                out.append(int(k))
                continue
            except Exception:
                pass
            try:
                out.append(float(k))
            except Exception:
                pass
        return out

    def _largest_chunk_in_dataset(ds_name: str) -> float:
        # Prefer metrics_by_dataset structure
        if isinstance(metrics_by_dataset, dict) and ds_name in metrics_by_dataset:
            by_model = metrics_by_dataset.get(ds_name) or {}
            maxi = None
            for _m, by_B in (by_model or {}).items():
                keys = _numeric_keys(by_B or {})
                if keys:
                    m_local = max(keys)
                    maxi = m_local if maxi is None else max(maxi, m_local)
            if maxi is not None:
                return float(maxi)
        # Fallback: use df's batch_size for that dataset
        if df is not None and not df.empty:
            sub = df[df["dataset"].astype(str) == str(ds_name)]
            if not sub.empty:
                try:
                    return float(np.nanmax(sub["batch_size"].astype(float).values))
                except Exception:
                    pass
        return 0.0

    def _xpos_for_B(val: float, batch_sorted, pos):
        return _xpos_for_val(batch_sorted, pos, float(val))

    # ---------- Determine models / datasets / x-axis ticks ----------
    if (df is None) or df.empty:
        model_names = sorted({
            m for ds in (metrics_by_dataset or {})
            for m in ((metrics_by_dataset or {}).get(ds) or {}).keys()
        })
        base_dataset_names = sorted(list((metrics_by_dataset or {}).keys()))
        all_chunks = _all_chunks_from_metrics(metrics_by_dataset or {})
    else:
        model_names = sorted(df["model"].unique().tolist())
        base_dataset_names = sorted(df["dataset"].unique().tolist())
        all_chunks = _ensure_sorted_numeric(df["batch_size"].unique())

    if len(model_names) == 0:
        print("[select-best] Nothing to plot: no models found.")
        return

    x_positions = list(range(len(all_chunks)))

    # ---------- Build size-ordered dataset info ----------
    dataset_info = []
    for ds in base_dataset_names:
        size_est = _largest_chunk_in_dataset(ds)
        dataset_info.append({
            "name": ds,
            "clean": _clean_ds(ds),
            "size": float(size_est if np.isfinite(size_est) else 0.0),
        })
    # Sort by size ascending (smallest first)
    dataset_info.sort(key=lambda x: x["size"])

    dataset_names = [d["name"] for d in dataset_info] if dataset_info else base_dataset_names
    dataset_display = {d["name"]: f"{d['clean']} ({int(d['size'])})" for d in dataset_info} \
                      if dataset_info else {ds: _clean_ds(ds) for ds in base_dataset_names}

    # ---------- Dataset colors in the same size order ----------
    pal = sns.color_palette("tab20", max(1, len(dataset_names)))
    dataset_color_map = {ds: pal[i % len(pal)] for i, ds in enumerate(dataset_names)}

    # ---------- Facet layout ----------
    n_models = len(model_names)
    n_cols = 3 if n_models > 4 else (2 if n_models > 1 else 1)
    n_rows = (n_models + n_cols - 1) // n_cols
    # Widen a bit to leave room for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols + 2.2, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(1234)  # deterministic jitter

    def _choose_box_corner(ax, x_all, y_all):
        if len(x_all) == 0:
            return 'upper right', dict(x=0.98, y=0.98, ha='right', va='top')
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        if xmax <= xmin: xmax = xmin + 1.0
        if ymax <= ymin: ymax = ymin + 1.0
        xn = (np.asarray(x_all, dtype=float) - xmin) / (xmax - xmin)
        yn = (np.asarray(y_all, dtype=float) - ymin) / (ymax - ymin)
        quads = {
            'upper left': ((xn < 0.5) & (yn >= 0.5)).sum(),
            'upper right': ((xn >= 0.5) & (yn >= 0.5)).sum(),
            'lower left': ((xn < 0.5) & (yn < 0.5)).sum(),
            'lower right': ((xn >= 0.5) & (yn < 0.5)).sum(),
        }
        order = ['upper right', 'upper left', 'lower right', 'lower left']
        corner = order[int(np.argmin([quads[c] for c in order]))]
        pos = {
            'upper left': dict(x=0.02, y=0.98, ha='left', va='top'),
            'upper right': dict(x=0.98, y=0.98, ha='right', va='top'),
            'lower right': dict(x=0.98, y=0.02, ha='right', va='bottom'),
            'lower left': dict(x=0.02, y=0.02, ha='left', va='bottom'),
        }[corner]
        return corner, pos

    # ---------- Panels ----------
    for idx, m in enumerate(model_names):
        ax = axes[idx]
        x_all_points, y_all_points = [], []

        # (A) Dataset-mean lines (size-ordered so colors match legend)
        if show_means and isinstance(metrics_by_dataset, dict) and len(metrics_by_dataset):
            for ds in dataset_names:
                md = (metrics_by_dataset.get(ds, {}) or {}).get(m, {}) or {}
                if not md:
                    continue
                yvals, yerr, xpos = [], [], []
                for j, B in enumerate(all_chunks):
                    entry = md.get(B, md.get(str(B)))
                    if entry and "multiclass" in entry and (entry.get("multiclass") or {}).get("avg_metrics"):
                        y = (entry["multiclass"]["avg_metrics"] or {}).get("macro_f1_score")
                        if y is None:
                            continue
                        yvals.append(float(y))
                        err = (entry.get("std_metrics") or {}).get("macro_f1_score")
                        yerr.append(np.nan if err is None else float(err))
                        xpos.append(j)
                if xpos:
                    ax.errorbar(
                        xpos, yvals,
                        yerr=yerr, fmt='o-', linewidth=1.1, capsize=3,
                        alpha=0.9, markersize=3.2, color=dataset_color_map.get(ds, "0.35"),
                        label=None, zorder=2
                    )
                    x_all_points.extend(xpos)
                    y_all_points.extend(yvals)

        # (B) Replicate-level scatter
        if show_scatter and (df is not None) and not df.empty:
            dfm = df[df["model"] == m]
            if not dfm.empty:
                x_scatter = dfm["batch_size"].apply(lambda v: _xpos_for_B(v, all_chunks, x_positions)).astype(float).values
                x_scatter = x_scatter + rng.uniform(-jitter, +jitter, size=x_scatter.shape)
                y_scatter = dfm["F1"].astype(float).values
                ds_series = dfm["dataset"].astype(str).values
                point_colors = [dataset_color_map.get(ds, "0.25") for ds in ds_series]
                ax.scatter(
                    x_scatter, y_scatter,
                    s=scatter_size, c=point_colors, alpha=scatter_alpha,
                    edgecolors="none", zorder=3
                )
                x_all_points.extend(x_scatter.tolist())
                y_all_points.extend(y_scatter.tolist())

        # (C) Best-fit curve
        stats = results_by_model.get(m, {})
        best_curve = stats.get("fitted_best")
        if best_curve:
            items = sorted(best_curve.items(), key=lambda kv: float(kv[0]))
            B_vals = [float(k) for k, _ in items]
            Y_vals = [float(v) for _, v in items]
            xs = [
                x_positions[all_chunks.index(B)] if B in all_chunks
                else _xpos_for_B(B, all_chunks, x_positions)
                for B in B_vals
            ]
            ax.plot(xs, Y_vals, color=curve_color, linewidth=curve_lw,
                    label=f"Best fit: {stats.get('winner','')}", zorder=4)

        # (D) τ line / CI when post-τ slope (logit) is NEGATIVE
        tau_B = stats.get("tau_batch")
        tau_lo = stats.get("tau_ci_lo")
        tau_hi = stats.get("tau_ci_hi")
        winner = stats.get("winner")
        b0 = float(stats.get("beta0", 0.0) or 0.0)       # pre-τ slope
        bplus = float(stats.get("beta_plus", 0.0) or 0.0) # incremental post-τ slope
        post_slope_logit = (b0 if winner == "FULL" else 0.0) + bplus
        show_tau_line = (
            (winner in ("HINGE", "FULL")) and
            (tau_B is not None and np.isfinite(tau_B))# and
            #(post_slope_logit < 0.0)
        )
        if show_tau_line:
            ax.axvline(
                _xpos_for_B(float(tau_B), all_chunks, x_positions),
                color=tau_line_color, linestyle=tau_line_ls, linewidth=tau_line_lw,
                label=f"τ ≈ {float(tau_B):.1f}", zorder=5
            )
            if (tau_lo is not None and np.isfinite(tau_lo)) and (tau_hi is not None and np.isfinite(tau_hi)):
                xl = _xpos_for_B(float(tau_lo), all_chunks, x_positions)
                xr = _xpos_for_B(float(tau_hi), all_chunks, x_positions)
                ax.axvspan(min(xl, xr), max(xl, xr),
                           color=tau_line_color, alpha=tau_ci_alpha, linewidth=0, zorder=1)


        # --- (E) Axes & annotation ---
        try:
            title = get_model_shorthand(m)
        except Exception:
            title = m
        ax.set_title(title)
        ax.set_xlabel("Batch Size (C)")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(all_chunks)))
        ax.set_xticklabels([str(cs) for cs in all_chunks], rotation=45, ha="center", va="top", fontsize=8)
        ax.grid(axis="y", alpha=0.2)

        lines = []
        crit_name = stats.get("crit_name", "BIC")

        # Existing p-values
        if "p_break" in stats and stats["p_break"] is not None:
            lines.append(f"Break p={float(stats['p_break']):.3f}")
        if "p_pre" in stats and stats["p_pre"] is not None:
            lines.append(f"Pre-slope p={float(stats['p_pre']):.3f}")

        # >>> NEW: slope line (logit scale; per 1 SD in batch size) <<<
        # b0 and bplus were already computed above in this function:
        #   b0     = float(stats.get("beta0", 0.0) or 0.0)
        #   bplus  = float(stats.get("beta_plus", 0.0) or 0.0)
        #   winner = stats.get("winner")

        # Map to direction arrows; small |slope| -> "flat"
        def _dir(v: float, tol: float = 1e-6) -> str:
            return "↑" if v > tol else ("↓" if v < -tol else "→")

        pre_slope_logit = b0 if winner in ("PRE", "FULL") else 0.0
        if winner == "FULL":
            post_slope_logit = b0 + bplus
        elif winner == "HINGE":
            post_slope_logit = bplus
        elif winner == "PRE":
            post_slope_logit = b0
        else:  # FLAT
            post_slope_logit = 0.0

        # --- Cleaner printing by model type ---
        if winner == "PRE":
            lines.append(f"Trend: {_dir(pre_slope_logit)}")
        elif winner == "FLAT":
            lines.append("Trend: → ")
        else:
            # HINGE or FULL: show both sides on one line
            lines.append(f"Trend pre/post: {_dir(pre_slope_logit)} / {_dir(post_slope_logit)}")

        lines.append(f"Best Model: {winner}")


        txt = "\n".join(lines)
        if txt:
            _, pos = _choose_box_corner(ax, x_all_points, y_all_points)
            ax.text(
                pos['x'], pos['y'], txt, transform=ax.transAxes,
                va=pos['va'], ha=pos['ha'], fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.7", boxstyle="round,pad=0.3")
            )


    # Hide unused axes
    for k in range(len(model_names), len(axes)):
        axes[k].set_visible(False)

    # ---------- Figure-level legend: datasets (size-ordered) + curve + τ ----------
    ds_handles = [
        Line2D([0], [0], marker='o', linestyle='', color=dataset_color_map.get(ds, "0.5"),
               label=dataset_display.get(ds, ds), markersize=6)
        for ds in dataset_names
    ]
    curve_handle = Line2D([0], [0], color=curve_color, lw=curve_lw, label="Best fit")
    tau_handle = Line2D([0], [0], color=tau_line_color, lw=tau_line_lw, ls=tau_line_ls, label="τ")
    legend_handles = ds_handles + [curve_handle, tau_handle]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=False,
        title="Datasets",
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1.0])  # leave right margin for legend

    out_path = os.path.join("Figures", f"{output_prefix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[select-best] Plot saved to '{out_path}'")




def replot_model_selection_from_csv(
    summary_csv: str,
    *,
    metrics_by_dataset: Optional[dict] = None,
    fig_prefix: str = "Batch_Modelled",
) -> pd.DataFrame:
    """
    Replot from an existing summary CSV written by run_model_selection_with_bootstrap_and_plot(...),
    without re-running selection or bootstrap.

    Also carries beta0 and beta_plus to enable conditional τ plotting.
    """
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        print("[replot] Empty summary; nothing to plot.")
        return summary_df

    # Rebuild per-model predictors
    fit_map = build_model_fit_results_from_summary(summary_df)  # {model: {predict, tau, ...}}

    # Build x-axis and a minimal df for plotting if possible
    df_min = _build_minimal_df_from_metrics(metrics_by_dataset) if isinstance(metrics_by_dataset, dict) else pd.DataFrame()
    all_chunks = _all_chunks_from_metrics(metrics_by_dataset) if isinstance(metrics_by_dataset, dict) else []

    # Compose results_by_model for the plotting function
    by_model = {}
    for _, r in summary_df.iterrows():
        m = r["model"]
        entry = {
            "winner": r.get("winner"),
            "crit_name": r.get("crit_name", "BIC"),
            "crit_best": r.get("crit_best"),
            "crit_delta_to_2nd": r.get("crit_delta_to_2nd"),
            "tau_batch": r.get("tau_batch"),
            "tau_ci_lo": r.get("tau_ci_lo", None),
            "tau_ci_hi": r.get("tau_ci_hi", None),
            "p_break": r.get("p_break"),
            "p_pre": r.get("p_pre"),
            "deltaF1_post_1sd": r.get("deltaF1_post_1sd"),
            # Carry slopes for τ-visibility logic
            "beta0": r.get("beta0"),
            "beta_plus": r.get("beta_plus"),
        }
        # Best-fit curve from predict() at available batch sizes
        predict_fn = fit_map.get(m, {}).get("predict")
        if callable(predict_fn) and all_chunks:
            entry["fitted_best"] = _make_best_curve_for_model(predict_fn, all_chunks)
        by_model[m] = entry

    _plot_best_fits_by_model(df_min if not df_min.empty else None, by_model, metrics_by_dataset, fig_prefix)
    return summary_df



# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
# --- NEW: top-level worker so ProcessPoolExecutor can pickle it ---
from typing import Optional, Tuple, Literal

def _do_one_model_payload(args) -> Optional[dict]:
    """
    Top-level thin wrapper so it is picklable by ProcessPoolExecutor.
    Args is a tuple to keep submit(...) simple.
    """
    (
        m, dfm, mu, sd, criterion,
        tau_quantiles, n_taus,
        B_break, B_preslope,
        n_jobs_inner, seed
    ) = args

    out = _analyze_one_model(
        dfm,
        mu=mu, sd=sd,
        criterion=criterion,
        tau_quantiles=tau_quantiles,
        n_taus=n_taus,
        B_break=B_break, B_preslope=B_preslope,
        n_jobs_inner=n_jobs_inner,
        seed=seed,
    )
    if out is not None:
        out["model"] = m
    return out


def run_model_selection_with_bootstrap_and_plot(
    analysis_results: Dict[str, Dict[str, Any]],
    *,
    metrics_by_dataset: Optional[dict] = None,
    out_dir: str = "Artifacts/ModelSelection",
    fig_prefix: str = "best_fits_by_model",
    criterion: Literal["bic","aic"] = "bic",
    tau_quantiles: Tuple[float, float] = (0.15, 0.85),
    n_taus: int = 31,
    trim_extremes: bool = True,
    trim_eps: float = 1e-4,
    B_break: int = 1999,
    B_preslope: int = 1999,
    n_jobs_models: Optional[int] = 1,
    n_jobs_inner: int = 1,
    seed: int = 123,
    verbose: bool = True,
    # --- NEW FLAGS ---
    replot_only: bool = False,
    summary_csv: Optional[str] = None,
):

    """
    End-to-end: selection (BIC/AIC), parallel bootstrap tests, and plotting.

    NOTE: Avoid nested pools—prefer either: (a) n_jobs_models>1 & n_jobs_inner=1, or
    (b) n_jobs_models=1 & n_jobs_inner>1.
    """
    # --- NEW: Replot-only path (skip selection & bootstrap) ---
    if replot_only:
        default_csv = "Artifacts/ModelSelection/per_model_best_fit_with_tests.csv"
        csv_path = summary_csv or default_csv
        print(f"[replot] Using summary CSV: {csv_path}")
        return replot_model_selection_from_csv(
            csv_path,
            metrics_by_dataset=metrics_by_dataset,
            fig_prefix=fig_prefix,
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build tidy frame
    df = build_mixed_model_df(analysis_results)
    if df.empty:
        if verbose:
            print("[select-best] No usable runs.")
        return pd.DataFrame()
    df = coerce_categoricals(df)

    # Trim extremes & logit
    if trim_extremes:
        df = df[df["F1"].between(trim_eps, 1 - trim_eps)].copy()
        if df.empty:
            if verbose:
                print("[select-best] All rows trimmed; nothing to do.")
            return pd.DataFrame()
        df["F1_clip"] = df["F1"].clip(trim_eps, 1 - trim_eps)
        df["F1_logit"] = np.log(df["F1_clip"] / (1 - df["F1_clip"]))
    else:
        eps = 1e-6
        df["F1_clip"] = df["F1"].clip(eps, 1 - eps)
        df["F1_logit"] = np.log(df["F1_clip"] / (1 - df["F1_clip"]))

    # Global scaling
    mu = df["batch_size"].mean()
    sd = df["batch_size"].std(ddof=0) or 1.0
    df["BatchSize_sc"] = (df["batch_size"] - mu) / sd

    model_names = sorted(df["model"].unique().tolist())

    # --- remove the nested _do_one(...) ---

    # Pre-split by model to reduce inter-process payload size
    df_by_model = {m: df[df["model"] == m].copy() for m in model_names}

    results: List[dict] = []

    if (n_jobs_models is None) or (int(n_jobs_models) <= 1) or (len(model_names) <= 1):
        # Serial path
        for m in model_names:
            r = _do_one_model_payload((
                m, df_by_model[m], mu, sd, criterion,
                tau_quantiles, n_taus,
                B_break, B_preslope,
                n_jobs_inner, seed
            ))
            if r is not None:
                results.append(r)
                if verbose:
                    print(f"[select-best] {m}: done.")
    else:
        # Parallel across models
        futs = []
        with ProcessPoolExecutor(max_workers=int(n_jobs_models)) as ex:
            for m in model_names:
                args = (
                    m, df_by_model[m], mu, sd, criterion,
                    tau_quantiles, n_taus,
                    B_break, B_preslope,
                    n_jobs_inner, seed
                )
                futs.append(ex.submit(_do_one_model_payload, args))

            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    results.append(r)
                    if verbose:
                        print(f"[select-best] {r['model']}: done (parallel).")

    if not results:
        if verbose:
            print("[select-best] No per-model results.")
        return pd.DataFrame()

    # Build results DataFrame and write CSV
    cols = [
        "model","winner","crit_name","crit_best","crit_delta_to_2nd",
        "tau_sc","tau_batch","tau_ci_lo","tau_ci_hi","tau_iqr",
        "p_break","p_pre","alpha","beta0","beta_plus","mu_B","sigma_B","dataset_fe",
        "deltaF1_post_1sd",
    ]
    df_out = (pd.DataFrame(results)
                .reindex(columns=[c for c in cols if c in pd.DataFrame(results).columns])
                .sort_values("model")
                .reset_index(drop=True))
    out_csv = os.path.join(out_dir, "per_model_best_fit_with_tests.csv")
    df_out.to_csv(out_csv, index=False)
    if verbose:
        print(f"[select-best] Wrote model-selection+tests summary to {out_csv}")

    # Build a dict for plotting
    by_model = {}
    for i, r in enumerate(results):
        entry = {
            "winner": r["winner"],
            "crit_name": r["crit_name"],
            "crit_best": r["crit_best"],
            "crit_delta_to_2nd": r["crit_delta_to_2nd"],
            "tau_batch": r.get("tau_batch", None),
            "tau_ci_lo": r.get("tau_ci_lo"),
            "tau_ci_hi": r.get("tau_ci_hi"),
            "p_break": r.get("p_break", None),
            "p_pre": r.get("p_pre", None),
            "deltaF1_post_1sd": r.get("deltaF1_post_1sd", None),
            # NEW: provide slopes to plotting layer
            "beta0": r.get("beta0", None),
            "beta_plus": r.get("beta_plus", None),
            # fitted curve
            "fitted_best": r.get("fitted_best", None),
        }
        by_model[r["model"]] = entry
    _plot_best_fits_by_model(df, by_model, metrics_by_dataset, fig_prefix,
                             tau_line_color="#1f77b4", tau_ci_alpha=0.20, tau_line_lw=2.0)


    return df_out



# ---------------------------------------------------------------------------
# __main__ (example)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example skeleton; replace with your actual data objects.
    # analysis_results = {...}
    # metrics_by_dataset = {...} or None
    print("This module provides `run_model_selection_with_bootstrap_and_plot(...)`.\n"
          "Import and call it with your `analysis_results` (and optionally `metrics_by_dataset`).")
