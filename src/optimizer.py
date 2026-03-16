# src/optimizer.py
"""
Demand curves, elasticity, and price optimization.

Objective: maximize PROFIT = (price - unit_cost) * Q(price)
If no unit_cost available, falls back to maximizing Revenue.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.features import get_feature_lists

N_PRICE_POINTS = 200
PRICE_MARGIN   = 0.15   # extend price grid ±15% beyond observed range


# ── Build synthetic X for price grid ─────────────────────────────────────────
def build_price_grid_X(
    df_ml: pd.DataFrame,
    product: str,
    zone: str,
    p_grid: np.ndarray,
    num_features: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    """
    Create a DataFrame with N_PRICE_POINTS rows, sweeping price while
    holding every other feature at its product×zone median/mode.
    """
    sub = df_ml[(df_ml["product_code"] == product) & (df_ml["zone"] == zone)]
    if len(sub) < 3:
        sub = df_ml[df_ml["product_code"] == product]
    if len(sub) < 3:
        sub = df_ml.copy()

    # numeric medians (exclude price itself)
    base_num: dict[str, float] = {}
    for f in num_features:
        if f == "unit_price_mean":
            continue
        val = sub[f].median(skipna=True) if f in sub.columns else 0.0
        base_num[f] = 0.0 if (math.isnan(val) if isinstance(val, float) else False) else float(val)

    # categorical modes
    base_cat: dict[str, str] = {}
    for f in cat_features:
        if f == "zone":
            base_cat[f] = zone
        elif f in sub.columns:
            mode = sub[f].mode()
            base_cat[f] = str(mode.iloc[0]) if not mode.empty else "__NA__"
        else:
            base_cat[f] = "__NA__"

    rows = []
    for p in p_grid:
        row: dict = {"unit_price_mean": float(p)}
        row.update(base_num)
        row.update(base_cat)
        rows.append(row)

    cols = [f for f in (num_features + cat_features) if f in (list(base_num) + list(base_cat) + ["unit_price_mean"])]
    Xgrid = pd.DataFrame(rows)
    # ensure column order matches what pipeline expects
    available = [c for c in (num_features + cat_features) if c in Xgrid.columns]
    return Xgrid[available]


# ── Point elasticity ──────────────────────────────────────────────────────────
def compute_elasticity(p_grid: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """
    Arc elasticity: E = (dQ/dP) * (P/Q)
    Computed numerically across the price grid.
    """
    dQ_dP = np.gradient(q_pred, p_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        elast = np.where(q_pred != 0, dQ_dP * (p_grid / q_pred), np.nan)
    return elast


# ── Optimal price ─────────────────────────────────────────────────────────────
def find_optimal_price(
    p_grid: np.ndarray,
    q_pred: np.ndarray,
    unit_cost: float | None = None,
) -> dict:
    """
    Returns dict with optimal_price, optimal_qty, max_objective,
    objective_label, and elasticity_at_optimum.
    """
    q_pred = np.clip(q_pred, 0, None)   # no negative demand

    if unit_cost is not None and not math.isnan(unit_cost) and unit_cost > 0:
        objective   = (p_grid - unit_cost) * q_pred
        obj_label   = "Profit"
    else:
        objective   = p_grid * q_pred
        obj_label   = "Revenue (no cost data)"

    idx    = int(np.nanargmax(objective))
    elast  = compute_elasticity(p_grid, q_pred)

    return {
        "optimal_price":        float(p_grid[idx]),
        "optimal_qty":          float(q_pred[idx]),
        "max_objective":        float(objective[idx]),
        "objective_label":      obj_label,
        "elasticity_at_optimum": float(elast[idx]) if not np.isnan(elast[idx]) else None,
        "revenue_at_optimum":   float(p_grid[idx] * q_pred[idx]),
        "p_grid":               p_grid,
        "q_pred":               q_pred,
        "objective_curve":      objective,
        "elasticity_curve":     elast,
        "optimal_idx":          idx,
    }


# ── Full optimization table ───────────────────────────────────────────────────
def optimize_all(
    df_ml: pd.DataFrame,
    all_results: dict[str, dict],
    artifacts_path: str | Path = "artifacts",
) -> pd.DataFrame:
    """
    For each product × zone, run price optimization with the best model.
    Returns a summary DataFrame saved to artifacts/price_optimization.csv.
    """
    artifacts_path = Path(artifacts_path)
    num_features, cat_features = get_feature_lists(df_ml)
    zones    = sorted(df_ml["zone"].dropna().unique())
    products = sorted(all_results.keys())

    rows = []
    for prod in products:
        result   = all_results[prod]
        pipe     = result["best_pipe"]
        best_mod = result["best_model"]

        # unit_cost: take mean from df_ml if available
        unit_cost = None
        if "unit_cost" in df_ml.columns:
            uc = df_ml.loc[df_ml["product_code"] == prod, "unit_cost"].dropna()
            if not uc.empty:
                unit_cost = float(uc.mean())

        prod_prices = df_ml.loc[df_ml["product_code"] == prod, "unit_price_mean"].dropna()
        if prod_prices.empty:
            continue
        pmin   = float(prod_prices.min() * (1 - PRICE_MARGIN))
        pmax   = float(prod_prices.max() * (1 + PRICE_MARGIN))
        p_grid = np.linspace(pmin, pmax, N_PRICE_POINTS)

        for zone in zones:
            try:
                Xgrid  = build_price_grid_X(df_ml, prod, zone, p_grid, num_features, cat_features)
                q_pred = pipe.predict(Xgrid)
                opt    = find_optimal_price(p_grid, q_pred, unit_cost)

                # current mean price for this product×zone
                mask         = (df_ml["product_code"] == prod) & (df_ml["zone"] == zone)
                current_mean = float(df_ml.loc[mask, "unit_price_mean"].mean()) if mask.any() else np.nan

                rows.append({
                    "product":            prod,
                    "zone":               zone,
                    "best_model":         best_mod,
                    "current_price_mean": round(current_mean, 2) if not math.isnan(current_mean) else None,
                    "optimal_price":      round(opt["optimal_price"], 2),
                    "price_delta_pct":    round((opt["optimal_price"] / current_mean - 1) * 100, 1) if current_mean else None,
                    "optimal_qty":        round(opt["optimal_qty"], 1),
                    "max_objective":      round(opt["max_objective"], 2),
                    "objective_label":    opt["objective_label"],
                    "elasticity":         round(opt["elasticity_at_optimum"], 3) if opt["elasticity_at_optimum"] is not None else None,
                    "unit_cost":          round(unit_cost, 2) if unit_cost else None,
                })
            except Exception as e:
                print(f"  [Optimizer] {prod} × {zone}: {e}")

    opt_df = pd.DataFrame(rows).sort_values(["product", "zone"]).reset_index(drop=True)
    opt_df.to_csv(artifacts_path / "price_optimization.csv", index=False)
    print(f"[Optimizer] Done – {len(opt_df)} product×zone combos → artifacts/price_optimization.csv")
    return opt_df


def load_optimization(artifacts_path: str | Path = "artifacts") -> pd.DataFrame | None:
    p = Path(artifacts_path) / "price_optimization.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)
