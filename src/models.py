# src/models.py
"""
Training pipeline: LinearRegression (baseline) vs LightGBM.
Si LightGBM no le gana a la regresión lineal, el modelo complejo no sirve.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("[Models] lightgbm not installed – only LinearRegression will run.")

from src.features import get_feature_lists

RANDOM_STATE  = 42
N_CV_SPLITS   = 4
MIN_ROWS      = 30    # minimum weekly rows per product to train


# ── Preprocessor factory ─────────────────────────────────────────────────────
def _build_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    transformers = []
    if cat_features:
        transformers.append(
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features)
        )
    if num_features:
        transformers.append(("num", "passthrough", num_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Model definitions ─────────────────────────────────────────────────────────
def _get_models(num_features: list[str], cat_features: list[str]) -> dict[str, Pipeline]:
    pre = _build_preprocessor(num_features, cat_features)

    models: dict[str, Pipeline] = {
        "LinearRegression": Pipeline([
            ("pre",    pre),
            ("scaler", StandardScaler()),
            ("est",    LinearRegression()),
        ])
    }

    if _HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("pre", _build_preprocessor(num_features, cat_features)),
            ("est", lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=RANDOM_STATE,
                verbose=-1,
            )),
        ])

    return models


# ── Per-product training ──────────────────────────────────────────────────────
def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}


def train_product(
    df_ml: pd.DataFrame,
    product: str,
    num_features: list[str],
    cat_features: list[str],
) -> dict | None:
    sub = df_ml[df_ml["product_code"] == product].sort_values("date").reset_index(drop=True)
    n   = len(sub)
    if n < MIN_ROWS:
        print(f"  [Models] {product}: only {n} rows – skipping (need ≥ {MIN_ROWS}).")
        return None

    X = sub[num_features + cat_features].copy()
    y = sub["units"].values.astype(float)

    # fill cat NaNs
    for c in cat_features:
        X[c] = X[c].fillna("__NA__").astype(str)

    # temporal train/test split (last 20 %)
    split = int(0.8 * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    tscv    = TimeSeriesSplit(n_splits=min(N_CV_SPLITS, max(2, split // 8)))
    models  = _get_models(num_features, cat_features)

    product_results: list[dict] = []
    best_test_rmse  = np.inf
    best_pipe       = None
    best_model_name = None

    for name, pipe in models.items():
        try:
            # Cross-val on train set
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=tscv, scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse = float(np.sqrt(-cv_scores.mean()))

            # Refit on full train, evaluate on test
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            test_metrics = _eval_metrics(y_test, y_pred)

            print(
                f"  {product} | {name:20s} | "
                f"CV-RMSE={cv_rmse:.2f}  "
                f"Test-RMSE={test_metrics['RMSE']:.2f}  "
                f"R2={test_metrics['R2']:.3f}"
            )

            row = {
                "product":    product,
                "model":      name,
                "n_rows":     n,
                "cv_rmse":    round(cv_rmse, 3),
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "estimator":  pipe,
            }
            product_results.append(row)

            if test_metrics["RMSE"] < best_test_rmse:
                best_test_rmse  = test_metrics["RMSE"]
                best_pipe       = pipe
                best_model_name = name

        except Exception as e:
            print(f"  [Models] {product}/{name} failed: {e}")

    if not product_results:
        return None

    # winner annotation
    for r in product_results:
        r["is_best"] = r["model"] == best_model_name

    return {
        "product":          product,
        "best_model":       best_model_name,
        "best_pipe":        best_pipe,
        "comparison":       product_results,
        "num_features":     num_features,
        "cat_features":     cat_features,
    }


def train_all_products(
    df_ml: pd.DataFrame,
    artifacts_path: str | Path = "artifacts",
) -> dict[str, dict]:
    artifacts_path = Path(artifacts_path)
    artifacts_path.mkdir(exist_ok=True, parents=True)

    num_features, cat_features = get_feature_lists(df_ml)
    print(f"[Models] num_features: {num_features}")
    print(f"[Models] cat_features: {cat_features}")

    products = sorted(df_ml["product_code"].dropna().unique())
    all_results: dict[str, dict] = {}

    for prod in products:
        print(f"\n── {prod} ──────────────────────────────────")
        result = train_product(df_ml, prod, num_features, cat_features)
        if result is None:
            continue
        all_results[prod] = result

        # persist best pipeline
        pipe_path = artifacts_path / f"pipeline_{prod}.joblib"
        joblib.dump(result["best_pipe"], pipe_path)

    # save summary CSV
    rows = [row for r in all_results.values() for row in r["comparison"]]
    summary = (
        pd.DataFrame([{k: v for k, v in row.items() if k != "estimator"} for row in rows])
        .sort_values(["product", "test_RMSE"])
        .reset_index(drop=True)
    )
    summary.to_csv(artifacts_path / "model_comparison.csv", index=False)
    print(f"\n[Models] Done. {len(all_results)} products trained. Summary → artifacts/model_comparison.csv")
    return all_results


def load_pipeline(product: str, artifacts_path: str | Path = "artifacts") -> Pipeline | None:
    p = Path(artifacts_path) / f"pipeline_{product}.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


def load_summary(artifacts_path: str | Path = "artifacts") -> pd.DataFrame | None:
    p = Path(artifacts_path) / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)
