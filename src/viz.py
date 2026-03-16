# src/viz.py
"""
Reusable Matplotlib/Seaborn charts.
All functions return (fig, axes) so they can be used in notebooks AND Streamlit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
_PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]


# ── 1. Demand & TR/Profit curves ─────────────────────────────────────────────
def plot_demand_and_objective(
    p_grid: np.ndarray,
    q_pred: np.ndarray,
    objective: np.ndarray,
    optimal_idx: int,
    elasticity: np.ndarray,
    product: str,
    zone: str,
    obj_label: str = "Revenue",
    current_price: float | None = None,
) -> tuple:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    p_opt = p_grid[optimal_idx]
    q_opt = q_pred[optimal_idx]
    obj_opt = objective[optimal_idx]

    # ── demand curve
    ax = axes[0]
    ax.plot(p_grid, q_pred, color=_PALETTE[0], lw=2)
    ax.scatter([p_opt], [q_opt], color=_PALETTE[2], zorder=6, s=80, label=f"Óptimo: ${p_opt:.2f}")
    if current_price:
        ax.axvline(current_price, color="gray", ls="--", lw=1.5, label=f"Precio actual: ${current_price:.2f}")
    ax.set_title(f"Curva de Demanda\n{product} – {zone}")
    ax.set_xlabel("Precio"); ax.set_ylabel("Cantidad estimada")
    ax.legend(fontsize=9)

    # ── objective curve
    ax = axes[1]
    ax.plot(p_grid, objective, color=_PALETTE[1], lw=2)
    ax.scatter([p_opt], [obj_opt], color=_PALETTE[2], zorder=6, s=80,
               label=f"Máx {obj_label}: ${obj_opt:,.0f}")
    if current_price:
        ax.axvline(current_price, color="gray", ls="--", lw=1.5)
    ax.set_title(f"{obj_label} total\n{product} – {zone}")
    ax.set_xlabel("Precio"); ax.set_ylabel(obj_label)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)

    # ── elasticity curve
    ax = axes[2]
    ax.plot(p_grid, elasticity, color=_PALETTE[4], lw=2)
    ax.axhline(-1, color="gray", ls="--", lw=1, label="E = -1 (unitaria)")
    ax.axhline(0, color="black", lw=0.5)
    ax.scatter([p_opt], [elasticity[optimal_idx]], color=_PALETTE[2], zorder=6, s=80,
               label=f"E óptimo: {elasticity[optimal_idx]:.2f}")
    ax.set_title(f"Elasticidad precio\n{product} – {zone}")
    ax.set_xlabel("Precio"); ax.set_ylabel("Elasticidad")
    ax.legend(fontsize=9)

    fig.suptitle(f"Análisis de precio óptimo — {product} | Zona: {zone}", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig, axes


# ── 2. Model comparison bar chart ────────────────────────────────────────────
def plot_model_comparison(summary_df: pd.DataFrame) -> tuple:
    df = summary_df.copy()
    products = sorted(df["product"].unique())
    n = len(products)
    fig, axes = plt.subplots(1, min(n, 3), figsize=(5 * min(n, 3), 4), squeeze=False)
    axes = axes.flatten()

    for i, prod in enumerate(products[:3]):
        sub = df[df["product"] == prod].sort_values("test_RMSE")
        colors = [_PALETTE[2] if r["is_best"] else _PALETTE[0] for _, r in sub.iterrows()]
        bars = axes[i].barh(sub["model"], sub["test_RMSE"], color=colors)
        axes[i].set_title(f"{prod}\n(rojo = ganador)")
        axes[i].set_xlabel("Test RMSE (↓ mejor)")
        # annotate R²
        for bar, (_, row) in zip(bars, sub.iterrows()):
            axes[i].text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                         f"R²={row['test_R2']:.2f}", va="center", fontsize=9)

    for j in range(len(products), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Comparación de modelos por producto (Test set)", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ── 3. Elasticity heatmap ────────────────────────────────────────────────────
def plot_elasticity_heatmap(opt_df: pd.DataFrame) -> tuple:
    pivot = opt_df.pivot(index="zone", columns="product", values="elasticity")
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(4, len(pivot) * 0.9)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=-1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Elasticidad precio"})
    ax.set_title("Elasticidad precio por producto × zona\n(< -1 elástico, > -1 inelástico)", fontsize=12)
    fig.tight_layout()
    return fig, ax


# ── 4. Price delta heatmap (recommended vs current) ──────────────────────────
def plot_price_delta_heatmap(opt_df: pd.DataFrame) -> tuple:
    if "price_delta_pct" not in opt_df.columns:
        return None, None
    pivot = opt_df.pivot(index="zone", columns="product", values="price_delta_pct")
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4), max(4, len(pivot) * 0.9)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "% cambio recomendado"})
    ax.set_title("Ajuste de precio recomendado vs actual (%)\n(verde = subir, rojo = bajar)", fontsize=12)
    fig.tight_layout()
    return fig, ax


# ── 5. Sales overview ────────────────────────────────────────────────────────
def plot_sales_overview(df_ml: pd.DataFrame, top_n: int = 6) -> tuple:
    top_prods = (
        df_ml.groupby("product_code")["units"]
        .sum().sort_values(ascending=False)
        .head(top_n).index.tolist()
    )
    sub = df_ml[df_ml["product_code"].isin(top_prods)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # units by product × zone
    agg = sub.groupby(["product_code", "zone"])["units"].sum().reset_index()
    sns.barplot(data=agg, x="product_code", y="units", hue="zone",
                palette=_PALETTE, ax=axes[0])
    axes[0].set_title(f"Top {top_n} productos — Unidades por zona")
    axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=30)

    # rolling weekly units for top 3 products
    top3 = top_prods[:3]
    for k, prod in enumerate(top3):
        ts = (
            df_ml[df_ml["product_code"] == prod]
            .set_index("date")["units"]
            .resample("W").sum()
            .rolling(4, center=True).mean()
        )
        axes[1].plot(ts.index, ts.values, label=prod, color=_PALETTE[k], lw=1.8)
    axes[1].set_title("Tendencia semanal — top 3 productos (media móvil 4 sem.)")
    axes[1].set_ylabel("Unidades"); axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig, axes
