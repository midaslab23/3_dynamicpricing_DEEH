# app.py
"""
Streamlit Pricing Dashboard
Ejecutar: streamlit run app.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.etl       import load_sales, load_masters, build_canonical
from src.features  import build_weekly_ml, get_feature_lists
from src.models    import train_all_products, load_pipeline, load_summary
from src.optimizer import (
    optimize_all, load_optimization,
    build_price_grid_X, find_optimal_price, compute_elasticity,
)
from src.viz import (
    plot_demand_and_objective, plot_model_comparison,
    plot_elasticity_heatmap, plot_price_delta_heatmap, plot_sales_overview,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pricing Optimizer | Heladería",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS = Path("artifacts")
DATA_PATH = Path("data")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F0F9FF;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-card-green  { border-left-color: #16A34A; background: #F0FDF4; }
    .metric-card-red    { border-left-color: #DC2626; background: #FFF1F2; }
    .metric-card-yellow { border-left-color: #D97706; background: #FFFBEB; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #1E3A5F; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos de ventas…")
def get_df_ml() -> pd.DataFrame | None:
    parquet_path = ARTIFACTS / "df_ml.parquet"

    # Load from parquet only if it exists AND has unit_cost (not stale)
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "unit_cost" in df.columns and df["unit_cost"].notna().any():
            return df
        # parquet is stale — fall through to rebuild

    # Try to rebuild from raw data
    candidates = ["source", "data", "ventas", "sales"]
    raw_path = None
    for name in candidates:
        p = Path(name)
        if p.exists() and list(p.glob("sales_*.xlsx")):
            raw_path = p
            break

    if raw_path is None:
        # Return stale parquet anyway if nothing else available
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return None

    sales = load_sales(str(raw_path))
    df_products, df_stores = load_masters(str(raw_path))
    df_canonical = build_canonical(sales, df_products, df_stores)
    df_ml = build_weekly_ml(df_canonical)
    ARTIFACTS.mkdir(exist_ok=True)
    df_ml.to_parquet(parquet_path, index=False)
    return df_ml


@st.cache_data(show_spinner="Cargando artefactos de modelos…")
def get_summary() -> pd.DataFrame | None:
    return load_summary(ARTIFACTS)


@st.cache_data(show_spinner="Cargando tabla de precios óptimos…")
def get_opt_df() -> pd.DataFrame | None:
    return load_optimization(ARTIFACTS)


# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar() -> str:
    st.sidebar.image("https://img.icons8.com/emoji/96/ice-cream-emoji.png", width=80)
    st.sidebar.title("🍦 Pricing ML")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navegar a",
        ["🏠 Dashboard", "🔍 Optimizador de Precio", "🗂️ Explorador de Datos", "📊 Comparación de Modelos"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Elaborado por:**<br>
**Diego Eduardo Enríquez Hernández**<br><br>
<a href="https://www.linkedin.com/in/diegoeduardoenriquezhernandez/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="26" style="vertical-align:middle"/>
    &nbsp; Ver perfil en LinkedIn
</a><br><br>
**Economics · ML/Data Science · Finance**
""", unsafe_allow_html=True)
    return page


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Dashboard
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard(df_ml: pd.DataFrame, opt_df: pd.DataFrame | None) -> None:
    st.title("🏠 Heladería Dashboard Principal")

    # ── Compute financials ────────────────────────────────────────
    total_rev   = df_ml["revenue"].sum()
    total_units = int(df_ml["units"].sum())
    n_products  = df_ml["product_code"].nunique()
    n_stores    = df_ml["store_code"].nunique()

    # cost & profit — only if unit_cost is available
    has_cost = "unit_cost" in df_ml.columns and df_ml["unit_cost"].notna().any()
    if has_cost:
        df_ml = df_ml.copy()
        df_ml["cost_total"]  = df_ml["unit_cost"] * df_ml["units"]
        df_ml["profit"]      = df_ml["revenue"] - df_ml["cost_total"]
        total_cost   = df_ml["cost_total"].sum()
        total_profit = df_ml["profit"].sum()
        margin_pct   = total_profit / total_rev * 100 if total_rev > 0 else 0
    else:
        total_cost = total_profit = margin_pct = None

    # ── KPI row 1: financials ─────────────────────────────────────
    if has_cost:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Revenue total",     f"${total_rev:,.0f}")
        c2.metric("💸 Costo total",       f"${total_cost:,.0f}")
        c3.metric("📈 Profit total",      f"${total_profit:,.0f}")
        c4.metric("🎯 Margen bruto",      f"{margin_pct:.1f}%")
    else:
        c1, c2 = st.columns(2)
        c1.metric("💰 Revenue total",     f"${total_rev:,.0f}")
        c2.metric("📦 Unidades totales",  f"{total_units:,}")
        st.caption("💡 Agrega `unit_cost` en `products.xlsx` para ver profit y margen.")

    # ── KPI row 2: operacionales ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Unidades totales",  f"{total_units:,}")
    c2.metric("💲 Precio prom.",      f"${df_ml['unit_price_mean'].mean():.2f}")
    c3.metric("🛒 Productos",         n_products)
    c4.metric("🏪 Tiendas",           n_stores)

    st.markdown("---")

    # ── Sales overview chart ──────────────────────────────────────
    fig, _ = plot_sales_overview(df_ml)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Optimization summary table ────────────────────────────────
    if opt_df is not None:
        st.markdown("---")
        st.subheader("📋 Resumen de precios óptimos")

        def _color_delta(val):
            if pd.isna(val):
                return ""
            color = "#16A34A" if val > 0 else "#DC2626"
            return f"color: {color}; font-weight: bold"

        display_cols = [
            "product", "zone", "current_price_mean", "optimal_price",
            "price_delta_pct", "elasticity", "objective_label",
        ]
        present = [c for c in display_cols if c in opt_df.columns]
        styled = (
            opt_df[present]
            .style
            .applymap(_color_delta, subset=["price_delta_pct"] if "price_delta_pct" in present else [])
            .format({
                "current_price_mean": "${:.2f}",
                "optimal_price":      "${:.2f}",
                "price_delta_pct":    "{:+.1f}%",
                "elasticity":         "{:.3f}",
            }, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, height=320)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌡️ Mapa de elasticidad")
            fig2, _ = plot_elasticity_heatmap(opt_df)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        with col2:
            st.subheader("💡 Ajuste recomendado de precio (%)")
            fig3, _ = plot_price_delta_heatmap(opt_df)
            if fig3:
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Price Optimizer (interactive)
# ════════════════════════════════════════════════════════════════════════════
def page_optimizer(df_ml: pd.DataFrame) -> None:
    st.title("🔍 Optimizador de Precio")
    st.markdown("Selecciona un producto y zona para ver la curva de demanda, ingreso/profit óptimo y elasticidad en tiempo real.")

    num_features, cat_features = get_feature_lists(df_ml)
    products = sorted(df_ml["product_code"].dropna().unique())
    zones    = sorted(df_ml["zone"].dropna().unique())

    # ── Controls ──────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        product = st.selectbox("🛒 Producto", products)
    with col_ctrl2:
        zone    = st.selectbox("📍 Zona", zones)
    with col_ctrl3:
        pipe = load_pipeline(product, ARTIFACTS)
        if pipe is None:
            st.warning("⚠️ No hay modelo entrenado para este producto. Ejecuta el pipeline primero.")
            return
        st.success(f"✅ Modelo cargado")

    # price range slider
    prod_prices = df_ml.loc[df_ml["product_code"] == product, "unit_price_mean"].dropna()
    pmin  = float(prod_prices.min() * 0.85)
    pmax  = float(prod_prices.max() * 1.25)
    p_range = st.slider(
        "Rango de precios a explorar ($)",
        min_value=float(round(pmin, 1)),
        max_value=float(round(pmax, 1)),
        value=(float(round(pmin, 1)), float(round(pmax, 1))),
        step=0.5,
    )

    # unit cost input
    default_cost = None
    if "unit_cost" in df_ml.columns:
        uc = df_ml.loc[df_ml["product_code"] == product, "unit_cost"].dropna()
        default_cost = float(uc.mean()) if not uc.empty else None
    unit_cost = st.number_input(
        "💲 Costo unitario ($)  — opcional, si no se captura se maximiza revenue",
        min_value=0.0, max_value=500.0,
        value=float(round(default_cost, 2)) if default_cost else 0.0,
        step=0.5,
    )
    if unit_cost == 0.0:
        unit_cost = None

    # ── Compute ───────────────────────────────────────────────────
    p_grid = np.linspace(p_range[0], p_range[1], 200)
    try:
        Xgrid  = build_price_grid_X(df_ml, product, zone, p_grid, num_features, cat_features)
        q_pred = pipe.predict(Xgrid)
        q_pred = np.clip(q_pred, 0, None)
        opt    = find_optimal_price(p_grid, q_pred, unit_cost)
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        return

    current_mean = float(
        df_ml.loc[(df_ml["product_code"] == product) & (df_ml["zone"] == zone), "unit_price_mean"].mean()
    )

    # ── Result metrics ────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Precio actual promedio",  f"${current_mean:.2f}")
    m2.metric("🎯 Precio óptimo",          f"${opt['optimal_price']:.2f}",
              delta=f"{(opt['optimal_price']/current_mean - 1)*100:+.1f}%")
    m3.metric("📦 Cantidad estimada",      f"{opt['optimal_qty']:.0f}")
    m4.metric(f"📈 {opt['objective_label']} máx.", f"${opt['max_objective']:,.0f}")
    e_val = opt["elasticity_at_optimum"]
    e_label = (
        "🔴 Elástico (sensible)" if e_val is not None and e_val < -1
        else "🟡 Inelástico" if e_val is not None
        else "—"
    )
    m5.metric("⚡ Elasticidad en óptimo", f"{e_val:.3f}" if e_val else "—", delta=e_label, delta_color="off")

    # ── Interpretation ────────────────────────────────────────────
    if e_val is not None:
        if e_val < -1:
            st.info(f"📊 **Elasticidad = {e_val:.2f}**: La demanda en esta zona es **elástica** — un aumento de 1% en precio reduce las ventas en {abs(e_val):.1f}%. Podría considerarse bajar el precio para crecer en volumen.")
        elif e_val > -1:
            st.info(f"📊 **Elasticidad = {e_val:.2f}**: La demanda es **inelástica** — los clientes en {zone} son poco sensibles al precio. Existe margen para subir sin perder mucho volumen.")

    # ── Charts ────────────────────────────────────────────────────
    fig, _ = plot_demand_and_objective(
        p_grid=p_grid,
        q_pred=q_pred,
        objective=opt["objective_curve"],
        optimal_idx=opt["optimal_idx"],
        elasticity=opt["elasticity_curve"],
        product=product,
        zone=zone,
        obj_label=opt["objective_label"],
        current_price=current_mean,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Raw data table ────────────────────────────────────────────
    with st.expander("🔢 Ver tabla de datos (precio → demanda → objetivo)"):
        step = max(1, len(p_grid) // 40)
        tbl = pd.DataFrame({
            "Precio ($)":        p_grid[::step].round(2),
            "Cantidad estimada": q_pred[::step].round(1),
            opt["objective_label"]: opt["objective_curve"][::step].round(2),
            "Elasticidad":       opt["elasticity_curve"][::step].round(3),
        })
        st.dataframe(tbl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3: Model Comparison + ML Insights
# ════════════════════════════════════════════════════════════════════════════
def page_model_comparison(summary: pd.DataFrame, df_ml: pd.DataFrame | None = None) -> None:
    st.title("📊 Comparación de Modelos y ML Insights")
    st.markdown("Regresión Lineal vs LightGBM — si el modelo complejo no le gana al lineal, no sirve de extra.")

    tab1, tab2, tab3 = st.tabs(["🏆 Comparación de métricas", "🌳 Cómo decide LightGBM", "📉 Regresión lineal vs ML"])

    # ── TAB 1: Metrics comparison ─────────────────────────────────
    with tab1:
        st.dataframe(
            summary.drop(columns=["is_best"], errors="ignore"),
            use_container_width=True, height=320,
        )
        st.markdown("---")
        fig, _ = plot_model_comparison(summary)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("🏆 Veredicto por producto")
        best_rows = (
            summary[summary["is_best"]] if "is_best" in summary.columns
            else summary.loc[summary.groupby("product")["test_RMSE"].idxmin()]
        )
        lgbm_wins = (best_rows["model"] == "LightGBM").sum()
        lr_wins   = (best_rows["model"] == "LinearRegression").sum()

        c1, c2 = st.columns(2)
        c1.metric("⚡ Victorias LightGBM",        lgbm_wins)
        c2.metric("📏 Victorias Regresión Lineal", lr_wins)

        if lr_wins > 0:
            st.warning(f"⚠️ La regresión lineal ganó en {lr_wins} producto(s). Considera agregar más features o más datos históricos para que LightGBM pueda aprender mejor.")

        for _, row in best_rows.iterrows():
            is_lgbm = row["model"] == "LightGBM"
            col_cls = "metric-card-green" if is_lgbm else "metric-card-yellow"
            icon    = "⚡" if is_lgbm else "⚠️"
            st.markdown(
                f'<div class="metric-card {col_cls}">'
                f'<b>{row["product"]}</b> → {row["model"]} | '
                f'RMSE={row["test_RMSE"]:.2f} | MAE={row["test_MAE"]:.2f} | R²={row["test_R2"]:.3f} | {icon}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── TAB 2: Feature importance & what LightGBM learned ────────
    with tab2:
        st.markdown("""
        ### ¿Cómo decide LightGBM la demanda estimada?

        LightGBM construye ~400 árboles de decisión en secuencia.
        Cada árbol aprende a corregir los errores del anterior (*boosting*).
        La **importancia de features** mide cuántas veces cada variable
        fue usada para hacer una división en todos los árboles.
        """)

        if df_ml is None:
            st.info("Carga los datos para ver importancia de features.")
        else:
            # Load LightGBM pipelines and extract feature importances
            from src.features import get_feature_lists
            from src.models   import load_pipeline

            num_features, cat_features = get_feature_lists(df_ml)
            products = sorted(summary["product"].unique())

            sel_prod = st.selectbox("Producto para análisis", products, key="fi_prod")
            pipe = load_pipeline(sel_prod, ARTIFACTS)

            if pipe is None:
                st.warning("No se encontró pipeline para este producto. Corre el pipeline primero.")
            else:
                # Check if it's LightGBM
                estimator = pipe.named_steps.get("est")
                is_lgbm   = hasattr(estimator, "feature_importances_")

                if is_lgbm:
                    # Get feature names after OHE
                    pre = pipe.named_steps["pre"]
                    try:
                        ohe_names = list(pre.named_transformers_["ohe"].get_feature_names_out(cat_features))
                    except Exception:
                        ohe_names = []
                    feat_names = ohe_names + num_features
                    importances = estimator.feature_importances_

                    # Trim to matching length
                    n = min(len(feat_names), len(importances))
                    feat_names   = feat_names[:n]
                    importances  = importances[:n]

                    # Aggregate OHE dummies back to their source feature
                    agg_imp: dict[str, float] = {}
                    for name, imp in zip(feat_names, importances):
                        # OHE features look like "zone__Norte", "store_code__NLE-ALL-01"
                        base = name.split("__")[0] if "__" in name else name
                        agg_imp[base] = agg_imp.get(base, 0) + imp

                    imp_df = (
                        pd.DataFrame({"feature": list(agg_imp), "importance": list(agg_imp.values())})
                        .sort_values("importance", ascending=True)
                    )

                    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.45)))
                    colors = ["#2563EB" if imp_df["feature"].iloc[i] == "unit_price_mean"
                              else "#16A34A" if imp_df["importance"].iloc[i] == imp_df["importance"].max()
                              else "#94A3B8" for i in range(len(imp_df))]
                    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
                    ax.set_xlabel("Importancia (splits en todos los árboles)")
                    ax.set_title(f"Importancia de features — {sel_prod} (LightGBM)")
                    # Annotate the price bar
                    price_row = imp_df[imp_df["feature"] == "unit_price_mean"]
                    if not price_row.empty:
                        pct = price_row["importance"].iloc[0] / imp_df["importance"].sum() * 100
                        ax.set_xlabel(f"Importancia — el precio explica el {pct:.0f}% de las decisiones")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.markdown("""
                    **Cómo leer este gráfico:**
                    - 🔵 **`unit_price_mean`** — cuánto pesa el precio en las predicciones
                    - 🟢 **Feature más importante** — el mayor driver de demanda
                    - Si el precio NO está entre los top 3, el modelo encontró que otros factores
                      (temperatura, quincena, historial) importan más que el precio mismo
                    """)

                else:
                    st.info(f"El mejor modelo para {sel_prod} es Regresión Lineal — no hay importancia de features de árbol. Ve a la pestaña de comparación para ver por qué.")

                # Partial dependence: price effect holding others constant
                st.markdown("---")
                st.subheader(f"📐 Efecto parcial del precio — {sel_prod}")
                st.markdown("Cómo cambia la demanda predicha al variar **solo el precio**, manteniendo todo lo demás constante en su mediana.")

                from src.optimizer import build_price_grid_X
                zones = sorted(df_ml["zone"].dropna().unique())

                fig2, ax2 = plt.subplots(figsize=(9, 4))
                colors_z = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]
                for i, z in enumerate(zones):
                    prod_prices = df_ml.loc[df_ml["product_code"] == sel_prod, "unit_price_mean"].dropna()
                    if prod_prices.empty:
                        continue
                    p_grid = np.linspace(float(prod_prices.min()*0.85), float(prod_prices.max()*1.15), 150)
                    try:
                        Xg     = build_price_grid_X(df_ml, sel_prod, z, p_grid, num_features, cat_features)
                        q_pred = np.clip(pipe.predict(Xg), 0, None)
                        ax2.plot(p_grid, q_pred, label=z, color=colors_z[i % len(colors_z)], lw=2)
                    except Exception:
                        pass
                ax2.set_xlabel("Precio ($)")
                ax2.set_ylabel("Demanda estimada (unidades/semana)")
                ax2.set_title(f"Curva de demanda parcial por zona — {sel_prod}")
                ax2.legend(); ax2.grid(alpha=0.2)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
                st.caption("Las curvas distintas por zona muestran que LightGBM aprendió elasticidades diferentes para cada mercado.")

    # ── TAB 3: Linear vs LightGBM visual comparison ───────────────
    with tab3:
        st.markdown("""
        ### Regresión lineal vs LightGBM: ¿qué diferencia hay en la práctica?

        La regresión lineal asume que la demanda cae (o sube) siempre a la misma tasa con el precio,
        independientemente del contexto. LightGBM puede aprender que la pendiente cambia según
        la temporada, la zona o el historial reciente.
        """)

        if df_ml is None:
            st.info("Carga los datos para ver la comparación visual.")
        else:
            from src.features import get_feature_lists
            from src.models   import load_pipeline, _get_models
            from sklearn.linear_model import LinearRegression as LR

            num_features, cat_features = get_feature_lists(df_ml)
            products = sorted(summary["product"].unique())
            sel_prod2 = st.selectbox("Producto", products, key="lr_prod")

            sub = df_ml[df_ml["product_code"] == sel_prod2].sort_values("date").reset_index(drop=True)
            if len(sub) < 20:
                st.warning("Pocos datos para este producto.")
            else:
                X = sub[num_features + cat_features].copy()
                for c in cat_features:
                    X[c] = X[c].fillna("__NA__").astype(str)
                y = sub["units"].values.astype(float)
                split = int(0.8 * len(sub))

                # Load best pipeline (LightGBM or LR)
                pipe_best = load_pipeline(sel_prod2, ARTIFACTS)

                # Build a fresh LR pipeline for comparison
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.compose import ColumnTransformer
                pre_lr = ColumnTransformer([
                    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
                    ("num", "passthrough", num_features),
                ], remainder="drop")
                pipe_lr = Pipeline([("pre", pre_lr), ("scaler", StandardScaler()), ("est", LR())])
                pipe_lr.fit(X.iloc[:split], y[:split])

                dates_test = sub["date"].iloc[split:].values
                y_test     = y[split:]
                y_pred_best = np.clip(pipe_best.predict(X.iloc[split:]), 0, None) if pipe_best else y_test
                y_pred_lr   = np.clip(pipe_lr.predict(X.iloc[split:]), 0, None)

                from sklearn.metrics import mean_absolute_error, r2_score
                rmse_best = float(np.sqrt(np.mean((y_test - y_pred_best)**2)))
                rmse_lr   = float(np.sqrt(np.mean((y_test - y_pred_lr)**2)))
                r2_best   = r2_score(y_test, y_pred_best)
                r2_lr     = r2_score(y_test, y_pred_lr)

                # Metric comparison
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RMSE — LightGBM",  f"{rmse_best:.1f}")
                c2.metric("RMSE — Lineal",    f"{rmse_lr:.1f}",
                          delta=f"{rmse_lr - rmse_best:+.1f} vs LightGBM", delta_color="inverse")
                c3.metric("R² — LightGBM",    f"{r2_best:.3f}")
                c4.metric("R² — Lineal",      f"{r2_lr:.3f}",
                          delta=f"{r2_lr - r2_best:+.3f} vs LightGBM", delta_color="inverse")

                # Prediction vs actual plot
                fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

                axes[0].plot(dates_test, y_test,      color="gray",    lw=1.5, label="Real",         alpha=0.8)
                axes[0].plot(dates_test, y_pred_best, color="#2563EB", lw=1.8, label="LightGBM",     alpha=0.9)
                axes[0].set_ylabel("Unidades/semana")
                axes[0].set_title(f"Predicción en test set — {sel_prod2}")
                axes[0].legend(); axes[0].grid(alpha=0.2)

                axes[1].plot(dates_test, y_test,    color="gray",    lw=1.5, label="Real",         alpha=0.8)
                axes[1].plot(dates_test, y_pred_lr, color="#DC2626", lw=1.8, label="Lineal",        alpha=0.9)
                axes[1].set_ylabel("Unidades/semana")
                axes[1].set_title("Regresión Lineal — ¿captura los picos?")
                axes[1].legend(); axes[1].grid(alpha=0.2)

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # Residuals
                st.subheader("📐 Residuales (error = real − predicho)")
                fig2, ax = plt.subplots(figsize=(13, 3))
                ax.axhline(0, color="black", lw=0.8)
                ax.fill_between(range(len(y_test)), y_test - y_pred_best,
                                alpha=0.5, color="#2563EB", label=f"LightGBM (RMSE={rmse_best:.1f})")
                ax.fill_between(range(len(y_test)), y_test - y_pred_lr,
                                alpha=0.4, color="#DC2626", label=f"Lineal (RMSE={rmse_lr:.1f})")
                ax.set_xlabel("Semana (test set)"); ax.set_ylabel("Error")
                ax.legend(); ax.grid(alpha=0.2)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
                st.caption("Residuales menores y más centrados en 0 = mejor modelo. Si los patrones del error son sistemáticos (siempre positivos en verano, por ejemplo), el modelo no capturó esa señal.")



# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Run Pipeline
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
#  PAGE: Metodología (reemplaza "Ejecutar Pipeline" en producción)
# ════════════════════════════════════════════════════════════════════════════
def page_metodologia() -> None:
    st.title("⚙️ Metodología del Proyecto")

    st.info(
        "**Nota para visitantes:** el pipeline se ejecuta localmente antes de publicar. "
        "Los modelos y datos procesados ya están incluidos en la app.",
        icon="ℹ️",
    )

    st.markdown("---")

    # ── Arquitectura ──────────────────────────────────────────────
    st.subheader("🏗️ Arquitectura del pipeline")
    st.code("""
pricing/
├── generate_data.py   ← Genera datos sintéticos realistas (o conecta a fuente real)
├── run_pipeline.py    ← Orquesta los 4 pasos; se corre localmente
├── src/
│   ├── etl.py         ← Carga y normaliza sales_*.xlsx + masters
│   ├── features.py    ← Feature engineering (calendario MX, temperatura, lags)
│   ├── models.py      ← LinearRegression (baseline) vs LightGBM
│   ├── optimizer.py   ← Curvas de demanda, elasticidad, precio óptimo
│   └── viz.py         ← Gráficas reutilizables (notebooks + Streamlit)
├── artifacts/         ← Modelos .joblib + CSVs de resultados (output del pipeline)
└── app.py             ← Esta app
    """, language="text")

    st.markdown("---")
    st.subheader("🤖 ¿Cómo se usa el Machine Learning aquí?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Problema")
        st.markdown("""
Queremos saber: si cambiamos el precio de un producto en una zona específica,
¿cuántas unidades se venderán? Esta es una función de demanda, pero no asumimos
ninguna forma (lineal, log-lineal, etc.) — dejamos que los datos la definan.
        """)
        st.markdown("#### Datos de entrada por observación (semana × producto × tienda)")
        st.markdown("""
| Feature | Descripción |
|---|---|
| `unit_price_mean` | Precio promedio ← variable de decisión |
| `units_lag1/2/4` | Demanda histórica (memoria del sistema) |
| `temperatura` | Temperatura semanal por zona |
| `es_quincena` | Efecto quincena (MX) |
| `es_vacaciones` | Vacaciones escolares |
| `es_semana_santa` | Semana Santa |
| `promo_days` | Días con promoción |
| `zone`, `store_code` | Identidad del mercado |
        """)

    with col2:
        st.markdown("#### Modelos comparados")
        st.markdown("""
**LinearRegression (baseline)**
- Ajusta `Q = β₀ + β₁·precio + β₂·temp + ...`
- Elasticidad constante en toda la curva
- Rápido, interpretable, pero no captura interacciones

**LightGBM (challenger)**
- Ensemble de ~400 árboles de decisión (boosting)
- Cada árbol corrige los errores del anterior
- Captura: *"en Norte + verano + quincena, el precio importa más"*
- Validación: **TimeSeriesSplit** (no data leakage del futuro)

**Regla de oro:** si LightGBM no supera a la regresión lineal en RMSE de test,
el modelo complejo no sirve de nada.
        """)
        st.markdown("#### Cómo se obtiene la elasticidad")
        st.markdown(r"""
La elasticidad no es un parámetro del modelo — es una *consecuencia* de la curva.

**Paso 1:** fijar todos los features en sus medianas (condición típica)

**Paso 2:** variar solo el precio en 200 puntos de min a max

**Paso 3:** predecir $\hat{Q}(P)$ → curva de demanda

**Paso 4:** calcular numéricamente:
$$E(P) = \frac{dQ}{dP} \cdot \frac{P}{Q}$$

**Paso 5:** evaluar $E$ en el precio óptimo → el número que aparece en la app

Como la curva puede ser no lineal (con LightGBM), la elasticidad **cambia en cada punto de precio**.
        """)

    st.markdown("---")
    st.subheader("💡 ¿Por qué ML y no solo elasticidad clásica?")
    st.markdown("""
La elasticidad clásica (OLS de log-precio vs log-cantidad) asume una sola pendiente
para todos los contextos. El ML permite que la respuesta al precio sea distinta según:

- **Zona:** Norte puede ser más sensible que Sur
- **Temporada:** en verano la demanda del helado es menos elástica (compran de todos modos)
- **Quincena:** los clientes tienen más dinero y son menos sensibles al precio
- **Historial:** si la semana pasada hubo promo y compraron mucho, esta semana compran menos

El precio óptimo que calcula la app incorpora todo eso — no es solo la intersección
de una línea de demanda con el costo marginal.
    """)

    st.markdown("---")
    st.subheader("🔄 Para actualizar los datos")
    st.code("""
# 1. Agrega o actualiza tus archivos sales_*.xlsx en la carpeta source/
# 2. Corre el pipeline localmente:
python run_pipeline.py

# 3. Commit y push de los artefactos actualizados:
git add artifacts/
git commit -m "update models and price optimization"
git push
    """, language="bash")



# ════════════════════════════════════════════════════════════════════════════
#  Main router
# ════════════════════════════════════════════════════════════════════════════
#  PAGE: Data Explorer
# ════════════════════════════════════════════════════════════════════════════
def page_explorer(df_ml: pd.DataFrame) -> None:
    st.title("🗂️ Explorador de Datos")
    st.markdown("Filtra y explora ventas por tienda, zona, producto y período.")

    # ── Sidebar filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔎 Filtros")
        all_zones    = sorted(df_ml["zone"].dropna().unique())
        all_stores   = sorted(df_ml["store_code"].dropna().unique())
        all_products = sorted(df_ml["product_code"].dropna().unique())

        sel_zones    = st.multiselect("Zona",     all_zones,    default=all_zones)
        sel_stores   = st.multiselect("Tienda",   all_stores,   default=all_stores)
        sel_products = st.multiselect("Producto", all_products, default=all_products)

        min_date = df_ml["date"].min().date()
        max_date = df_ml["date"].max().date()
        date_range = st.date_input("Período", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

    # ── Apply filters ─────────────────────────────────────────────
    df = df_ml.copy()
    if sel_zones:
        df = df[df["zone"].isin(sel_zones)]
    if sel_stores:
        df = df[df["store_code"].isin(sel_stores)]
    if sel_products:
        df = df[df["product_code"].isin(sel_products)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df = df[(df["date"] >= pd.Timestamp(date_range[0])) &
                (df["date"] <= pd.Timestamp(date_range[1]))]

    if df.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    # ── KPIs filtrados ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Revenue",        f"${df['revenue'].sum():,.0f}")
    c2.metric("📦 Unidades",       f"{int(df['units'].sum()):,}")
    c3.metric("💲 Precio promedio", f"${df['unit_price_mean'].mean():.2f}")
    c4.metric("📅 Semanas",        f"{df['date'].nunique()}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Series de tiempo", "📊 Comparativas", "🌡️ Temperatura", "📋 Datos crudos"])

    # ── TAB 1: Time series ────────────────────────────────────────
    with tab1:
        st.subheader("Ventas semanales")
        ts_group = st.selectbox("Agrupar por", ["producto", "tienda", "zona"], key="ts_group")
        group_col = {"producto": "product_code", "tienda": "store_code", "zona": "zone"}[ts_group]

        ts_data = df.groupby(["date", group_col])["units"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(13, 4))
        for name, grp in ts_data.groupby(group_col):
            ax.plot(grp["date"], grp["units"], label=name, lw=1.8)
        ax.set_title("Unidades vendidas por semana")
        ax.set_ylabel("Unidades"); ax.set_xlabel("")
        ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # precio promedio semanal
        st.subheader("Precio promedio semanal")
        price_data = df.groupby(["date", group_col])["unit_price_mean"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(13, 3))
        for name, grp in price_data.groupby(group_col):
            ax2.plot(grp["date"], grp["unit_price_mean"], label=name, lw=1.5)
        ax2.set_ylabel("Precio ($)"); ax2.grid(alpha=0.2); ax2.legend(fontsize=9, ncol=3)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    # ── TAB 2: Comparativas ───────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Unidades por producto × zona")
            agg = df.groupby(["product_code", "zone"])["units"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            zones_u = sorted(agg["zone"].unique())
            prods_u = sorted(agg["product_code"].unique())
            x       = np.arange(len(prods_u))
            w       = 0.8 / max(len(zones_u), 1)
            colors  = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]
            for i, z in enumerate(zones_u):
                vals = [agg[(agg["product_code"]==p) & (agg["zone"]==z)]["units"].sum() for p in prods_u]
                ax.bar(x + i*w, vals, w, label=z, color=colors[i % len(colors)])
            ax.set_xticks(x + w*(len(zones_u)-1)/2)
            ax.set_xticklabels(prods_u, rotation=30)
            ax.set_ylabel("Unidades"); ax.legend(); ax.grid(axis="y", alpha=0.2)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with col_b:
            st.subheader("Revenue por tienda")
            rev_store = df.groupby("store_code")["revenue"].sum().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(rev_store.index, rev_store.values, color="#2563EB")
            ax.set_xlabel("Revenue ($)")
            for i, v in enumerate(rev_store.values):
                ax.text(v * 1.01, i, f"${v:,.0f}", va="center", fontsize=8)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.subheader("Día de la semana — promedio de unidades")
        day_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        df2 = df.copy()
        df2["dow"] = pd.to_datetime(df2["date"]).dt.weekday
        hm_data = (
            df2.groupby(["dow", "product_code"])["units"]
            .mean().unstack(fill_value=0)
            .reindex(range(7))
        )
        hm_data.index = day_names
        fig, ax = plt.subplots(figsize=(10, 3))
        import seaborn as sns
        sns.heatmap(hm_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, linewidths=0.4)
        ax.set_xlabel("Producto"); ax.set_ylabel("")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── TAB 3: Temperatura ────────────────────────────────────────
    with tab3:
        if "temperatura" not in df.columns:
            st.info("No hay columna de temperatura en los datos.")
        else:
            st.subheader("Relación temperatura → ventas")
            prod_t = st.selectbox("Producto", sel_products, key="prod_temp")
            sub_t  = df[df["product_code"] == prod_t]

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            # scatter temperatura vs unidades
            for z in sub_t["zone"].dropna().unique():
                zd = sub_t[sub_t["zone"] == z]
                axes[0].scatter(zd["temperatura"], zd["units"], alpha=0.4, s=20, label=z)
            axes[0].set_xlabel("Temperatura (°C)"); axes[0].set_ylabel("Unidades vendidas")
            axes[0].set_title(f"Temperatura vs Ventas — {prod_t}")
            axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)

            # temperatura media mensual
            sub_t2 = sub_t.copy()
            sub_t2["month"] = pd.to_datetime(sub_t2["date"]).dt.month
            monthly = sub_t2.groupby("month").agg(
                temp=("temperatura", "mean"), units=("units", "mean")
            ).reset_index()
            ax2 = axes[1]
            ax2b = ax2.twinx()
            ax2.bar(monthly["month"], monthly["units"], color="#2563EB", alpha=0.5, label="Unidades prom")
            ax2b.plot(monthly["month"], monthly["temp"], color="#DC2626", lw=2, marker="o", label="Temp °C")
            ax2.set_xlabel("Mes"); ax2.set_ylabel("Unidades prom", color="#2563EB")
            ax2b.set_ylabel("Temperatura (°C)", color="#DC2626")
            ax2.set_title(f"Estacionalidad mensual — {prod_t}")
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels(["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"], rotation=30)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

            # correlación por zona
            st.subheader("Correlación temperatura–ventas por zona")
            corr_rows = []
            for z in sub_t["zone"].dropna().unique():
                zd = sub_t[sub_t["zone"] == z]
                if len(zd) > 5:
                    r = zd[["temperatura", "units"]].corr().iloc[0, 1]
                    corr_rows.append({"Zona": z, "Correlación (r)": round(r, 3),
                                      "Interpretación": "Alta" if abs(r) > 0.5 else "Moderada" if abs(r) > 0.3 else "Baja"})
            if corr_rows:
                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

    # ── TAB 4: Raw data ───────────────────────────────────────────
    with tab4:
        st.subheader(f"Datos filtrados — {len(df):,} registros semanales")
        st.dataframe(
            df.sort_values("date", ascending=False).reset_index(drop=True),
            use_container_width=True, height=400,
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv, "datos_filtrados.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    page = sidebar()

    df_ml   = get_df_ml()
    opt_df  = get_opt_df()
    summary = get_summary()

    if page == "🏠 Dashboard":
        if df_ml is None:
            st.warning("No se encontraron datos. Coloca tus archivos `sales_*.xlsx` en la carpeta `data/` y ejecuta el pipeline.")
        else:
            page_dashboard(df_ml, opt_df)

    elif page == "🔍 Optimizador de Precio":
        if df_ml is None:
            st.warning("No hay datos cargados. Ejecuta el pipeline primero.")
        else:
            page_optimizer(df_ml)

    elif page == "🗂️ Explorador de Datos":
        if df_ml is None:
            st.warning("No hay datos cargados. Ejecuta el pipeline primero.")
        else:
            page_explorer(df_ml)

    elif page == "📊 Comparación de Modelos":
        if summary is None:
            st.warning("No hay resultados de modelos. Ejecuta el pipeline primero.")
        else:
            page_model_comparison(summary, df_ml)

    #elif page == "⚙️ Metodología":
        #page_metodologia()


if __name__ == "__main__":
    main()