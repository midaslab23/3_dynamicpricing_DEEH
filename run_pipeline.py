# run_pipeline.py
"""
Script principal — ejecuta el pipeline completo.

Uso:
    python run_pipeline.py
    python run_pipeline.py --data ./source --artifacts ./artifacts
"""
import argparse
import sys
from pathlib import Path

# ── sys.path AQUÍ ARRIBA, antes de cualquier import de src ──────────────────
# Apunta a la CARPETA DEL PROYECTO (donde está run_pipeline.py)
# así Python puede encontrar la subcarpeta src/ con from src.xxx import ...
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ahora sí se pueden importar los módulos del proyecto
from src.etl       import load_sales, load_masters, build_canonical
from src.features  import build_weekly_ml
from src.models    import train_all_products
from src.optimizer import optimize_all


# ── Detección automática de carpeta de datos ─────────────────────────────────
def _auto_detect_data_path(given: str) -> str:
    p = Path(given)
    if p.exists() and list(p.glob("sales_*.xlsx")):
        return str(p)

    for name in ["source", "data", "ventas", "sales"]:
        candidate = _PROJECT_ROOT / name
        if candidate.exists() and list(candidate.glob("sales_*.xlsx")):
            if name != given:
                print(f"  [Pipeline] '{given}' -> usando '{candidate}' automaticamente.")
            return str(candidate)

    return given


# ── Pipeline principal ────────────────────────────────────────────────────────
def main(data_path: str = "source", artifacts_path: str = "artifacts") -> None:
    data_path     = _auto_detect_data_path(data_path)
    artifacts_dir = _PROJECT_ROOT / artifacts_path

    print("=" * 60)
    print("  PRICING ML PIPELINE")
    print(f"  Datos:      {Path(data_path).resolve()}")
    print(f"  Artefactos: {artifacts_dir.resolve()}")
    print("=" * 60)

    # 1. ETL
    print("\n[1/4] ETL")
    sales                  = load_sales(data_path)
    df_products, df_stores = load_masters(data_path)

    # ── DIAGNOSTICO: muestra exactamente qué leyó de products.xlsx ──────────
    products_file = Path(data_path) / "products.xlsx"
    print(f"\n  [DIAG] products.xlsx encontrado en: {products_file.resolve()}")
    print(f"  [DIAG] Ultima modificacion: {__import__('datetime').datetime.fromtimestamp(products_file.stat().st_mtime)}")
    if df_products is not None:
        print(f"  [DIAG] Columnas leidas: {df_products.columns.tolist()}")
        print(f"  [DIAG] Contenido completo de products.xlsx:")
        print(df_products.to_string(index=False))
    else:
        print("  [DIAG] *** products.xlsx NO se cargo — verifica que exista y no este abierto en Excel ***")
    print()
    # ────────────────────────────────────────────────────────────────────────

    df_canonical = build_canonical(sales, df_products, df_stores)

    # ── DIAGNOSTICO: verifica que unit_cost llegó a df_canonical ─────────────
    if "unit_cost" in df_canonical.columns:
        cost_check = df_canonical.groupby("product_code")["unit_cost"].mean().round(2)
        print(f"\n  [DIAG] unit_cost en df_canonical por producto:")
        print(cost_check.to_string())
    else:
        print("\n  [DIAG] *** unit_cost NO esta en df_canonical — el merge no funciono ***")
    print()
    # ────────────────────────────────────────────────────────────────────────

    # 2. Feature engineering
    print("\n[2/4] Feature engineering")
    df_ml = build_weekly_ml(df_canonical)
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    df_ml.to_parquet(artifacts_dir / "df_ml.parquet", index=False)
    print(f"  Guardado -> {artifacts_dir / 'df_ml.parquet'}")

    # 3. Entrenamiento
    print("\n[3/4] Entrenando modelos (LinearRegression vs LightGBM)")
    all_results = train_all_products(df_ml, artifacts_path=str(artifacts_dir))

    # 4. Optimizacion de precios
    print("\n[4/4] Optimizando precios")
    opt_df = optimize_all(df_ml, all_results, artifacts_path=str(artifacts_dir))

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETO")
    print("=" * 60)
    cols = [c for c in ["product", "zone", "current_price_mean",
                        "optimal_price", "price_delta_pct",
                        "elasticity", "objective_label"] if c in opt_df.columns]
    print(opt_df[cols].to_string(index=False))
    print(f"\nArtefactos en: {artifacts_dir.resolve()}")
    print("Corre la app:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pricing ML Pipeline")
    parser.add_argument("--data",      default="source",    help="Carpeta con sales_*.xlsx")
    parser.add_argument("--artifacts", default="artifacts", help="Carpeta de salida")
    args = parser.parse_args()
    main(args.data, args.artifacts)