# src/etl.py
"""
ETL: carga y normaliza archivos de ventas + maestros (products, stores).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np


# ── column-name heuristics ────────────────────────────────────────────────────
_COLMAP_RULES: list[tuple[list[str], str]] = [
    (["date", "fecha", "sale_date"],                         "date"),
    (["store_code", "store", "tienda", "storeid", "store_id"],         "store_code"),
    (["product_code", "product", "producto", "id_prod", "sku"],        "product_code"),
    (["units", "cantidad", "qty", "amount"],                  "units"),
    (["unit_price", "price", "precio"],                       "unit_price"),
    (["promo"],                                               "promo"),
    (["zone", "zona", "region"],                              "zone"),
    (["temperatura", "temp", "temperature"],                  "temperatura"),
    (["es_vacaciones", "vacaciones", "vacation"],             "es_vacaciones"),
    (["es_semana_santa", "semana_santa", "holy_week"],        "es_semana_santa"),
    (["es_quincena", "quincena"],                             "es_quincena"),
]

def _guess_colname(raw: str) -> str | None:
    low = raw.strip().lower()
    for candidates, canonical in _COLMAP_RULES:
        for cand in candidates:
            if cand in low:
                return canonical
    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    colmap = {}
    for c in df.columns:
        mapped = _guess_colname(c)
        if mapped and mapped not in colmap.values():
            colmap[c] = mapped
    df = df.rename(columns=colmap)

    # coerce types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "units" in df.columns:
        df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(int)
    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    if "promo" in df.columns:
        df["promo"] = pd.to_numeric(df["promo"], errors="coerce").fillna(0).astype(int)
    else:
        df["promo"] = 0

    # optional columns – if user already put them in Excel, coerce them
    for col in ["temperatura", "es_vacaciones", "es_semana_santa", "es_quincena"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _read_file(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  [ETL] Could not read {path.name}: {e}")
            return None


def load_sales(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    sales_files = sorted(data_path.glob("sales_*.xlsx")) + sorted(data_path.glob("sales_*.csv"))
    if not sales_files:
        raise FileNotFoundError(f"No sales_*.xlsx/csv files found in {data_path}")

    frames = []
    for p in sales_files:
        raw = _read_file(p)
        if raw is None:
            continue
        df = normalize_df(raw)
        # inject store_code from filename if missing
        if "store_code" not in df.columns:
            stem = p.stem  # e.g. "sales_NLE-ALL-01"
            if "sales_" in stem:
                df["store_code"] = stem.split("sales_", 1)[-1]
        frames.append(df)

    if not frames:
        raise RuntimeError("No dataframes loaded – check file format.")

    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[ETL] Loaded {len(out):,} rows from {len(frames)} files.")
    return out


def load_masters(data_path: str | Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    data_path = Path(data_path)
    df_products = df_stores = None

    products_path = data_path / "products.xlsx"
    if products_path.exists():
        df_products = _read_file(products_path)
        if df_products is not None:
            df_products.columns = [c.strip().lower() for c in df_products.columns]
            # ensure product_code and unit_cost
            for c in df_products.columns:
                if "code" in c or "sku" in c:
                    df_products = df_products.rename(columns={c: "product_code"})
                    break
            print(f"[ETL] Loaded products master ({len(df_products)} rows).")

    stores_path = data_path / "stores.xlsx"
    if stores_path.exists():
        df_stores = _read_file(stores_path)
        if df_stores is not None:
            df_stores.columns = [c.strip().lower() for c in df_stores.columns]
            for c in df_stores.columns:
                if "code" in c or "id" in c:
                    df_stores = df_stores.rename(columns={c: "store_code"})
                    break
            if "zona" in df_stores.columns:
                df_stores = df_stores.rename(columns={"zona": "zone"})
            print(f"[ETL] Loaded stores master ({len(df_stores)} rows).")

    return df_products, df_stores


def build_canonical(
    sales: pd.DataFrame,
    df_products: pd.DataFrame | None = None,
    df_stores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = sales.copy()

    if df_products is not None and "product_code" in df_products.columns:
        df = df.merge(df_products, on="product_code", how="left", suffixes=("", "_prod"))

    if df_stores is not None and "store_code" in df_stores.columns:
        df = df.merge(df_stores, on="store_code", how="left", suffixes=("", "_store"))
        if "zone" not in df.columns and "zone_store" in df.columns:
            df["zone"] = df["zone_store"]

    # derive zone from store_code prefix if still missing
    if "zone" not in df.columns or df["zone"].isna().all():
        zone_map = {
            "NLE": "Norte", "TAM": "Norte", "COA": "Norte",
            "CDMX": "Centro", "MEX": "Centro", "HID": "Centro", "MOR": "Centro",
            "JAL": "Occidente", "COL": "Occidente", "AGU": "Occidente",
            "YUC": "Sur", "OAX": "Sur", "CHI": "Sur", "TAB": "Sur",
        }
        def _infer_zone(code: str) -> str:
            if pd.isna(code):
                return "Centro"
            prefix = str(code).split("-")[0].upper()
            return zone_map.get(prefix, "Centro")
        df["zone"] = df["store_code"].apply(_infer_zone)

    df["revenue"] = df["units"] * df["unit_price"]

    # unit_cost may arrive as "unit_cost_prod" if there was a column collision —
    # normalize it back to "unit_cost" before the keep filter
    if "unit_cost" not in df.columns and "unit_cost_prod" in df.columns:
        df["unit_cost"] = df["unit_cost_prod"]

    canonical = ["date", "store_code", "product_code",
                 "units", "unit_price", "revenue", "promo", "zone"]
    optional = ["unit_cost", "temperatura", "es_vacaciones",
                "es_semana_santa", "es_quincena"]
    keep = canonical + [c for c in optional if c in df.columns]

    for c in canonical:
        if c not in df.columns:
            df[c] = np.nan

    df = df[keep].copy()
    df = df.sort_values("date").reset_index(drop=True)

    has_cost = "unit_cost" in df.columns and df["unit_cost"].notna().any()
    print(f"[ETL] Canonical table: {df.shape}  |  products: {df['product_code'].nunique()}  |  stores: {df['store_code'].nunique()}  |  unit_cost: {'✓' if has_cost else '✗ NOT FOUND — check products.xlsx'}")
    return df