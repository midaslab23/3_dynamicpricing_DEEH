# src/features.py
"""
Feature engineering: calendario mexicano, temperatura, lags, aggregación semanal.
Si el usuario ya llenó las columnas en Excel, se usan tal cual.
Si no, se generan automáticamente.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date, timedelta


# ── Easter (Pascua) ── algoritmo de Butcher ───────────────────────────────────
def _easter_date(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _semana_santa_range(year: int) -> tuple[date, date]:
    """Domingo de Ramos (Palm Sunday) a Domingo de Pascua."""
    easter = _easter_date(year)
    start = easter - timedelta(days=7)   # Palm Sunday
    return start, easter


# ── Mexican school vacation windows ──────────────────────────────────────────
_VACATION_RANGES: list[tuple[str, str]] = [
    # Navidad / Año Nuevo  (approx for any year, we'll handle year dynamically)
]

def _is_school_vacation(d: pd.Timestamp) -> int:
    """Returns 1 if date falls in Mexican school vacation period."""
    m, day, wd = d.month, d.day, d.weekday()
    # Navidad-Año Nuevo: Dec 20 – Jan 6
    if (m == 12 and day >= 20) or (m == 1 and day <= 6):
        return 1
    # Verano: July 1 – Aug 31
    if m in (7, 8):
        return 1
    # Semana Santa handled separately
    return 0


def _is_semana_santa(d: pd.Timestamp) -> int:
    start, end = _semana_santa_range(d.year)
    return int(start <= d.date() <= end)


def _is_quincena(d: pd.Timestamp) -> int:
    """Days around the 15th and last day of month (±2 days)."""
    day = d.day
    last_day = pd.Timestamp(d.year, d.month, 1).days_in_month
    return int(abs(day - 15) <= 2 or abs(day - last_day) <= 2 or day == 1)


# ── Temperature simulation by state ──────────────────────────────────────────
# Monthly mean temperatures (°C) per climate zone
_TEMP_PROFILES: dict[str, list[float]] = {
    # Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    "Norte":     [14, 16, 21, 26, 29, 31, 30, 30, 26, 22, 16, 13],
    "Centro":    [13, 15, 18, 20, 20, 19, 18, 18, 18, 17, 15, 13],
    "Occidente": [15, 17, 20, 23, 24, 22, 21, 21, 21, 19, 17, 15],
    "Sur":       [24, 25, 28, 30, 30, 28, 27, 27, 26, 25, 24, 23],
}
_DEFAULT_PROFILE = [17, 19, 22, 24, 25, 23, 22, 22, 21, 20, 18, 16]

def _simulate_temperatura(date: pd.Timestamp, zone: str) -> float:
    profile = _TEMP_PROFILES.get(zone, _DEFAULT_PROFILE)
    base = profile[date.month - 1]
    # add small deterministic noise based on day-of-year
    noise = np.sin(date.timetuple().tm_yday / 365 * 2 * np.pi) * 1.5
    return round(base + noise, 1)


# ── Public API ────────────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, month, week_of_year, is_weekend."""
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.weekday          # 0=Mon, 6=Sun
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["date"].dt.weekday >= 5).astype(int)
    return df


def add_mexico_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add es_quincena, es_vacaciones, es_semana_santa (auto if not in df)."""
    df = df.copy()

    if "es_quincena" not in df.columns or df["es_quincena"].isna().all():
        df["es_quincena"] = df["date"].apply(_is_quincena).astype(int)
    else:
        df["es_quincena"] = df["es_quincena"].fillna(0).astype(int)

    if "es_vacaciones" not in df.columns or df["es_vacaciones"].isna().all():
        df["es_vacaciones"] = df["date"].apply(_is_school_vacation).astype(int)
    else:
        df["es_vacaciones"] = df["es_vacaciones"].fillna(0).astype(int)

    if "es_semana_santa" not in df.columns or df["es_semana_santa"].isna().all():
        df["es_semana_santa"] = df["date"].apply(_is_semana_santa).astype(int)
    else:
        df["es_semana_santa"] = df["es_semana_santa"].fillna(0).astype(int)

    return df


def add_temperatura(df: pd.DataFrame) -> pd.DataFrame:
    """Use column if present, otherwise simulate."""
    df = df.copy()
    if "temperatura" not in df.columns or df["temperatura"].isna().all():
        df["temperatura"] = df.apply(
            lambda r: _simulate_temperatura(r["date"], r.get("zone", "Centro")), axis=1
        )
    else:
        # fill missing rows
        mask = df["temperatura"].isna()
        df.loc[mask, "temperatura"] = df.loc[mask].apply(
            lambda r: _simulate_temperatura(r["date"], r.get("zone", "Centro")), axis=1
        )
    return df


def add_lags(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str = "units",
    lags: list[int] = [1, 2, 4],
) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + ["date"]).reset_index(drop=True)
    grp = df.groupby(group_cols)[target]
    for lag in lags:
        df[f"{target}_lag{lag}"] = grp.shift(lag).fillna(0)
    # rolling mean: transform keeps the original index aligned, no reset_index needed
    df[f"{target}_roll4_mean"] = (
        grp.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        .fillna(0)
    )
    df["price_pct_change"] = (
        df.groupby(group_cols)["unit_price_mean"]
        .pct_change()
        .fillna(0)
    )
    return df


def build_weekly_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily canonical df to weekly per product×store×zone,
    then add all features.
    Returns df_ml ready for modeling.
    """
    daily = df.copy()
    daily = add_calendar_features(daily)
    daily = add_mexico_calendar(daily)
    daily = add_temperatura(daily)

    # weekly aggregation
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="d")

    agg_dict: dict[str, tuple] = {
        "units":          ("units", "sum"),
        "unit_price_mean": ("unit_price", "mean"),
        "revenue":        ("revenue", "sum"),
        "promo_days":     ("promo", "sum"),
        "temperatura":    ("temperatura", "mean"),
        "es_quincena":    ("es_quincena", "max"),
        "es_vacaciones":  ("es_vacaciones", "max"),
        "es_semana_santa":("es_semana_santa", "max"),
        "is_weekend_days":("is_weekend", "sum"),
    }
    if "unit_cost" in daily.columns:
        agg_dict["unit_cost"] = ("unit_cost", "mean")

    group_cols = ["week_start", "product_code", "store_code", "zone"]
    weekly = daily.groupby(group_cols).agg(**agg_dict).reset_index()
    weekly = weekly.rename(columns={"week_start": "date"})

    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["month"]        = weekly["date"].dt.month
    weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)

    weekly = add_lags(weekly, ["product_code", "store_code"], target="units", lags=[1, 2, 4])
    weekly = weekly.dropna(subset=["product_code", "unit_price_mean"]).reset_index(drop=True)

    print(f"[Features] df_ml shape: {weekly.shape}  |  products: {weekly['product_code'].nunique()}")
    return weekly


# ── Feature lists (shared with models.py) ────────────────────────────────────
BASE_NUM_FEATURES = [
    "unit_price_mean",
    "units_lag1", "units_lag2", "units_lag4",
    "units_roll4_mean",
    "price_pct_change",
    "promo_days",
    "temperatura",
    "es_quincena",
    "es_vacaciones",
    "es_semana_santa",
    "is_weekend_days",
    "week_of_year",
    "month",
]
BASE_CAT_FEATURES = ["zone", "store_code"]


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (num_features, cat_features) filtered to columns present in df."""
    num = [f for f in BASE_NUM_FEATURES if f in df.columns]
    cat = [f for f in BASE_CAT_FEATURES if f in df.columns]
    return num, cat