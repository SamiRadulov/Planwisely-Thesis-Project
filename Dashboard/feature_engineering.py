"""
Feature Engineering
===================
Builds a per-product weekly feature set from preprocessed sales data.
Mirrors feature_engineering.ipynb exactly.

External CSVs are loaded from the sibling 'Initial Data' folder (with fallbacks).
If a file is missing the corresponding feature group is silently skipped.

Exported symbols used by app.py and shap_utils.py
---------------------------------------------------
  build_features(df_in)          -> feature-engineered DataFrame
  get_external_for_date(date)    -> dict of external feature values for one future date
  HAS_WEATHER, HAS_HOLIDAY, HAS_COVID, HAS_SCHOOL
  WEATHER_COLS, HOLIDAY_COLS, COVID_COLS, SCHOOL_COLS
  FCOLS_BASE
"""

import pathlib
import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).parent
_ROOT = _HERE.parent

def _data_path(name):
    """Locate an external data file. After the repo reorganisation these live in
    the sibling 'Initial Data' folder; the fallbacks keep older layouts working."""
    for _p in (_ROOT / "Initial Data" / name, _HERE / name, _ROOT / name):
        if _p.exists():
            return _p
    return _ROOT / "Initial Data" / name

# ── Load external data files ──────────────────────────────────────────────────

try:
    _WEATHER_RAW = pd.read_csv(_data_path("weather_weekly.csv"))
    HAS_WEATHER  = True
except Exception:
    _WEATHER_RAW = None
    HAS_WEATHER  = False

try:
    _HOLIDAY_RAW = pd.read_csv(_data_path("holiday_weekly.csv"))
    HAS_HOLIDAY  = True
except Exception:
    _HOLIDAY_RAW = None
    HAS_HOLIDAY  = False

try:
    _COVID_RAW = pd.read_csv(_data_path("covid_weekly.csv"))
    HAS_COVID  = True
except Exception:
    _COVID_RAW = None
    HAS_COVID  = False

try:
    _SCHOOL_RAW = pd.read_csv(_data_path("school_holidays_weekly.csv"))
    HAS_SCHOOL  = True
except Exception:
    _SCHOOL_RAW = None
    HAS_SCHOOL  = False

HAS_EXTERNAL = HAS_WEATHER or HAS_HOLIDAY or HAS_COVID or HAS_SCHOOL

# ── Column groups (matching feature_engineering.ipynb) ────────────────────────

WEATHER_COLS = [
    "temp_mean", "temp_min", "temp_max",
    "precip_sum", "sunshine_sum", "temp_anomaly", "heavy_rain",
]
HOLIDAY_COLS = [
    "has_holiday", "min_days_to_holiday",
    "hol_new_year", "hol_easter", "hol_kings_day", "hol_liberation_day",
    "hol_ascension", "hol_pentecost", "hol_christmas",
]
COVID_COLS = ["lockdown", "lockdown_days"]
SCHOOL_COLS = [
    "school_holiday",
    "school_autumn", "school_christmas", "school_may",
    "school_spring", "school_summer",
]

FCOLS_BASE = [
    "trend", "month", "week_of_year", "quarter",
    "is_month_start", "is_month_end",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "rolling_std_8",
]

# Pre-compute week-of-year averages for weather imputation (matching notebook § 3)
_WEATHER_AVGS = (
    _WEATHER_RAW.groupby("week")[WEATHER_COLS].mean().reset_index()
    if HAS_WEATHER else None
)


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df_in):
    """
    Feature-engineer a per-product weekly sales series.

    Parameters
    ----------
    df_in : DataFrame with columns ['date', 'sales']
        One product, any date range.

    Returns
    -------
    DataFrame with all engineered features; rows with NaN lags are dropped.
    """
    df = df_in[["date", "sales"]].copy().sort_values("date").reset_index(drop=True)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["sales"])

    # § 4  Calendar features  (feature_engineering.ipynb)
    df["year"]           = df["date"].dt.isocalendar().year.astype(int)
    df["week"]           = df["date"].dt.isocalendar().week.astype(int)
    df["month"]          = df["date"].dt.month
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]        = df["date"].dt.quarter
    df["is_month_start"] = (df["date"].dt.day <= 7).astype(int)
    df["is_month_end"]   = (df["date"].dt.day >= 24).astype(int)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]       = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]       = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["trend"]          = np.arange(len(df))

    # § 5  Lag & rolling features  (feature_engineering.ipynb)
    # groupby not needed here — df is already per-product
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for w in [4, 8]:
        df[f"rolling_mean_{w}"] = df["sales"].shift(1).rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"]  = (
            df["sales"].shift(1).rolling(w, min_periods=1).std().fillna(0)
        )

    # § 2  Merge external sources  (feature_engineering.ipynb)

    # Weather — impute missing by week-of-year average (§ 3)
    if HAS_WEATHER:
        wdf = _WEATHER_RAW.rename(columns={"week": "week_of_year"})
        df  = df.merge(
            wdf[["year", "week_of_year"] + WEATHER_COLS],
            on=["year", "week_of_year"], how="left",
        )
        week_avg_map = (
            _WEATHER_AVGS.set_index("week") if _WEATHER_AVGS is not None else None
        )
        for col in WEATHER_COLS:
            if col in df.columns and week_avg_map is not None:
                df[col] = df.apply(
                    lambda r, c=col: (
                        week_avg_map[c].get(r["week_of_year"], df[c].mean())
                        if pd.isna(r[c]) else r[c]
                    ),
                    axis=1,
                )

    # Public holidays — missing → 0
    if HAS_HOLIDAY:
        hdf = _HOLIDAY_RAW.rename(columns={"week": "week_of_year"})
        df  = df.merge(
            hdf[["year", "week_of_year"] + HOLIDAY_COLS],
            on=["year", "week_of_year"], how="left",
        )
        for col in HOLIDAY_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # COVID — missing → 0
    if HAS_COVID:
        cdf = _COVID_RAW.rename(columns={"week": "week_of_year"})
        df  = df.merge(
            cdf[["year", "week_of_year"] + COVID_COLS],
            on=["year", "week_of_year"], how="left",
        )
        for col in COVID_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # School holidays — missing → 0
    if HAS_SCHOOL:
        sdf = _SCHOOL_RAW.rename(columns={"week": "week_of_year"})
        df  = df.merge(
            sdf[["year", "week_of_year"] + SCHOOL_COLS],
            on=["year", "week_of_year"], how="left",
        )
        for col in SCHOOL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # Drop rows where lag features are NaN (first 8 rows per product — § 5)
    return df.dropna(subset=["lag_1", "lag_2", "lag_4", "lag_8"]).reset_index(drop=True)


def get_external_for_date(nd):
    """
    Return a dict of external feature values for a single future date.
    Used by the 4-period ahead forecasting loop in app.py.
    Weather is filled from week-of-year averages; holidays/COVID/school from
    the raw tables (0 if the week is not found).
    """
    week = int(nd.isocalendar()[1])
    year = int(nd.isocalendar()[0])
    r    = {}

    if HAS_WEATHER and _WEATHER_AVGS is not None:
        w_row = _WEATHER_AVGS[_WEATHER_AVGS["week"] == week]
        for col in WEATHER_COLS:
            r[col] = float(w_row[col].values[0]) if len(w_row) > 0 else 0.0

    if HAS_HOLIDAY and _HOLIDAY_RAW is not None:
        h_row = _HOLIDAY_RAW[
            (_HOLIDAY_RAW["year"] == year) & (_HOLIDAY_RAW["week"] == week)
        ]
        for col in HOLIDAY_COLS:
            r[col] = float(h_row[col].values[0]) if len(h_row) > 0 else 0.0

    if HAS_COVID and _COVID_RAW is not None:
        c_row = _COVID_RAW[
            (_COVID_RAW["year"] == year) & (_COVID_RAW["week"] == week)
        ]
        for col in COVID_COLS:
            r[col] = float(c_row[col].values[0]) if len(c_row) > 0 else 0.0

    if HAS_SCHOOL and _SCHOOL_RAW is not None:
        s_row = _SCHOOL_RAW[
            (_SCHOOL_RAW["year"] == year) & (_SCHOOL_RAW["week"] == week)
        ]
        for col in SCHOOL_COLS:
            r[col] = float(s_row[col].values[0]) if len(s_row) > 0 else 0.0

    return r
