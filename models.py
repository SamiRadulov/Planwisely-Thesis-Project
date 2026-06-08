"""
Models
======
Model definitions, hyperparameters, training, and feature-importance helpers.

Sources
-------
  Pipeline definitions  : compare_models.py
  XGBoost hyperparams   : XGB_prediction.ipynb  (max_depth=5, min_child_weight=5)
  Feature column list   : feature_engineering.ipynb  § 6

Note on XGBoost pipeline
------------------------
XGBoost is invariant to feature scaling, so StandardScaler is omitted for
that model — matching XGB_prediction.ipynb which trains a plain XGBRegressor.
Ridge and Lasso retain their StandardScaler.
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from lightgbm_model import build_pipeline as _build_lightgbm

from feature_engineering import (
    FCOLS_BASE,
    WEATHER_COLS, HOLIDAY_COLS, COVID_COLS, SCHOOL_COLS,
    HAS_WEATHER, HAS_HOLIDAY, HAS_COVID, HAS_SCHOOL,
)

# ── Feature column list  (feature_engineering.ipynb § 6) ─────────────────────

EXT_COLS = (
    (WEATHER_COLS if HAS_WEATHER else []) +
    (HOLIDAY_COLS if HAS_HOLIDAY else []) +
    (COVID_COLS   if HAS_COVID   else []) +
    (SCHOOL_COLS  if HAS_SCHOOL  else [])
)
FCOLS = FCOLS_BASE + EXT_COLS

# ── Human-readable feature labels ────────────────────────────────────────────

FEAT_LABELS = {
    "lag_1": "Last period sales",       "lag_2": "Sales 2 periods ago",
    "lag_4": "Sales 4 periods ago",     "lag_8": "Sales 8 periods ago",
    "rolling_mean_4": "4-period avg",   "rolling_mean_8": "8-period avg",
    "rolling_std_4": "4-period volatility", "rolling_std_8": "8-period volatility",
    "trend": "Long-term trend",         "month": "Month of year",
    "week_of_year": "Week of year",     "quarter": "Quarter",
    "is_month_start": "Month start",    "is_month_end": "Month end",
    "month_sin": "Seasonality (month)", "month_cos": "Seasonality (month)",
    "week_sin": "Seasonality (week)",   "week_cos": "Seasonality (week)",
    "temp_mean": "Avg temperature",     "temp_min": "Min temperature",
    "temp_max": "Max temperature",      "precip_sum": "Precipitation",
    "sunshine_sum": "Sunshine hours",   "temp_anomaly": "Temp anomaly",
    "heavy_rain": "Heavy rain",         "has_holiday": "Holiday week",
    "min_days_to_holiday": "Days to holiday",
    "hol_ascension": "Ascension Day",   "hol_christmas": "Christmas",
    "hol_easter": "Easter",             "hol_kings_day": "King's Day",
    "hol_liberation_day": "Liberation Day",
    "hol_new_year": "New Year",         "hol_pentecost": "Pentecost",
    "lockdown": "Lockdown",             "lockdown_days": "Lockdown days",
    "school_holiday": "School holiday", "school_autumn": "Autumn holiday",
    "school_christmas": "Christmas holiday", "school_may": "May holiday",
    "school_spring": "Spring holiday",  "school_summer": "Summer holiday",
}

# ── XGBoost hyperparameters  (XGB_prediction.ipynb) ──────────────────────────

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
)

# ── Model constructors  (compare_models.py) ───────────────────────────────────

def _build_ridge():
    return Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=10.0)),
    ])

def _build_lasso():
    return Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    Lasso(alpha=1.0, max_iter=10000)),
    ])

def _build_xgboost():
    # No StandardScaler — matches XGB_prediction.ipynb
    return Pipeline([
        ("var", VarianceThreshold(threshold=0.0)),
        ("reg", XGBRegressor(**XGB_PARAMS)),
    ])

def _build_ebm():
    return Pipeline([
        ("var", VarianceThreshold(threshold=0.0).set_output(transform="pandas")),
        ("reg", ExplainableBoostingRegressor(
            random_state=42, n_jobs=-1,
            max_bins=128, interactions=5,
            max_rounds=500, learning_rate=0.05,
        )),
    ])

_BUILDERS = {
    "Ridge":   _build_ridge,
    "Lasso":   _build_lasso,
    "XGBoost": _build_xgboost,
    "EBM":     _build_ebm,
    "LightGBM": _build_lightgbm,
}


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(df, model_type="Ridge"):
    """
    Train a model on the feature-engineered dataframe.

    Parameters
    ----------
    df         : DataFrame output of build_features(), must contain 'sales'
    model_type : one of 'Ridge', 'Lasso', 'XGBoost', 'EBM'

    Returns
    -------
    (model, fcols) — fitted Pipeline and list of feature columns used
    """
    fcols = [c for c in FCOLS if c in df.columns]
    model = _BUILDERS[model_type]()
    model.fit(df[fcols], df["sales"])
    return model, fcols


def get_feature_importance(model, fcols):
    """
    Extract feature importances from any supported pipeline.

    Returns a list of (feature_name, importance_score) sorted by |score|.
    """
    support = model.named_steps["var"].get_support()
    kept    = [f for f, s in zip(fcols, support) if s]
    reg     = model.named_steps["reg"]

    if isinstance(reg, XGBRegressor):
        scores = reg.get_booster().get_score(importance_type="gain")
        result = [(f, scores.get(f"f{i}", 0.0)) for i, f in enumerate(kept)]
        return sorted(result, key=lambda x: abs(x[1]), reverse=True)

    if isinstance(reg, ExplainableBoostingRegressor):
        data   = reg.explain_global().data()
        names  = data["names"]
        scores = [float(s) for s in data["scores"]]
        return sorted(zip(names, scores), key=lambda x: abs(x[1]), reverse=True)

    # Ridge / Lasso
    coef = reg.coef_
    return sorted(
        [(f, float(c)) for f, c in zip(kept, coef)],
        key=lambda x: abs(x[1]), reverse=True,
    )
