"""
SHAP Utilities
==============
SHAP computation and natural-language explanation generation.

Source: SHAP_XGB_explainability.ipynb

Key difference from the old app.py
------------------------------------
SHAP is computed on VarianceThreshold-filtered but otherwise RAW (unscaled)
features — matching the notebook, which trains a plain XGBRegressor without
StandardScaler and calls shap.TreeExplainer directly on it.
"""

import numpy as np
import shap

from models import FEAT_LABELS
from feature_engineering import (
    WEATHER_COLS, HOLIDAY_COLS, COVID_COLS, SCHOOL_COLS,
)


def compute_shap(model, fcols, df):
    """
    Compute SHAP values for an XGBoost pipeline.

    Matches SHAP_XGB_explainability.ipynb §§ 1-2:
      - extracts the raw XGBRegressor from the pipeline
      - computes SHAP on VarianceThreshold-filtered, unscaled features

    Parameters
    ----------
    model  : fitted Pipeline with steps 'var' and 'reg'
    fcols  : list of feature column names passed to the pipeline
    df     : DataFrame containing those feature columns

    Returns
    -------
    pairs      : list of (feature_name, mean_abs_shap) sorted descending
    shap_vals  : raw SHAP value array, shape (n_rows, n_kept_features)
    kept       : list of feature names that survived VarianceThreshold
    """
    support   = model.named_steps["var"].get_support()
    kept      = [f for f, s in zip(fcols, support) if s]
    X_var     = model.named_steps["var"].transform(df[fcols])

    # Direct TreeExplainer on the raw regressor — matches the notebook
    explainer = shap.TreeExplainer(model.named_steps["reg"])
    shap_vals = explainer.shap_values(X_var)

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    pairs     = sorted(zip(kept, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs, shap_vals, kept


def generate_shap_text(pairs, fcast, hist, sel_pid):
    """
    Produce a natural-language explanation from SHAP feature importances.

    Parameters
    ----------
    pairs   : output of compute_shap() — list of (feature_name, mean_abs_shap)
    fcast   : forecast DataFrame with column 'forecast'
    hist    : history DataFrame with column 'sales'
    sel_pid : product ID string for display

    Returns
    -------
    HTML string describing the forecast direction and main drivers.
    """
    fc_avg   = fcast["forecast"].mean()
    prev_avg = hist["sales"].iloc[-4:].mean() if len(hist) >= 4 else hist["sales"].mean()
    pct      = (fc_avg - prev_avg) / prev_avg * 100 if prev_avg else 0
    direction = "increase" if pct > 0 else "decrease"

    top3 = [FEAT_LABELS.get(f, f) for f, _ in pairs[:3]]
    drivers_str = (
        top3[0] if len(top3) == 1
        else ", ".join(top3[:-1]) + f", and {top3[-1]}"
    )

    text = (
        f"Demand for Product {sel_pid} is forecast to <b>{direction} "
        f"by {abs(pct):.1f}%</b> versus the previous 4 periods. "
        f"The primary drivers identified by the model are: <b>{drivers_str}</b>. "
    )

    top6_feats    = [f for f, _ in pairs[:6]]
    lag_feats     = [f for f in top6_feats if "lag" in f]
    rolling_feats = [f for f in top6_feats if "rolling" in f]
    weather_feats = [f for f in top6_feats if f in WEATHER_COLS]
    holiday_feats = [f for f in top6_feats if f in HOLIDAY_COLS]
    covid_feats   = [f for f in top6_feats if f in COVID_COLS]
    school_feats  = [f for f in top6_feats if f in SCHOOL_COLS]
    trend_feats   = [f for f in top6_feats if f == "trend"]

    if lag_feats or rolling_feats:
        text += (
            "Recent sales history is the strongest signal — the model is largely "
            "extrapolating momentum from prior periods. "
        )
    if weather_feats:
        wlabel = FEAT_LABELS.get(weather_feats[0], weather_feats[0]).lower()
        text += (
            f"Weather conditions (especially {wlabel}) are contributing "
            "meaningfully to the forecast. "
        )
    if holiday_feats:
        text += "Upcoming or recent public holidays are also factored into the prediction. "
    if covid_feats:
        text += "COVID-related lockdown periods are influencing this forecast. "
    if school_feats:
        text += "School holiday periods are contributing to the expected demand shift. "
    if trend_feats:
        text += "A detectable long-term trend in this product's sales is shaping the outlook. "

    return text

def compute_shap_lightgbm(model, fcols, df):
    support = model.named_steps["var"].get_support()
    kept = [f for f, s in zip(fcols, support) if s]

    X_var = model.named_steps["var"].transform(df[fcols])

    explainer = shap.TreeExplainer(model.named_steps["reg"])
    shap_vals = explainer.shap_values(X_var)

    mean_abs = np.abs(shap_vals).mean(axis=0)

    pairs = sorted(
        zip(kept, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return pairs, shap_vals, kept
