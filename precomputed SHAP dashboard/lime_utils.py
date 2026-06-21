"""
LIME Utilities
==============
Local Interpretable Model-agnostic Explanations (LIME) for XGBoost and LightGBM.

LIME explains a single prediction by fitting a local linear surrogate model
around the input, in contrast to SHAP which gives a global/aggregate view.

Usage: call compute_lime() with the fitted pipeline, feature column list,
and a single-row DataFrame of features for the prediction you want to explain.
"""

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from models import FEAT_LABELS


def compute_lime(model, fcols, train_df, predict_row_df, num_features=10):
    """
    Compute LIME explanation for a single prediction row.

    Parameters
    ----------
    model         : fitted sklearn Pipeline with steps 'var' and 'reg'
    fcols         : list of feature column names passed to the pipeline
    train_df      : DataFrame used for training (background distribution for LIME)
    predict_row_df: single-row DataFrame with the features to explain
    num_features  : number of top features to show (default 10)

    Returns
    -------
    pairs  : list of (feature_name, weight) sorted by abs weight descending
    """
    # Filter to features that exist in the data
    available_fcols = [c for c in fcols if c in train_df.columns]

    # Apply VarianceThreshold transform to get the kept features
    support = model.named_steps["var"].get_support()
    kept = [f for f, s in zip(available_fcols, support) if s]

    # Transform training data through VarianceThreshold only (no scaling for tree models)
    X_train_var = model.named_steps["var"].transform(train_df[available_fcols])

    # The regressor (XGBRegressor or LGBMRegressor) is a tree model — predict directly
    regressor = model.named_steps["reg"]

    def predict_fn(X):
        return regressor.predict(X)

    explainer = LimeTabularExplainer(
        training_data=X_train_var,
        feature_names=kept,
        mode="regression",
        discretize_continuous=False,   # keep continuous for time-series features
        random_state=42,
    )

    # Transform the single prediction row
    row_var = model.named_steps["var"].transform(predict_row_df[available_fcols])

    explanation = explainer.explain_instance(
        data_row=row_var[0],
        predict_fn=predict_fn,
        num_features=num_features,
    )

    # explanation.as_list() returns [(feature_condition_str, weight), ...]
    # We map back to clean feature names
    raw_pairs = explanation.as_list()

    # LIME returns condition strings like "lag_1 > 120.5" — extract the feature name
    pairs = []
    for condition_str, weight in raw_pairs:
        # Find which kept feature name appears in this condition string
        matched = None
        for feat in kept:
            if feat in condition_str:
                matched = feat
                break
        label = FEAT_LABELS.get(matched, matched) if matched else condition_str
        pairs.append((label, float(weight)))

    # Sort by absolute weight descending
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return pairs


def generate_lime_text(pairs, fcast, hist, sel_pid):
    """
    Produce a natural-language explanation from LIME feature weights.

    Parameters
    ----------
    pairs   : output of compute_lime() — list of (label, weight)
    fcast   : forecast DataFrame with column 'forecast'
    hist    : history DataFrame with column 'sales'
    sel_pid : product ID string for display

    Returns
    -------
    HTML string describing the local LIME explanation.
    """
    fc_avg   = fcast["forecast"].mean()
    prev_avg = hist["sales"].iloc[-4:].mean() if len(hist) >= 4 else hist["sales"].mean()
    pct      = (fc_avg - prev_avg) / prev_avg * 100 if prev_avg else 0
    direction = "increase" if pct >= 0 else "decrease"

    top3 = [label for label, _ in pairs[:3]]
    if len(top3) == 1:
        drivers_str = top3[0]
    else:
        drivers_str = ", ".join(top3[:-1]) + f", and {top3[-1]}"

    # Positive weights push forecast up, negative weights pull it down
    pushers = [label for label, w in pairs[:6] if w > 0]
    pullers = [label for label, w in pairs[:6] if w < 0]

    text = (
        f"For this specific forecast, LIME identified the most influential local drivers. "
        f"Demand is expected to <b>{direction} by {abs(pct):.1f}%</b> vs. the prior 4 periods. "
        f"The primary local drivers are: <b>{drivers_str}</b>. "
    )

    if pushers:
        text += f"Features pushing the forecast <b>up</b>: {', '.join(pushers[:3])}. "
    if pullers:
        text += f"Features pulling the forecast <b>down</b>: {', '.join(pullers[:3])}. "

    text += (
        "Note: LIME explains this individual prediction locally, "
        "which may differ from the global SHAP feature importances shown above."
    )

    return text
