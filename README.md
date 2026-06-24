# Planwisely: Explainable Demand Forecasting Dashboard

An interactive Streamlit dashboard that lets a non-technical planner upload a
sales history CSV, pick a forecasting model, and get a weekly demand forecast
together with a plain-language explanation of the drivers behind it.

The dashboard brings five models and three explanation methods into one
workflow, so accuracy and interpretability live in the same place. It was built
as the artifact for a BSc Business Analytics thesis on explainable demand
forecasting.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Input data format](#input-data-format)
4. [How the forecast is produced](#how-the-forecast-is-produced)
5. [Models](#models)
6. [Explainability](#explainability)
7. [Analysis views](#analysis-views)
8. [Per-model feature sets](#per-model-feature-sets)
9. [External data](#external-data)
10. [Project structure](#project-structure)
11. [Notes and limitations](#notes-and-limitations)

---

## What it does

- **Upload and go.** Drop in a CSV of historical sales. Feature engineering,
  external-data merging, and forecasting all run automatically.
- **Five models.** Ridge, Lasso, XGBoost, LightGBM, and an Explainable Boosting
  Machine (EBM).
- **Three explanation methods.** SHAP and LIME for the black-box tree models, and
  the EBM's own native (glass-box) explanations.
- **Per-product forecasts.** A four-week-ahead forecast for every product, with a
  history chart, model fit overlay, and a forecast band.
- **Year-over-year view.** Compares the forecast with the same weeks one year
  earlier and decomposes the change into per-feature drivers.
- **Portfolio summary.** Total demand, biggest movers, trajectory, and a CSV
  export across all products.

---

## Quick start

Requires Python 3.10 or newer.

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the dashboard
streamlit run app.py
```

Streamlit opens the app in your browser (usually http://localhost:8501).
Upload a CSV, choose a model, choose an explanation method (for XGBoost or
LightGBM), and click **Run**.

---

## Input data format

The dashboard auto-detects column roles, so most sales exports work without
editing. It looks for:

- **A product column:** named `product_id`, `product`, `sku`, or `item_id`
  (optional; if absent, the whole file is treated as one series).
- **A sales column:** named `sku_sold`, `sales`, `quantity`, `units`, `revenue`,
  or `amount` (if none match, the first numeric column is used). You can override
  the choice in the dashboard.
- **A date**, in either of two layouts:
  - a single date column (any parseable date format), or
  - three separate `year`, `month`, and `day` columns.

Example (single-date layout):

```csv
product_id,date,sku_sold
101,2021-01-04,38
101,2021-01-11,42
102,2021-01-04,17
```

Only products with **at least 52 weeks** of history are forecast. Daily data is
aggregated to weekly demand automatically.

---

## How the forecast is produced

1. **Weekly aggregation.** Each product's sales are summed into ISO weeks. The
   dashboard forecasts weekly regardless of whether the upload is daily or weekly.
2. **Filtering.** Products with fewer than 52 active weeks, or fewer than 20 rows
   after feature engineering, are skipped.
3. **Feature engineering** (`feature_engineering.py`): calendar features (month,
   week of year, cyclical sin/cos encodings), a trend index, lag features
   (`lag_1, 2, 4, 8`), rolling means and standard deviations (computed on a
   one-step shift to avoid target leakage), and any available external data.
4. **Training.** The selected model is trained per product on its history.
5. **Recursive forecast** (`forecast_4`): the next four weeks are predicted one at
   a time, feeding each prediction back as the lag for the following week. Each
   prediction is clipped to a sensible range to prevent runaway or negative
   values.

Note: the R-squared shown in the dashboard is an in-sample fit per product, a
descriptive indicator of how well the model fits that series. It is not a
held-out generalization score.

---

## Models

Defined in `models.py` (and `lightgbm_model.py`).

| Model | Type | Notes |
|-------|------|-------|
| Ridge | Linear (L2) | StandardScaler + Ridge(alpha=10) |
| Lasso | Linear (L1) | StandardScaler + Lasso(alpha=1) |
| XGBoost | Gradient-boosted trees | depth 5, 300 trees, lr 0.05 (no scaling needed) |
| LightGBM | Gradient-boosted trees | 1000 trees, lr 0.03, 31 leaves |
| EBM | Explainable Boosting Machine | Generalized Additive Model, glass-box, intrinsically interpretable |

All models run inside a scikit-learn `Pipeline` that first applies a
`VarianceThreshold` to drop constant features.

---

## Explainability

The explanation panel depends on the model:

- **XGBoost / LightGBM:** choose **SHAP** or **LIME** in the dashboard.
  - **SHAP** (`shap_utils.py`) uses TreeSHAP for fast, exact, signed feature
    contributions, shown as a global importance bar chart with a plain-language
    summary.
  - **LIME** (`lime_utils.py`) explains the single latest prediction with a local
    linear surrogate.
- **EBM:** shows its **native** glass-box feature importances, read directly from
  the model structure (no approximation needed).
- **Ridge / Lasso:** no XAI panel; the dashboard prompts you to pick a model that
  supports explanations.

---

## Analysis views

**Product Detail**
- Four period forecast tiles with change versus the previous four weeks.
- A history chart with actual sales, the in-sample model fit, the forecast, and a
  15 percent band.
- The explanation panel (SHAP / LIME / EBM native).
- **Prediction vs last year:** the four-week forecast compared with the same ISO
  weeks one year earlier, with a per-feature breakdown of what is driving the
  change. If the product has no recorded sales for that period last year, the
  view says so rather than guessing.

**Summary and Analysis**
- Portfolio key numbers, top products by volume, total demand across the four
  periods, biggest movers, a period-by-period trajectory, and a full CSV download.

---

## Per-model feature sets

Each model trains on its own feature list, defined in the `MODEL_FCOLS`
dictionary in `models.py`. Every model currently defaults to a reduced
19-feature set (`REDUCED_FCOLS`), which performs on par with the full set while
training faster.

To change which features your model uses, edit only that model's entry. See
**`how_to_change_feature_set.md`** for the three options (reduced set, full set,
or a custom list) and why it is safe (features not present in the data are
skipped automatically).

---

## External data

Feature engineering will merge any of these weekly files if they sit next to
`feature_engineering.py`:

- `weather_weekly.csv` (KNMI weather)
- `holiday_weekly.csv` (Dutch public holidays)
- `covid_weekly.csv` (COVID lockdown indicators)
- `school_holidays_weekly.csv` (Dutch school holidays)

If a file is missing, the corresponding feature group is skipped automatically
(no crash). A small pill in the dashboard header shows which external sources
were loaded.

---

## Project structure

```
app.py                      Streamlit dashboard (UI, forecasting loop, analysis views)
feature_engineering.py      Per-product weekly feature builder + external merges
models.py                   Model definitions, training, feature importance, MODEL_FCOLS
lightgbm_model.py           LightGBM pipeline builder
shap_utils.py               SHAP (TreeSHAP) computation + natural-language summary
lime_utils.py               LIME local explanation + natural-language summary
how_to_change_feature_set.md  Guide to per-model feature lists
model_features.json         Saved reduced feature list
requirements.txt            Python dependencies

Research notebooks (how the pipeline was built and validated):
  Preprocessing.ipynb, eda.ipynb, external_data_processing.ipynb,
  feature_engineering.ipynb, compare_models.py, SHAP_analysis.ipynb,
  XGBoost_prediction_model.ipynb

Data:
  sales_history.csv, sales_preprocessed_weekly.csv,
  features_final.csv, features_reduced.csv, weather_weekly.csv, holiday_weekly.csv
```

(The repository also contains supporting folders for the written report:
Business Understanding, Data Understanding, Modelling and Design, and others.)

---

## Notes and limitations

- **Recursive forecasting** feeds predictions back as inputs, so error compounds
  over the horizon. The dashboard forecasts four weeks and clips outputs to keep
  them realistic.
- **In-sample R-squared.** The per-product R-squared in the dashboard reflects
  fit, not out-of-sample accuracy. Rigorous accuracy was evaluated separately on a
  chronological holdout.
- **No zero-filling.** Weeks with no sales are not zero-filled, matching the
  weekly series used during development.
- The first run per model can be slower while features and explanations are
  computed; results are cached for repeated views.

---

## Requirements

See `requirements.txt`:

```
streamlit, pandas, numpy, scikit-learn, matplotlib, plotly,
xgboost, lightgbm, shap, lime, interpret
```
