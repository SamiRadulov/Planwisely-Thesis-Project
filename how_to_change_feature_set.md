# How to change which features YOUR model uses

Each model trains on **its own feature list**, so you can change your model's
features without affecting anyone else's. The lists live in **one place**:
`models.py`, in the `MODEL_FCOLS` dictionary.

Right now every model is set to the **reduced set** (`REDUCED_FCOLS`).

## Where it's defined (`models.py`)

```python
REDUCED_FCOLS = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "rolling_std_8",
    "week_of_year", "week_cos",
    "hol_easter", "hol_kings_day", "hol_pentecost", "hol_christmas",
    "temp_mean", "precip_sum", "sunshine_sum",
    "lockdown_days", "school_holiday",   # skipped if the COVID/school data is absent
]

MODEL_FCOLS = {
    "Ridge":    REDUCED_FCOLS,
    "Lasso":    REDUCED_FCOLS,
    "XGBoost":  REDUCED_FCOLS,
    "EBM":      REDUCED_FCOLS,
    "LightGBM": REDUCED_FCOLS,
}
```

## To change your model
Edit only **your** model's entry in `MODEL_FCOLS`. Three options:

1. **Reduced set (current default):**
   ```python
   "XGBoost": REDUCED_FCOLS,
   ```
2. **Full set (everything available, ~34 features):**
   ```python
   "XGBoost": FCOLS,
   ```
3. **Your own custom list:**
   ```python
   "XGBoost": ["lag_1", "lag_2", "rolling_mean_4", "temp_mean", "hol_easter"],
   ```

Save and restart Streamlit (`streamlit run app.py`). Only that model changes;
the others keep their own lists.

## Why it's safe
`train_model()` does `fcols = [c for c in MODEL_FCOLS[model_type] if c in df.columns]`,
so any feature you list that isn't in the engineered data is silently skipped (no
crash), and you never touch the model code itself — just the list.

## Note on `lockdown_days` and `school_holiday`
These are in `REDUCED_FCOLS` but the live dashboard does **not** build them (no
`covid_weekly.csv` / `school_holidays_weekly.csv` in the repo, so `HAS_COVID` /
`HAS_SCHOOL` are `False`). They are skipped automatically, leaving an effective
17-feature set. To use them for real, add those two weekly CSVs next to
`feature_engineering.py` (same format as `weather_weekly.csv`).

## Does the reduced set hurt accuracy?
No. On the EBM, pooled chronological 80/20 holdout (same setup as the thesis):

| Feature set | # | R^2 test | RMSE |
|---|---|---|---|
| Full (33) | 33 | 0.760 | 58.35 |
| Reduced (19) | 19 | 0.762 | 58.02 |

Statistically tied, so the reduced set is a fine, leaner default. Switch your
model to `FCOLS` any time if you'd rather use everything.
