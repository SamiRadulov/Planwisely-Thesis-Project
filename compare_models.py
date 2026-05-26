import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("features_final.csv")

TARGET   = "sku_sold"   # adjust if your sales column is named differently
DATE_COL = "date"       # adjust if needed

# ── Features (same list as app.py) ───────────────────────────────────────────
FCOLS_BASE = [
    "trend", "month", "week_of_year", "quarter",
    "is_month_start", "is_month_end",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "rolling_std_8",
]
EXT_WEATHER = ["temp_mean","temp_min","temp_max","precip_sum",
               "sunshine_sum","temp_anomaly","heavy_rain"]
EXT_HOLIDAY = ["has_holiday","min_days_to_holiday","hol_ascension",
               "hol_christmas","hol_easter","hol_kings_day",
               "hol_liberation_day","hol_new_year","hol_pentecost"]

FCOLS = [c for c in FCOLS_BASE + EXT_WEATHER + EXT_HOLIDAY if c in df.columns]

print(f"Features used: {len(FCOLS)}")
print(f"  Base: {len([c for c in FCOLS_BASE if c in df.columns])}")
print(f"  Weather: {len([c for c in EXT_WEATHER if c in df.columns])}")
print(f"  Holiday: {len([c for c in EXT_HOLIDAY if c in df.columns])}")

# ── Drop rows with missing values in features or target ───────────────────────
df = df.dropna(subset=FCOLS + [TARGET])
if DATE_COL in df.columns:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

print(f"Total rows after cleaning: {len(df)}\n")

# ── Train / test split — last 20% as hold-out ────────────────────────────────
split    = int(len(df) * 0.8)
X_train  = df[FCOLS].iloc[:split]
X_test   = df[FCOLS].iloc[split:]
y_train  = df[TARGET].iloc[:split]
y_test   = df[TARGET].iloc[split:]

print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}\n")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Ridge": Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=10.0)),
    ]),
    "Lasso": Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    Lasso(alpha=1.0, max_iter=10000)),
    ]),
    "XGBoost": Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )),
    ]),
    "EBM": Pipeline([
        ("var", VarianceThreshold(threshold=0.0).set_output(transform="pandas")),
        ("reg", ExplainableBoostingRegressor(random_state=42, n_jobs=-1,
                                              max_bins=128, interactions=5)),
    ]),
}

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = {}
header  = f"\n{'Model':<12} {'R² train':>9} {'R² test':>9} {'MAE test':>10} {'RMSE test':>10} {'Overfit':>8}"
print(header)
print("-" * len(header.strip()))

for name, model in models.items():
    print(f"  Training {name}...", end="\r")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    r2_tr  = r2_score(y_train, train_pred)
    r2_te  = r2_score(y_test,  test_pred)
    mae    = mean_absolute_error(y_test, test_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, test_pred))
    overfit = r2_tr - r2_te   # how much worse on unseen data

    results[name] = dict(r2_train=r2_tr, r2_test=r2_te, mae=mae, rmse=rmse, overfit=overfit)
    print(f"{name:<12} {r2_tr:>9.4f} {r2_te:>9.4f} {mae:>10.2f} {rmse:>10.2f} {overfit:>+8.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
best_r2   = max(results, key=lambda m: results[m]["r2_test"])
best_mae  = min(results, key=lambda m: results[m]["mae"])
best_rmse = min(results, key=lambda m: results[m]["rmse"])
least_of  = min(results, key=lambda m: results[m]["overfit"])

print("\n── Best model per metric ──────────────────────────────")
print(f"  Highest R² (test):  {best_r2}   ({results[best_r2]['r2_test']:.4f})")
print(f"  Lowest MAE:         {best_mae}  ({results[best_mae]['mae']:.2f})")
print(f"  Lowest RMSE:        {best_rmse}  ({results[best_rmse]['rmse']:.2f})")
print(f"  Least overfitting:  {least_of}  (gap={results[least_of]['overfit']:+.4f})")
