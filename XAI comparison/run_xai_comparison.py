"""
run_xai_comparison.py
=====================
Compares three explanation methods on the SAME Explainable Boosting Machine
(EBM):

  1. EBM native shape-function contributions   (intrinsic / ante-hoc)
  2. SHAP                                       (post-hoc, model-agnostic)
  3. LIME                                       (post-hoc, local surrogate)

It also reproduces the multi-model accuracy comparison (Ridge / Lasso /
XGBoost / LightGBM / EBM) from one place.

WHY THIS MATTERS
----------------
The research question is whether XAI methods give *meaningful* interpretation.
EBM's explanations are exact by construction. If SHAP and LIME, the standard
post-hoc methods applied to black-box models, agree with EBM's exact
attributions, that is direct evidence that (a) the explanations are
trustworthy and (b) a black-box-plus-post-hoc pipeline buys you nothing here
that the intrinsically interpretable EBM does not already provide exactly.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

HERE      = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(HERE, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA  (mirrors compare_models.py so accuracy numbers reproduce exactly)
# ──────────────────────────────────────────────────────────────────────────────
TARGET   = "sku_sold"
DATE_COL = "week_start"

FCOLS_BASE = [
    "trend", "month", "week_of_year", "quarter",
    "is_month_start", "is_month_end",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "rolling_std_8",
]
EXT_WEATHER = ["temp_mean", "temp_min", "temp_max", "precip_sum",
               "sunshine_sum", "temp_anomaly", "heavy_rain"]
EXT_HOLIDAY = ["has_holiday", "min_days_to_holiday", "hol_ascension",
               "hol_christmas", "hol_easter", "hol_kings_day",
               "hol_liberation_day", "hol_new_year", "hol_pentecost"]

# Optional human-readable labels for nicer plots (falls back to raw name)
LABELS = {
    "lag_1": "Last week sales", "lag_2": "Sales 2w ago", "lag_4": "Sales 4w ago",
    "lag_8": "Sales 8w ago", "rolling_mean_4": "4w avg", "rolling_mean_8": "8w avg",
    "rolling_std_4": "4w volatility", "rolling_std_8": "8w volatility",
    "trend": "Long-term trend", "month": "Month", "week_of_year": "Week of year",
    "quarter": "Quarter", "temp_mean": "Avg temp", "temp_anomaly": "Temp anomaly",
    "hol_easter": "Easter", "min_days_to_holiday": "Days to holiday",
    "has_holiday": "Holiday week", "sunshine_sum": "Sunshine",
}
lab = lambda c: LABELS.get(c, c)

def _find_data():
    for _c in (os.path.join(HERE, "features_final.csv"),
               os.path.join(os.path.dirname(HERE), "features_final.csv")):
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError("features_final.csv not found in this folder or its parent.")
df = pd.read_csv(_find_data())
if "quarter" not in df.columns and "month" in df.columns:
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

FCOLS = [c for c in FCOLS_BASE + EXT_WEATHER + EXT_HOLIDAY if c in df.columns]
df = df.dropna(subset=FCOLS + [TARGET])
if DATE_COL in df.columns:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

split   = int(len(df) * 0.8)
X_train = df[FCOLS].iloc[:split].reset_index(drop=True)
X_test  = df[FCOLS].iloc[split:].reset_index(drop=True)
y_train = df[TARGET].iloc[:split].reset_index(drop=True)
y_test  = df[TARGET].iloc[split:].reset_index(drop=True)

print(f"Features: {len(FCOLS)} | train rows: {len(X_train)} | test rows: {len(X_test)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  MULTI-MODEL ACCURACY TABLE  (reproduces the comparison)
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from interpret.glassbox import ExplainableBoostingRegressor

pipes = {
    "Ridge": Pipeline([("var", VarianceThreshold()), ("sc", StandardScaler()),
                       ("reg", Ridge(alpha=10.0))]),
    "Lasso": Pipeline([("var", VarianceThreshold()), ("sc", StandardScaler()),
                       ("reg", Lasso(alpha=1.0, max_iter=10000))]),
    "XGBoost": Pipeline([("var", VarianceThreshold()),
                         ("reg", XGBRegressor(n_estimators=300, max_depth=5,
                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, random_state=RNG, verbosity=0))]),
    "LightGBM": Pipeline([("var", VarianceThreshold()),
                          ("reg", LGBMRegressor(objective="regression",
                           n_estimators=1000, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=RNG, verbose=-1))]),
    "EBM": Pipeline([("var", VarianceThreshold().set_output(transform="pandas")),
                     ("reg", ExplainableBoostingRegressor(random_state=RNG,
                      n_jobs=-1, max_bins=128, interactions=5))]),
}

rows = []
for name, p in pipes.items():
    p.fit(X_train, y_train)
    r2_tr = r2_score(y_train, p.predict(X_train))
    r2_te = r2_score(y_test,  p.predict(X_test))
    mae   = mean_absolute_error(y_test, p.predict(X_test))
    rmse  = np.sqrt(mean_squared_error(y_test, p.predict(X_test)))
    rows.append([name, r2_tr, r2_te, mae, rmse, r2_tr - r2_te])
metrics = pd.DataFrame(rows, columns=["model", "r2_train", "r2_test",
                                      "mae", "rmse", "overfit_gap"])
metrics.to_csv(os.path.join(FIG_DIR, "model_metrics.csv"), index=False)
print("\n=== Four-model accuracy (test set) ===")
print(metrics.round(3).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 3.  ONE EBM, THREE EXPLANATIONS
#     Fit a clean EBM directly on the DataFrame so feature names propagate to
#     SHAP and LIME without any pipeline plumbing.
# ──────────────────────────────────────────────────────────────────────────────
ebm = ExplainableBoostingRegressor(random_state=RNG, n_jobs=-1,
                                   max_bins=128, interactions=5)
ebm.fit(X_train, y_train)

# Sample the test set for the (slower) post-hoc methods
N_GLOBAL = min(300, len(X_test))
N_LIME   = min(150, len(X_test))
Xs = X_test.sample(N_GLOBAL, random_state=RNG).reset_index(drop=True)

# --- 3a. EBM NATIVE: mean |local contribution| per main-effect feature --------
loc = ebm.explain_local(Xs, ebm.predict(Xs))
native = np.zeros(len(FCOLS))
for i in range(len(Xs)):
    d = loc.data(i)
    for nm, sc in zip(d["names"], d["scores"]):
        if nm in FCOLS:                       # ignore pairwise interaction terms
            native[FCOLS.index(nm)] += abs(float(sc))
native /= len(Xs)

# --- 3b. SHAP (post-hoc, model-agnostic) --------------------------------------
import shap
bg = shap.sample(X_train, 100, random_state=RNG)
shap_expl = shap.Explainer(ebm.predict, bg)
shap_vals = shap_expl(Xs)
shap_imp  = np.abs(shap_vals.values).mean(axis=0)

# --- 3c. LIME (post-hoc, local linear surrogate) ------------------------------
from lime.lime_tabular import LimeTabularExplainer
lime_expl = LimeTabularExplainer(
    training_data=X_train.values, feature_names=FCOLS,
    mode="regression", discretize_continuous=True, random_state=RNG)
lime_imp = np.zeros(len(FCOLS))
for i in range(N_LIME):
    e = lime_expl.explain_instance(Xs.iloc[i].values, ebm.predict,
                                   num_features=len(FCOLS))
    for idx, w in e.local_exp[1]:
        lime_imp[idx] += abs(w)
lime_imp /= N_LIME

imp = pd.DataFrame({"feature": FCOLS, "native": native,
                    "shap": shap_imp, "lime": lime_imp})

# ──────────────────────────────────────────────────────────────────────────────
# 4.  AGREEMENT METRICS  (Spearman rank + Pearson, implemented in numpy)
# ──────────────────────────────────────────────────────────────────────────────
def _rank(a):
    order = np.argsort(np.argsort(-np.asarray(a, float)))
    return order.astype(float)

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def spearman(a, b):
    return pearson(_rank(a), _rank(b))

sp_sn = spearman(imp.shap,  imp.native)
sp_ln = spearman(imp.lime,  imp.native)
sp_ls = spearman(imp.lime,  imp.shap)
pe_sn = pearson(imp.shap,   imp.native)

print("\n=== Global agreement on EBM (rank correlation of feature importance) ===")
print(f"  SHAP vs EBM-native : Spearman {sp_sn:.3f} | Pearson {pe_sn:.3f}")
print(f"  LIME vs EBM-native : Spearman {sp_ln:.3f}")
print(f"  LIME vs SHAP       : Spearman {sp_ls:.3f}")

top = imp.sort_values("native", ascending=False).head(10)
print("\nTop-10 features by EBM-native importance:")
print(top.round(3).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 5.  FIGURES
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _norm(x):
    x = np.asarray(x, float); s = x.sum()
    return x / s if s else x

# 5a. global grouped bar (top 10 by native, importances normalised to compare shape)
g = imp.sort_values("native", ascending=False).head(10).iloc[::-1]
y = np.arange(len(g)); h = 0.26
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(y + h, _norm(g.native), h, label="EBM native", color="#1b3a5b")
ax.barh(y,      _norm(g.shap),   h, label="SHAP",       color="#3d7ab8")
ax.barh(y - h, _norm(g.lime),   h, label="LIME",        color="#9ec6e6")
ax.set_yticks(y); ax.set_yticklabels([lab(c) for c in g.feature])
ax.set_xlabel("Relative importance (normalised)")
ax.set_title("Global feature importance on the EBM: native vs SHAP vs LIME")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "xai_global_bar.png"), dpi=200)
plt.close(fig)

# 5b. local example: one test instance, three attributions, top 8 by |native|
i0 = 0
d  = loc.data(i0)
native_local = {nm: float(sc) for nm, sc in zip(d["names"], d["scores"]) if nm in FCOLS}
sv_local     = dict(zip(FCOLS, shap_vals.values[i0]))
el           = lime_expl.explain_instance(Xs.iloc[i0].values, ebm.predict,
                                          num_features=len(FCOLS))
lime_local   = {FCOLS[idx]: w for idx, w in el.local_exp[1]}
feats8 = sorted(native_local, key=lambda k: abs(native_local[k]), reverse=True)[:8][::-1]
yy = np.arange(len(feats8))
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(yy + h, [native_local.get(f, 0) for f in feats8], h, label="EBM native", color="#1b3a5b")
ax.barh(yy,      [sv_local.get(f, 0)     for f in feats8], h, label="SHAP",       color="#3d7ab8")
ax.barh(yy - h, [lime_local.get(f, 0)   for f in feats8], h, label="LIME",        color="#9ec6e6")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([lab(c) for c in feats8])
ax.set_xlabel("Contribution to this prediction (units of demand)")
ax.set_title("Local explanation for one product-week: native vs SHAP vs LIME")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "xai_local_example.png"), dpi=200)
plt.close(fig)
