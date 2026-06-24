"""
normalized_pearson_check.py
Reproduces the EBM native vs SHAP vs LIME importance vectors EXACTLY as in
run_xai_comparison.py (same config, seed, sample sizes), then compares the three
methods with Pearson and Spearman on:
  (a) raw importances
  (b) L1-normalized importances (x / sum)
  (c) z-scored importances ((x - mean) / std)
to show that linear normalization leaves Pearson unchanged.
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

def _find_data():
    _here = os.path.dirname(os.path.abspath(__file__))
    for _c in (os.path.join(_here, "features_final.csv"),
               os.path.join(os.path.dirname(_here), "features_final.csv")):
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError("features_final.csv not found in this folder or its parent.")
CSV = _find_data()
TARGET, DATE_COL = "sku_sold", "week_start"

FCOLS_BASE = ["trend","month","week_of_year","quarter","is_month_start","is_month_end",
              "month_sin","month_cos","week_sin","week_cos","lag_1","lag_2","lag_4","lag_8",
              "rolling_mean_4","rolling_mean_8","rolling_std_4","rolling_std_8"]
EXT_WEATHER = ["temp_mean","temp_min","temp_max","precip_sum","sunshine_sum","temp_anomaly","heavy_rain"]
EXT_HOLIDAY = ["has_holiday","min_days_to_holiday","hol_ascension","hol_christmas","hol_easter",
               "hol_kings_day","hol_liberation_day","hol_new_year","hol_pentecost"]

df = pd.read_csv(CSV)
if "quarter" not in df.columns and "month" in df.columns:
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
FCOLS = [c for c in FCOLS_BASE + EXT_WEATHER + EXT_HOLIDAY if c in df.columns]
df = df.dropna(subset=FCOLS + [TARGET])
if DATE_COL in df.columns:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
split = int(len(df) * 0.8)
X_train, X_test = df[FCOLS].iloc[:split].reset_index(drop=True), df[FCOLS].iloc[split:].reset_index(drop=True)
y_train = df[TARGET].iloc[:split].reset_index(drop=True)
print(f"Features: {len(FCOLS)} | train: {len(X_train)} | test: {len(X_test)}")

from interpret.glassbox import ExplainableBoostingRegressor
ebm = ExplainableBoostingRegressor(random_state=RNG, n_jobs=-1, max_bins=128, interactions=5)
ebm.fit(X_train, y_train)

N_GLOBAL = min(300, len(X_test))
N_LIME   = min(150, len(X_test))
Xs = X_test.sample(N_GLOBAL, random_state=RNG).reset_index(drop=True)

# native
loc = ebm.explain_local(Xs, ebm.predict(Xs))
native = np.zeros(len(FCOLS))
for i in range(len(Xs)):
    d = loc.data(i)
    for nm, sc in zip(d["names"], d["scores"]):
        if nm in FCOLS:
            native[FCOLS.index(nm)] += abs(float(sc))
native /= len(Xs)

# shap
import shap
bg = shap.sample(X_train, 100, random_state=RNG)
shap_vals = shap.Explainer(ebm.predict, bg)(Xs)
shap_imp = np.abs(shap_vals.values).mean(axis=0)

# lime
from lime.lime_tabular import LimeTabularExplainer
lx = LimeTabularExplainer(training_data=X_train.values, feature_names=FCOLS,
                          mode="regression", discretize_continuous=True, random_state=RNG)
lime_imp = np.zeros(len(FCOLS))
for i in range(N_LIME):
    e = lx.explain_instance(Xs.iloc[i].values, ebm.predict, num_features=len(FCOLS))
    for idx, w in e.local_exp[1]:
        lime_imp[idx] += abs(w)
lime_imp /= N_LIME

imp = pd.DataFrame({"feature": FCOLS, "native": native, "shap": shap_imp, "lime": lime_imp})
imp.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "importance_vectors.csv"), index=False)

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float("nan") if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0,1])
def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(-np.asarray(x,float))).astype(float)
    return pearson(r(a), r(b))
l1 = lambda x: np.asarray(x,float)/np.asarray(x,float).sum()
z  = lambda x: (np.asarray(x,float)-np.asarray(x,float).mean())/np.asarray(x,float).std()

pairs = [("SHAP","native"),("LIME","native"),("LIME","SHAP")]
col = {"native":imp.native,"SHAP":imp.shap,"LIME":imp.lime}
print("\n%-14s %8s %8s %8s %9s" % ("pair","Pear_raw","Pear_L1","Pear_z","Spearman"))
for a,b in pairs:
    A,B = col[a], col[b]
    print("%-14s %8.3f %8.3f %8.3f %9.3f" % (
        f"{a} vs {b}", pearson(A,B), pearson(l1(A),l1(B)), pearson(z(A),z(B)), spearman(A,B)))
print("\n(Pear_raw = Pear_L1 = Pear_z by construction: Pearson is invariant to linear rescaling.)")
