"""
LIME Analysis: Global, Local, Stability & Faithfulness
======================================================
Parallel to SHAP_analysis.ipynb, but for LIME on the XGBoost model.

Key difference from SHAP:
  - SHAP is deterministic and exactly additive, so its stability/completeness
    checks are trivially perfect.
  - LIME uses random perturbation sampling and an APPROXIMATE local surrogate,
    so stability (across reruns) and faithfulness (local fit quality) are real,
    measured quantities. These are the most informative LIME analyses.

LIME is local by nature; a "global" view is obtained by aggregating the
absolute local weights over a sample of instances.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

os.makedirs("lime_output", exist_ok=True)
os.makedirs("lime_analysis", exist_ok=True)

rng = np.random.RandomState(42)

# ── 1. Load & setup (mirrors SHAP notebook) ─────────────────────────────────
df = pd.read_csv("features_final.csv")
df["week_start"] = pd.to_datetime(df["week_start"])
df = df.sort_values(["week_start", "product_id"]).reset_index(drop=True)

TARGET    = "sku_sold"
DROP_COLS = ["product_id", "week_start", "year", "week", TARGET]
FEATURES  = [c for c in df.columns if c not in DROP_COLS]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, random_state=42, verbosity=0,
)

FOLD_SPLITS = [
    ("2018-02-19", "2019-12-30", "2020-01-06", "2020-03-30"),
    ("2018-02-19", "2020-03-30", "2020-04-06", "2020-06-29"),
    ("2018-02-19", "2020-06-29", "2020-07-06", "2020-09-28"),
    ("2018-02-19", "2020-09-28", "2020-10-05", "2020-12-28"),
]

train_final = df[df["week_start"] <= "2020-12-28"]
test_final  = df[(df["week_start"] >= "2021-01-01") & (df["week_start"] <= "2021-12-31")]
test_reset  = test_final.reset_index(drop=True)

final_model = xgb.XGBRegressor(**XGB_PARAMS)
final_model.fit(train_final[FEATURES], train_final[TARGET].values)

# One explainer over the training distribution; reused everywhere.
NUM_SAMPLES = 1000   # perturbations per instance (default 5000; lowered for speed)

def make_explainer(train_X):
    return LimeTabularExplainer(
        training_data=train_X.values,
        feature_names=FEATURES,
        mode="regression",
        discretize_continuous=False,
        random_state=42,
    )

def lime_weights(explainer, model, row_values, seed=None, num_samples=NUM_SAMPLES):
    """Return a dict {feature: signed weight} for one instance."""
    if seed is not None:
        explainer.random_state = np.random.RandomState(seed)
    exp = explainer.explain_instance(
        data_row=row_values,
        predict_fn=model.predict,
        num_features=len(FEATURES),
        num_samples=num_samples,
    )
    # discretize_continuous=False -> as_list gives (feature_name, weight)
    return dict(exp.as_list()), exp.score

final_expl = make_explainer(train_final[FEATURES])
print(f"Features: {len(FEATURES)}  |  2021 holdout rows: {len(test_reset)}")

# ── 2. Global LIME — aggregate |weight| over a sample ───────────────────────
GLOBAL_N = 250
samp_idx = rng.choice(len(test_reset), size=min(GLOBAL_N, len(test_reset)), replace=False)

global_abs = pd.DataFrame(0.0, index=range(len(samp_idx)), columns=FEATURES)
local_scores = []
for i, ridx in enumerate(samp_idx):
    w, score = lime_weights(final_expl, final_model, test_reset.loc[ridx, FEATURES].values)
    for f, val in w.items():
        global_abs.loc[i, f] = abs(val)
    local_scores.append(score)

global_mean = global_abs.mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
top20 = global_mean.head(20)[::-1]
plt.barh(top20.index, top20.values, color="#2563EB")
plt.xlabel("Mean |LIME weight| (sampled 2021 instances)")
plt.title("Global Feature Importance (mean |LIME|) — XGBoost (2021 holdout)")
plt.tight_layout()
plt.savefig("lime_output/global_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lime_output/global_bar.png")

# ── 3. Local LIME — signed bar for 3 sample products ────────────────────────
product_means = test_final.groupby("product_id")[TARGET].mean().sort_values()
sample_products = {
    "high_volume": product_means.index[-1],
    "median":      product_means.index[len(product_means) // 2],
    "low_volume":  product_means.index[0],
}

for label, pid in sample_products.items():
    rows = test_reset[test_reset["product_id"] == pid]
    if rows.empty:
        continue
    ridx = rows.index[-1]
    w, score = lime_weights(final_expl, final_model, test_reset.loc[ridx, FEATURES].values)
    s = pd.Series(w).reindex(global_mean.index).dropna()
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(15)[::-1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in s.values]

    plt.figure(figsize=(8, 6))
    plt.barh(s.index, s.values, color=colors)
    plt.axvline(0, color="black", lw=0.6)
    plt.xlabel("LIME weight (green = pushes up, red = pulls down)")
    plt.title(f"Local LIME — product {pid} ({label}), last week 2021\n"
              f"local surrogate $R^2$ = {score:.3f}")
    plt.tight_layout()
    fname = f"lime_output/local_bar_{label}_product{pid}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ── 4. Faithfulness — distribution of local surrogate R² ────────────────────
scores = np.array(local_scores)
plt.figure(figsize=(8, 4.5))
plt.hist(scores, bins=25, color="#2563EB", edgecolor="white")
plt.axvline(scores.mean(), color="red", ls="--",
            label=f"mean = {scores.mean():.3f}")
plt.xlabel("Local surrogate $R^2$ (LIME faithfulness)")
plt.ylabel("Number of instances")
plt.title("LIME Faithfulness — Local Surrogate Fit Across 2021 Instances")
plt.legend()
plt.tight_layout()
plt.savefig("lime_analysis/faithfulness_hist.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: lime_analysis/faithfulness_hist.png  (mean R2={scores.mean():.3f})")

# ── 5. Stability across REPEATED RUNS (the key LIME-specific check) ──────────
# SHAP would give identical results; LIME varies because of random sampling.
N_REPEAT = 30
stab_pid = sample_products["median"]
stab_ridx = test_reset[test_reset["product_id"] == stab_pid].index[-1]
stab_row = test_reset.loc[stab_ridx, FEATURES].values

repeat_w = []
for r in range(N_REPEAT):
    w, _ = lime_weights(final_expl, final_model, stab_row, seed=1000 + r)
    repeat_w.append(pd.Series(w).reindex(FEATURES))
repeat_df = pd.DataFrame(repeat_w)

# Rank correlation between every pair of runs (top features)
top_feats = repeat_df.abs().mean().sort_values(ascending=False).head(10).index
rank_runs = repeat_df[top_feats].rank(axis=1, ascending=False)
corrs = []
for a in range(N_REPEAT):
    for b in range(a + 1, N_REPEAT):
        corrs.append(spearmanr(rank_runs.iloc[a], rank_runs.iloc[b]).correlation)
mean_pair_corr = np.nanmean(corrs)

plt.figure(figsize=(9, 5))
means = repeat_df[top_feats].mean()
stds  = repeat_df[top_feats].std()
order = means.abs().sort_values(ascending=False).index[::-1]
plt.barh(order, means[order].values, xerr=stds[order].values,
         color="#2563EB", ecolor="#e74c3c", capsize=3)
plt.axvline(0, color="black", lw=0.6)
plt.xlabel("LIME weight (mean ± std over 30 reruns)")
plt.title(f"LIME Stability Across {N_REPEAT} Reruns — product {stab_pid}\n"
          f"mean pairwise rank correlation = {mean_pair_corr:.3f}")
plt.tight_layout()
plt.savefig("lime_analysis/stability_reruns.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: lime_analysis/stability_reruns.png  (mean rank corr={mean_pair_corr:.3f})")

# ── 6. Stability across FOLDS — rank heatmap (mirrors SHAP) ──────────────────
all_splits  = FOLD_SPLITS + [("2018-02-19", "2020-12-28", "2021-01-01", "2021-12-31")]
label_names = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Final 2021"]
FOLD_N = 120

fold_means, fold_ranks = [], []
for label, (trs, tre, tes, tee) in zip(label_names, all_splits):
    tr = df[(df["week_start"] >= trs) & (df["week_start"] <= tre)]
    te = df[(df["week_start"] >= tes) & (df["week_start"] <= tee)].reset_index(drop=True)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(tr[FEATURES], tr[TARGET].values)
    expl = make_explainer(tr[FEATURES])

    idx = rng.choice(len(te), size=min(FOLD_N, len(te)), replace=False)
    acc = pd.Series(0.0, index=FEATURES)
    for ridx in idx:
        w, _ = lime_weights(expl, m, te.loc[ridx, FEATURES].values)
        acc += pd.Series(w).reindex(FEATURES).abs().fillna(0)
    acc /= len(idx)
    fold_means.append(acc)
    fold_ranks.append(acc.rank(ascending=False).astype(int))
    print(f"{label} done — sampled {len(idx)} rows")

means_df = pd.DataFrame(fold_means, index=label_names)
ranks_df = pd.DataFrame(fold_ranks, index=label_names)

mean_rank = ranks_df.mean().sort_values()
top_features = mean_rank.head(20).index.tolist()
plot_ranks = ranks_df[top_features]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(plot_ranks.values, aspect="auto", cmap="RdYlGn_r",
               vmin=1, vmax=len(FEATURES))
plt.colorbar(im, ax=ax, label="Rank (1 = most important)")
ax.set_xticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(label_names)))
ax.set_yticklabels(label_names)
for i in range(len(label_names)):
    for j in range(len(top_features)):
        ax.text(j, i, str(plot_ranks.values[i, j]),
                ha="center", va="center", fontsize=8, color="black")
ax.set_title("LIME Feature Rank Stability — Top 20 Features across Folds")
plt.tight_layout()
plt.savefig("lime_analysis/stability_rank_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lime_analysis/stability_rank_heatmap.png")

# ── 7. Stability — rank summary table ───────────────────────────────────────
stability_df = pd.DataFrame({
    "mean_rank": ranks_df.mean(),
    "std_rank":  ranks_df.std(),
    "min_rank":  ranks_df.min(),
    "max_rank":  ranks_df.max(),
}).sort_values("mean_rank").round(2)
stability_df.to_csv("lime_analysis/stability_summary.csv")
print("\nFeature rank stability (lower std = more stable):")
print(stability_df.head(15).to_string())

# ── 8. Export ───────────────────────────────────────────────────────────────
global_mean.to_csv("lime_output/global_importance.csv", header=["mean_abs_weight"])
pd.Series(scores).describe().to_csv("lime_analysis/faithfulness_summary.csv")
repeat_df.to_csv("lime_analysis/rerun_weights.csv", index=False)

print("\nDone. Key results:")
print(f"  Global top-5 features : {list(global_mean.head(5).index)}")
print(f"  Mean local R^2 (faith): {scores.mean():.3f}  (min {scores.min():.3f}, max {scores.max():.3f})")
print(f"  Rerun rank corr       : {mean_pair_corr:.3f}")
