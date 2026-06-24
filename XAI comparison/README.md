# XAI Comparison: EBM Native vs SHAP vs LIME

Compares three explanation methods on the same Explainable Boosting Machine (EBM)
for weekly demand forecasting:

1. **EBM native** shape-function contributions (intrinsic, exact)
2. **SHAP** (post-hoc, model-agnostic)
3. **LIME** (post-hoc, local linear surrogate)

The EBM's native explanation is exact by construction, so it serves as a ground
truth for testing whether the post-hoc methods recover the same feature drivers.

## Files

| File | Description |
|------|-------------|
| `run_xai_comparison.py` | Main script: trains the models, explains one EBM three ways, computes agreement (Spearman / Pearson), and writes the figures. |
| `normalized_pearson_check.py` | Supplementary check: compares the importance vectors with Pearson on raw, L1-normalized, and z-scored weights, showing Pearson is invariant to linear rescaling. |
| `Figures/model_metrics.csv` | Accuracy table for all models on the held-out test set. |
| `Figures/xai_global_bar.png` | Top-10 global feature importance: native vs SHAP vs LIME. |
| `Figures/xai_local_example.png` | Local explanation for one product-week: native vs SHAP vs LIME. |
| `importance_vectors.csv` | Per-feature native / SHAP / LIME importances. |

## Requirements

```
pip install numpy pandas scikit-learn xgboost lightgbm interpret shap lime matplotlib
```

## Data

Both scripts expect `features_final.csv` (the engineered weekly dataset) in this
folder or in its parent directory. It is not bundled here.

## Run

```
python run_xai_comparison.py
python normalized_pearson_check.py
```

## Headline result

- SHAP reproduces the EBM native explanation almost perfectly: Spearman 0.97, Pearson 0.99.
- LIME does not: Spearman 0.32.
- LIME vs SHAP agree only 0.35.

The two post-hoc methods are not interchangeable: SHAP is faithful to the model's
true reasoning here, while LIME is not.
