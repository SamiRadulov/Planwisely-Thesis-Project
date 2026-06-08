from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMRegressor


def build_pipeline():
    return Pipeline([
        ("var", VarianceThreshold(threshold=0.0)),
        ("reg", LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ))
    ])
