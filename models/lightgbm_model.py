"""LightGBM regression model for cinema footfall forecasting."""

from lightgbm import LGBMRegressor


def build_model():
    """Return a reproducible LightGBM regressor (CPU-compatible by default)."""
    return LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
