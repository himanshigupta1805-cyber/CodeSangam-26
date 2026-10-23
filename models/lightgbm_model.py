"""LightGBM regression model for audience forecasting."""

from lightgbm import LGBMRegressor


def build_model():
    """Return a deterministic CPU LightGBM regressor."""
    return LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        device_type="cpu",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
