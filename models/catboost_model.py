"""CatBoost regression model configured for CPU training."""

from catboost import CatBoostRegressor


def build_model():
    """Return a CatBoost regressor configured for CPU training."""
    return CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="CPU",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
