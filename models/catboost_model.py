"""CatBoost regression model configured to train on the NVIDIA GPU."""

from catboost import CatBoostRegressor


def build_model():
    """Return a CatBoost regressor configured for GPU training."""
    return CatBoostRegressor(
        iterations=1000,
        depth=10,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
