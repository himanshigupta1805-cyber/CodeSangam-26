"""Train and save the selected footfall regression model."""

from pathlib import Path
import pickle
import sys

import numpy as np
from features import TARGET, make_training_frame
from reconcile import prepare_data
from data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Select the model here. LightGBM can be selected after its model implementation
# is added; for now CatBoost is the only implemented model.
MODEL_NAME = "catboost"
IMPLEMENTED_MODELS = {"catboost"}
PLANNED_MODELS = {"lightgbm"}
CATEGORICAL_COLUMNS = ["book_theater_id", "theater_area", "theater_type"]


def get_model_path(model_name=None):
    """Return the saved-model path for the selected model."""
    selected = model_name or MODEL_NAME
    return PROJECT_ROOT / "models" / f"{selected}_footfall.pkl"


def split_features(frame, feature_columns=None, model_name=None):
    """Prepare model inputs, using native categorical features for CatBoost."""
    selected = model_name or MODEL_NAME
    if selected not in IMPLEMENTED_MODELS:
        if selected in PLANNED_MODELS:
            raise NotImplementedError(
                f"{selected} has not been implemented yet. Use 'catboost' for now."
            )
        raise ValueError(f"Unknown model name: {selected!r}")

    x = frame.drop(columns=[TARGET], errors="ignore").copy()

    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in x.columns
    ]
    for column in categorical_columns:
        x[column] = x[column].fillna("__MISSING__").astype(str)

    if feature_columns is not None:
        x = x.reindex(columns=feature_columns, fill_value=0)
        for column in categorical_columns:
            x[column] = x[column].fillna("__MISSING__").astype(str)

    x = x.replace([np.inf, -np.inf], np.nan)
    numeric_columns = [
        column for column in x.columns if column not in CATEGORICAL_COLUMNS
    ]
    x[numeric_columns] = x[numeric_columns].fillna(-1)

    return x


def fit_model(frame, model_name=None):
    """Fit the chosen model and return it with its feature metadata."""
    selected = model_name or MODEL_NAME
    if selected not in IMPLEMENTED_MODELS:
        if selected in PLANNED_MODELS:
            raise NotImplementedError(
                f"{selected} has not been implemented yet. Use 'catboost' for now."
            )
        choices = ", ".join(sorted(IMPLEMENTED_MODELS))
        raise ValueError(f"Unknown model {selected!r}. Choose one of: {choices}")

    x = split_features(frame, model_name=selected)
    y = frame[TARGET]

    try:
        from models.catboost_model import build_model
    except ModuleNotFoundError as error:
        if error.name == "catboost":
            raise RuntimeError(
                "CatBoost is not installed in this Python environment. "
                "Activate the Python 3.14 project environment and install it "
                "with: python -m pip install catboost"
            ) from error
        raise

    model = build_model()
    categorical_features = [
        column for column in CATEGORICAL_COLUMNS if column in x.columns
    ]
    model.fit(x, y, cat_features=categorical_features)

    return {
        "model": model,
        "model_name": selected,
        "feature_columns": list(x.columns),
        "categorical_features": categorical_features,
    }


def main():
    data = prepare_data(load_data())
    frame = make_training_frame(
        data["visits"],
        data["booknow_theaters"],
        data["bookings_daily"],
    )
    bundle = fit_model(frame)

    path = get_model_path(bundle["model_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)

    print(f"Trained model: {bundle['model_name']}")
    print(f"Saved model to: {path}")


if __name__ == "__main__":
    main()

