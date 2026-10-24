"""Train and save CatBoost, LightGBM, or their weighted ensemble."""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_data
from features import TARGET, make_training_frame
from reconcile import prepare_data

# Select "catboost", "lightgbm", or "ensemble". The default uses both.
MODEL_NAME = "ensemble"
IMPLEMENTED_MODELS = {"catboost", "lightgbm", "ensemble"}
CATEGORICAL_COLUMNS = [
    "book_theater_id",
    "theater_area",
    "theater_type",
    "theater_cluster_id",
]
ENSEMBLE_WEIGHTS = {"catboost": 0.60, "lightgbm": 0.40}


def get_model_path(model_name=None):
    """Return the saved-model path for the selected model."""
    selected = model_name or MODEL_NAME
    return PROJECT_ROOT / "models" / f"{selected}_footfall.pkl"


def split_features(frame, feature_columns=None, model_name=None):
    """Prepare consistent feature inputs for CatBoost or LightGBM."""
    selected = model_name or MODEL_NAME
    if selected == "ensemble":
        raise ValueError("Prepare ensemble inputs through predict_frame().")
    if selected not in IMPLEMENTED_MODELS:
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
            if column in x.columns:
                x[column] = x[column].fillna("__MISSING__").astype(str)

    numeric_columns = [
        column for column in x.columns if column not in CATEGORICAL_COLUMNS
    ]
    for column in numeric_columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    if numeric_columns:
        x[numeric_columns] = x[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(-1)

    if selected == "lightgbm":
        for column in categorical_columns:
            if column in x.columns:
                x[column] = x[column].astype("category")
    return x


def _fit_single_model(frame, selected):
    """Train one model and retain its feature metadata."""
    x = split_features(frame, model_name=selected)
    y = frame[TARGET]
    categorical_features = [
        column for column in CATEGORICAL_COLUMNS if column in x.columns
    ]

    if selected == "catboost":
        try:
            from models.catboost_model import build_model
        except ModuleNotFoundError as error:
            if error.name == "catboost":
                raise RuntimeError(
                    "CatBoost is missing from this environment. Install it with "
                    "python -m pip install catboost."
                ) from error
            raise
        model = build_model()
        model.fit(x, y, cat_features=categorical_features)
    else:
        try:
            from models.lightgbm_model import build_model
        except ModuleNotFoundError as error:
            if error.name == "lightgbm":
                raise RuntimeError(
                    "LightGBM is missing from this environment. Install it with "
                    "python -m pip install lightgbm."
                ) from error
            raise
        model = build_model()
        model.fit(x, y, categorical_feature=categorical_features)

    return {
        "model": model,
        "model_name": selected,
        "feature_columns": list(x.columns),
        "categorical_features": categorical_features,
    }


def fit_model(frame, model_name=None):
    """Fit the selected model, or fit both and save a 60/40 ensemble bundle."""
    selected = model_name or MODEL_NAME
    if selected not in IMPLEMENTED_MODELS:
        choices = ", ".join(sorted(IMPLEMENTED_MODELS))
        raise ValueError(f"Unknown model {selected!r}. Choose one of: {choices}")

    if selected == "ensemble":
        components = {
            name: _fit_single_model(frame, name)
            for name in ENSEMBLE_WEIGHTS
        }
        return {
            "model_name": "ensemble",
            "components": components,
            "weights": dict(ENSEMBLE_WEIGHTS),
        }
    return _fit_single_model(frame, selected)


def predict_frame(model_bundle, features):
    """Predict from a feature frame using a single model or weighted ensemble."""
    if model_bundle["model_name"] == "ensemble":
        predictions = []
        weights = model_bundle["weights"]
        for name, component in model_bundle["components"].items():
            x = split_features(
                features,
                feature_columns=component["feature_columns"],
                model_name=name,
            )
            predictions.append(
                weights[name] * np.asarray(component["model"].predict(x))
            )
        return np.sum(predictions, axis=0)

    x = split_features(
        features,
        feature_columns=model_bundle["feature_columns"],
        model_name=model_bundle["model_name"],
    )
    return np.asarray(model_bundle["model"].predict(x))


def main():
    data = prepare_data(load_data())
    frame = make_training_frame(
        data["visits"],
        data["booknow_theaters"],
        data["bookings_daily"],
        theater_relation=data["theater_relation"],
    )
    bundle = fit_model(frame)

    path = get_model_path(bundle["model_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)

    if bundle["model_name"] == "ensemble":
        print("Trained 60% CatBoost + 40% LightGBM ensemble")
    else:
        print(f"Trained model: {bundle['model_name']}")
    print(f"Saved model to: {path}")


if __name__ == "__main__":
    main()
