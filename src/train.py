"""Train and save the baseline Random Forest model."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_loader import load_data
from features import TARGET, make_training_frame
from reconcile import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "footfall_model.pkl"


def split_features(frame, feature_columns=None):
    """One-hot encode categories and align columns for later prediction."""
    x = frame.drop(columns=[TARGET], errors="ignore").copy()
    x = pd.get_dummies(x, columns=["book_theater_id", "theater_area", "theater_type"], dummy_na=True)
    if feature_columns is not None:
        x = x.reindex(columns=feature_columns, fill_value=0)
    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)


def fit_model(frame):
    x = split_features(frame)
    y = frame[TARGET]
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3,
                                  max_features=0.8, random_state=42, n_jobs=-1)
    model.fit(x, y)
    return {"model": model, "feature_columns": list(x.columns)}


def main():
    data = prepare_data(load_data())
    frame = make_training_frame(data["visits"], data["booknow_theaters"], data["bookings_daily"])
    bundle = fit_model(frame)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
