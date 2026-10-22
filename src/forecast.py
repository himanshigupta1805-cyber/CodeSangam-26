"""Create recursive forecasts for the rows and dates in test.csv."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from data_loader import load_data
from features import make_features
from reconcile import prepare_data
from train import MODEL_PATH, split_features, fit_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "submission.csv"


def forecast(data, model_bundle=None):
    visits = data["visits"].copy()
    test = data["test"].copy()
    theaters = data["booknow_theaters"]
    if model_bundle is None:
        # Train from all available history if no saved artifact exists.
        from features import make_training_frame
        model_bundle = fit_model(make_training_frame(visits, theaters, data.get("bookings_daily")))

    output = []
    for date in sorted(test.show_date.unique()):
        rows = test[test.show_date == date]
        features = make_features(visits, rows, theaters, data.get("bookings_daily"))
        x = split_features(features, model_bundle["feature_columns"])
        result = rows[["ID", "book_theater_id", "show_date"]].copy()
        result["audience_count"] = np.maximum(model_bundle["model"].predict(x), 0)
        output.append(result)
        # Append today's predictions so later test dates use recursively forecast lags.
        visits = pd.concat([visits, result.rename(columns={"audience_count": "audience_count"})[
            ["book_theater_id", "show_date", "audience_count"]]], ignore_index=True)
    return pd.concat(output, ignore_index=True)


def main():
    data = prepare_data(load_data())
    if MODEL_PATH.is_file():
        with MODEL_PATH.open("rb") as file:
            bundle = pickle.load(file)
    else:
        bundle = None
    predictions = forecast(data, bundle)
    submission = data["sample_submission"][["ID"]].merge(
        predictions[["ID", "audience_count"]], on="ID", how="left", validate="one_to_one"
    )
    if submission["audience_count"].isna().any():
        raise ValueError("Some test IDs did not receive a forecast.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(submission):,} forecasts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
