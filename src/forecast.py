"""Create recursive test forecasts using the selected saved model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from data_loader import load_data
from features import TARGET, make_features, make_training_frame
from reconcile import prepare_data
from train import (
    MODEL_NAME,
    fit_model,
    get_model_path,
    split_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "submission.csv"


def forecast(data, model_bundle=None, model_name=None):
    """Forecast every test row, recursively feeding predictions into history."""
    selected = model_name or MODEL_NAME
    visits = data["visits"].copy()
    test = data["test"].copy()
    theaters = data["booknow_theaters"]

    if model_bundle is None:
        training_frame = make_training_frame(
            visits,
            theaters,
            data.get("bookings_daily"),
        )
        model_bundle = fit_model(training_frame, model_name=selected)

    output = []
    for date in sorted(test["show_date"].unique()):
        rows = test[test["show_date"] == date]
        features = make_features(
            visits,
            rows,
            theaters,
            data.get("bookings_daily"),
        )
        x = split_features(
            features,
            model_bundle["feature_columns"],
            model_name=model_bundle["model_name"],
        )

        result = rows[["ID", "book_theater_id", "show_date"]].copy()
        result[TARGET] = np.maximum(model_bundle["model"].predict(x), 0)
        output.append(result)

        # Feed this day's predicted audience into history for later dates.
        history_rows = result[
            ["book_theater_id", "show_date", TARGET]
        ]
        visits = pd.concat([visits, history_rows], ignore_index=True)

    return pd.concat(output, ignore_index=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", choices=["catboost", "lightgbm"], default=MODEL_NAME)
    selected = parser.parse_args().model
    data = prepare_data(load_data())
    model_path = get_model_path(selected)

    if model_path.is_file():
        with model_path.open("rb") as file:
            bundle = pickle.load(file)
        if bundle.get("model_name") != selected:
            raise ValueError(
                f"Saved model at {model_path} is not the selected {selected} model."
            )
    else:
        print(f"No saved {selected} model found; training it now.")
        bundle = None

    predictions = forecast(data, bundle, model_name=selected)
    submission = data["sample_submission"][["ID"]].merge(
        predictions[["ID", TARGET]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    if submission[TARGET].isna().any():
        raise ValueError("Some test IDs did not receive a forecast.")

    output_path = PROJECT_ROOT / "outputs" / (
        "submission.csv" if selected == "catboost" else f"submission_{selected}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
