"""Create recursive test forecasts using the selected saved model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from data_loader import load_data
from features import TARGET, make_features, make_training_frame
from reconcile import prepare_data
from train import (
    DEFAULT_RUN_MODE,
    MODEL_NAME,
    fit_model,
    fit_hybrid_models,
    get_hybrid_model_path,
    get_model_path,
    split_frame,
    split_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORIES = {
    "catboost": PROJECT_ROOT / "outputs" / "catboost",
    "lightgbm": PROJECT_ROOT / "outputs" / "lightgbm",
    "hybrid": PROJECT_ROOT / "outputs" / "hybrid",
}


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
            data.get("location_groups"),
            data.get("bookings_by_location"),
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
            data.get("location_groups"),
            data.get("bookings_by_location"),
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


def forecast_hybrid(data, model_bundles):
    """Use CatBoost for 60% and LightGBM for 40% of test rows."""
    catboost_rows, lightgbm_rows = split_frame(
        data["test"], first_fraction=0.6
    )
    model_ids = {
        "catboost": set(catboost_rows["ID"]),
        "lightgbm": set(lightgbm_rows["ID"]),
    }

    visits = data["visits"].copy()
    test = data["test"].copy()
    theaters = data["booknow_theaters"]
    output = []

    for date in sorted(test["show_date"].unique()):
        date_rows = test[test["show_date"] == date]
        daily_predictions = []

        for model_name in ("catboost", "lightgbm"):
            rows = date_rows[date_rows["ID"].isin(model_ids[model_name])]
            if rows.empty:
                continue

            features = make_features(
                visits,
                rows,
                theaters,
                data.get("bookings_daily"),
                data.get("location_groups"),
                data.get("bookings_by_location"),
            )
            bundle = model_bundles[model_name]
            x = split_features(
                features,
                bundle["feature_columns"],
                model_name=model_name,
            )
            result = rows[["ID", "book_theater_id", "show_date"]].copy()
            result[TARGET] = np.maximum(bundle["model"].predict(x), 0)
            daily_predictions.append(result)

        day_result = pd.concat(daily_predictions, ignore_index=True)
        output.append(day_result)
        # Both models' predictions enter history before forecasting the next day.
        visits = pd.concat(
            [
                visits,
                day_result[["book_theater_id", "show_date", TARGET]],
            ],
            ignore_index=True,
        )

    return pd.concat(output, ignore_index=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        choices=["catboost", "lightgbm", "hybrid"],
        default=DEFAULT_RUN_MODE,
    )
    selected = parser.parse_args().model
    data = prepare_data(load_data())

    if selected == "hybrid":
        model_paths = {
            model_name: get_hybrid_model_path(model_name)
            for model_name in ("catboost", "lightgbm")
        }
        if all(path.is_file() for path in model_paths.values()):
            bundles = {}
            for model_name, path in model_paths.items():
                with path.open("rb") as file:
                    bundles[model_name] = pickle.load(file)
                if bundles[model_name].get("model_name") != model_name:
                    raise ValueError(
                        f"Saved model at {path} is not the selected {model_name} model."
                    )
        else:
            print("Training CatBoost and LightGBM on the 60/40 training split.")
            training_frame = make_training_frame(
                data["visits"],
                data["booknow_theaters"],
                data.get("bookings_daily"),
                data.get("location_groups"),
                data.get("bookings_by_location"),
            )
            bundles = fit_hybrid_models(training_frame)
            for model_name, bundle in bundles.items():
                path = model_paths[model_name]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as file:
                    pickle.dump(bundle, file)
        predictions = forecast_hybrid(data, bundles)
    else:
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

    output_path = OUTPUT_DIRECTORIES[selected] / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
