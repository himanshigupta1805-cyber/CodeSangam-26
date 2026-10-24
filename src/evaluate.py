"""Temporal holdout evaluation: recursively forecast the final 28 days."""

import numpy as np
import pandas as pd

from data_loader import load_data
from features import TARGET, make_features, make_training_frame
from reconcile import prepare_data
from train import (
    MODEL_NAME,
    fit_hybrid_models,
    fit_model,
    split_features,
    split_frame,
)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return {
        "WAPE": float(
            np.abs(actual - predicted).sum()
            / max(np.abs(actual).sum(), 1e-9)
        ),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAE": float(np.mean(np.abs(actual - predicted))),
    }


def temporal_validation(
    visits,
    theaters,
    bookings_daily=None,
    location_groups=None,
    bookings_by_location=None,
    holdout_days=28,
    model_name=None,
):
    """Train before a cutoff, then recursively forecast the holdout period."""
    selected = model_name or MODEL_NAME
    visits = visits.sort_values("show_date").copy()
    cutoff = visits["show_date"].max() - pd.Timedelta(days=holdout_days - 1)
    history = visits[visits["show_date"] < cutoff].copy()
    actual = visits[visits["show_date"] >= cutoff].copy()

    training_frame = make_training_frame(
        history,
        theaters,
        bookings_daily,
        location_groups,
        bookings_by_location,
    )
    if selected == "hybrid":
        model_bundles = fit_hybrid_models(training_frame)
        catboost_actual, lightgbm_actual = split_frame(
            actual, first_fraction=0.6
        )
        model_indices = {
            "catboost": set(catboost_actual.index),
            "lightgbm": set(lightgbm_actual.index),
        }

        predictions = []
        for date in sorted(actual["show_date"].unique()):
            date_rows = actual.loc[
                actual["show_date"] == date,
                ["book_theater_id", "show_date"],
            ]
            daily_predictions = []
            for model_name in ("catboost", "lightgbm"):
                rows = date_rows.loc[
                    date_rows.index.isin(model_indices[model_name])
                ]
                if rows.empty:
                    continue
                features = make_features(
                    history,
                    rows,
                    theaters,
                    bookings_daily,
                    location_groups,
                    bookings_by_location,
                )
                bundle = model_bundles[model_name]
                x = split_features(
                    features,
                    bundle["feature_columns"],
                    model_name=model_name,
                )
                batch = rows.copy()
                batch[TARGET] = np.maximum(bundle["model"].predict(x), 0)
                daily_predictions.append(batch)

            day_predictions = pd.concat(daily_predictions, ignore_index=True)
            predictions.append(day_predictions)
            history = pd.concat([history, day_predictions], ignore_index=True)
    else:
        model_bundle = fit_model(training_frame, model_name=selected)
        predictions = []
        for date in sorted(actual["show_date"].unique()):
            rows = actual.loc[
                actual["show_date"] == date,
                ["book_theater_id", "show_date"],
            ]
            features = make_features(
                history,
                rows,
                theaters,
                bookings_daily,
                location_groups,
                bookings_by_location,
            )
            x = split_features(
                features,
                model_bundle["feature_columns"],
                model_name=model_bundle["model_name"],
            )
            batch = rows.copy()
            batch[TARGET] = np.maximum(
                model_bundle["model"].predict(x), 0
            )
            predictions.append(batch)
            history = pd.concat([history, batch], ignore_index=True)

    predicted = pd.concat(predictions, ignore_index=True)
    scored = actual.merge(
        predicted,
        on=["book_theater_id", "show_date"],
        suffixes=("_actual", "_pred"),
    )
    return regression_metrics(
        scored[f"{TARGET}_actual"],
        scored[f"{TARGET}_pred"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["catboost", "lightgbm", "hybrid", "all"],
        default="all",
        help="Model to evaluate; 'all' compares all three on the same holdout.",
    )
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()

    data = prepare_data(load_data())
    model_names = (
        ["catboost", "lightgbm", "hybrid"]
        if args.model == "all"
        else [args.model]
    )
    results = {}
    for model_name in model_names:
        results[model_name] = temporal_validation(
            data["visits"],
            data["booknow_theaters"],
            data["bookings_daily"],
            data["location_groups"],
            data["bookings_by_location"],
            holdout_days=args.holdout_days,
            model_name=model_name,
        )
    print(
        pd.DataFrame.from_dict(results, orient="index").to_string(
            float_format="{:.4f}".format
        )
    )
