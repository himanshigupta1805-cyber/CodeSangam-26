"""Temporal holdout evaluation: recursively forecast the final 28 days."""

import numpy as np
import pandas as pd

from data_loader import load_data
from features import TARGET, make_features, make_training_frame
from reconcile import prepare_data
from train import MODEL_NAME, fit_model, predict_frame


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
    holdout_days=28,
    model_name=None,
    theater_relation=None,
):
    """Train before a cutoff, then recursively forecast the holdout period."""
    selected = model_name or MODEL_NAME
    visits = visits.sort_values("show_date").copy()
    cutoff = visits["show_date"].max() - pd.Timedelta(days=holdout_days - 1)
    history = visits[visits["show_date"] < cutoff].copy()
    actual = visits[visits["show_date"] >= cutoff].copy()

    model_bundle = fit_model(
        make_training_frame(
            history, theaters, bookings_daily, theater_relation=theater_relation
        ),
        model_name=selected,
    )

    predictions = []
    for date in sorted(actual["show_date"].unique()):
        rows = actual.loc[
            actual["show_date"] == date,
            ["book_theater_id", "show_date"],
        ]
        features = make_features(
            history, rows, theaters, bookings_daily, theater_relation=theater_relation
        )
        yhat = predict_frame(model_bundle, features)

        batch = rows.copy()
        batch[TARGET] = yhat
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
    data = prepare_data(load_data())
    print(
        temporal_validation(
            data["visits"],
            data["booknow_theaters"],
            data["bookings_daily"],
            model_name=MODEL_NAME,
            theater_relation=data["theater_relation"],
        )
    )
