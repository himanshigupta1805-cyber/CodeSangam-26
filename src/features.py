"""Create calendar, history, and booking features for forecasting."""

import numpy as np
import pandas as pd


TARGET = "audience_count"
BOOKING_COLUMNS = ["tickets_booked", "tickets_sold"]


def make_features(history, rows, theaters, bookings_daily=None):
    """Build features for requested theater/date rows.

    Only target values from dates before each requested date are used.
    Booking features use the previous day's totals.
    """
    history = history.copy()
    rows = rows.copy()

    history["show_date"] = pd.to_datetime(history["show_date"]).dt.normalize()
    rows["show_date"] = pd.to_datetime(rows["show_date"]).dt.normalize()

    theaters = theaters.drop_duplicates("book_theater_id").copy()

    # Create a date-indexed audience history for each theater.
    history_by_theater = {
        theater_id: (
            group.drop_duplicates("show_date")
            .set_index("show_date")[TARGET]
            .sort_index()
        )
        for theater_id, group in history.groupby("book_theater_id", sort=False)
    }

    # Create a lookup for daily booking totals, if booking data was provided.
    booking_lookup = None
    if bookings_daily is not None and not bookings_daily.empty:
        bookings_daily = bookings_daily.copy()
        bookings_daily["show_date"] = (
            pd.to_datetime(bookings_daily["show_date"]).dt.normalize()
        )
        booking_lookup = bookings_daily.set_index(
            ["book_theater_id", "show_date"]
        )

    # Add theater area/type/coordinates to each requested theater/date row.
    rows = rows.merge(
        theaters,
        on="book_theater_id",
        how="left",
        validate="many_to_one",
    )

    feature_rows = []

    for row in rows.itertuples(index=False):
        theater_id = row.book_theater_id
        date = row.show_date

        # Important: use a date-indexed empty series when there is no history.
        # This allows the comparison with `date` below to work.
        by_date = history_by_theater.get(
            theater_id,
            pd.Series(index=pd.DatetimeIndex([]), dtype=float),
        )

        # Exclude the requested date and anything later to prevent leakage.
        by_date = by_date[by_date.index < date]

        item = {
            "book_theater_id": theater_id,
            "theater_area": getattr(row, "theater_area", None),
            "theater_type": getattr(row, "theater_type", None),
            "latitude": getattr(row, "latitude", np.nan),
            "longitude": getattr(row, "longitude", np.nan),
            "weekday": date.dayofweek,
            "month": date.month,
            "day_of_month": date.day,
            "day_of_year": date.dayofyear,
            "is_weekend": int(date.dayofweek >= 5),
        }

        # Audience counts from exactly 1, 7, 14, and 28 days earlier.
        for lag in (1, 7, 14, 28):
            lag_date = date - pd.Timedelta(days=lag)
            item[f"lag_{lag}"] = by_date.get(lag_date, np.nan)

        # Statistics from the preceding 7, 14, and 28 calendar days.
        for window in (7, 14, 28):
            first_date = date - pd.Timedelta(days=window)
            recent_values = by_date[by_date.index >= first_date]

            item[f"rolling_mean_{window}"] = (
                recent_values.mean() if not recent_values.empty else np.nan
            )
            item[f"rolling_std_{window}"] = (
                recent_values.std() if len(recent_values) > 1 else np.nan
            )

        # Use only booking totals from the prior day, not the requested date.
        for column in BOOKING_COLUMNS:
            prior_date = date - pd.Timedelta(days=1)
            key = (theater_id, prior_date)

            if booking_lookup is not None and column in booking_lookup.columns:
                item[f"prior_{column}"] = booking_lookup[column].get(key, 0)
            else:
                item[f"prior_{column}"] = 0

        feature_rows.append(item)

    return pd.DataFrame(feature_rows)


def make_training_frame(visits, theaters, bookings_daily=None):
    """Create model-training features for all observed visit rows."""
    visits = visits.copy()
    visits["show_date"] = pd.to_datetime(visits["show_date"]).dt.normalize()
    visits = visits.sort_values("show_date").reset_index(drop=True)

    # make_features checks each row's date and only uses earlier visit history.
    features = make_features(
        history=visits,
        rows=visits[["book_theater_id", "show_date"]],
        theaters=theaters,
        bookings_daily=bookings_daily,
    )

    features[TARGET] = visits[TARGET].to_numpy()
    return features