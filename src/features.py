"""Create calendar, history, and booking features for forecasting."""

import numpy as np
import pandas as pd


TARGET = "audience_count"
BOOKING_COLUMNS = ["tickets_booked", "tickets_sold"]


def make_features(
    history,
    rows,
    theaters,
    bookings_daily=None,
    location_groups=None,
    bookings_by_location=None,
):
    """Build features for requested theater/date rows.

    Only target values from dates before each requested date are used.
    Booking features use the previous day's totals.
    """
    history = history.copy()
    rows = rows.copy()
    history["show_date"] = pd.to_datetime(history["show_date"]).dt.normalize()
    rows["show_date"] = pd.to_datetime(rows["show_date"]).dt.normalize()
    theaters = theaters.drop_duplicates("book_theater_id").copy()

    # Use a shared location key where available; otherwise keep each BookNow ID
    # isolated, matching the prior behavior for IDs without coordinates.
    group_key = "__history_group"
    location_lookup = None
    if location_groups is not None:
        location_lookup = location_groups.loc[
            location_groups["source"] == "booknow",
            ["theater_id", "location_group_id"],
        ].drop_duplicates("theater_id")
        location_lookup = location_lookup.rename(
            columns={"theater_id": "book_theater_id"}
        )
        history = history.merge(
            location_lookup,
            on="book_theater_id",
            how="left",
            validate="many_to_one",
        )
        history["location_group_id"] = history["location_group_id"].fillna(
            "unlocated_booknow_" + history["book_theater_id"].astype(str)
        )
        history[group_key] = history["location_group_id"]
    else:
        history[group_key] = history["book_theater_id"]

    rows = rows.merge(
        theaters,
        on="book_theater_id",
        how="left",
        validate="many_to_one",
    )
    if location_lookup is not None:
        rows = rows.merge(
            location_lookup,
            on="book_theater_id",
            how="left",
            validate="many_to_one",
        )
        rows["location_group_id"] = rows["location_group_id"].fillna(
            "unlocated_booknow_" + rows["book_theater_id"].astype(str)
        )
        rows[group_key] = rows["location_group_id"]
    else:
        rows["location_group_id"] = rows["book_theater_id"]
        rows[group_key] = rows["book_theater_id"]

    # Calculate temporal features once per location/date, even when several
    # BookNow IDs in the same group request features for the same date.
    history_daily = (
        history.groupby([group_key, "show_date"], as_index=False)[TARGET]
        .mean()
        .sort_values([group_key, "show_date"])
    )
    request_keys = rows[[group_key, "show_date"]].drop_duplicates().reset_index(
        drop=True
    )
    temporal_features = _build_temporal_features(
        history_daily, request_keys, group_key
    )
    rows = rows.merge(
        temporal_features,
        on=[group_key, "show_date"],
        how="left",
        validate="many_to_one",
    )

    date = rows["show_date"]
    rows["weekday"] = date.dt.dayofweek
    rows["month"] = date.dt.month
    rows["day_of_month"] = date.dt.day
    rows["day_of_year"] = date.dt.dayofyear
    rows["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)

    # Join all prior-day booking totals in one batch instead of looking up each
    # row individually in a large pandas MultiIndex.
    booking_table = bookings_daily
    booking_key = "book_theater_id"
    if bookings_by_location is not None and not bookings_by_location.empty:
        booking_table = bookings_by_location
        booking_key = "location_group_id"

    if booking_table is not None and not booking_table.empty:
        booking_table = booking_table[[booking_key, "show_date", *BOOKING_COLUMNS]].copy()
        booking_table["show_date"] = pd.to_datetime(
            booking_table["show_date"]
        ).dt.normalize()
        prior_rows = rows[[booking_key, "show_date"]].copy()
        prior_rows["show_date"] -= pd.Timedelta(days=1)
        prior_rows = prior_rows.merge(
            booking_table,
            on=[booking_key, "show_date"],
            how="left",
            validate="many_to_one",
        )
        for column in BOOKING_COLUMNS:
            rows[f"prior_{column}"] = prior_rows[column].fillna(0).to_numpy()
    else:
        for column in BOOKING_COLUMNS:
            rows[f"prior_{column}"] = 0

    columns = [
        "book_theater_id",
        "location_group_id",
        "theater_area",
        "theater_type",
        "latitude",
        "longitude",
        "weekday",
        "month",
        "day_of_month",
        "day_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        "rolling_mean_28",
        "rolling_std_28",
        "prior_tickets_booked",
        "prior_tickets_sold",
    ]
    return rows[columns]


def _build_temporal_features(history_daily, request_keys, group_key):
    """Build lag and calendar-window features using vectorized date searches."""
    lag_days = (1, 7, 14, 28)
    windows = (7, 14, 28)
    result = request_keys.copy()
    row_count = len(result)
    values = {
        f"lag_{days}": np.full(row_count, np.nan)
        for days in lag_days
    }
    for window in windows:
        values[f"rolling_mean_{window}"] = np.full(row_count, np.nan)
        values[f"rolling_std_{window}"] = np.full(row_count, np.nan)

    history_by_group = {
        group_id: group
        for group_id, group in history_daily.groupby(group_key, sort=False)
    }
    for group_id, positions in result.groupby(group_key, sort=False).indices.items():
        group_history = history_by_group.get(group_id)
        if group_history is None or group_history.empty:
            continue

        history_dates = group_history["show_date"].to_numpy(
            dtype="datetime64[ns]"
        )
        audience = group_history[TARGET].to_numpy(dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(audience)))
        cumulative_squared = np.concatenate(([0.0], np.cumsum(audience**2)))
        target_dates = result.iloc[positions]["show_date"].to_numpy(
            dtype="datetime64[ns]"
        )

        for days in lag_days:
            wanted_dates = target_dates - np.timedelta64(days, "D")
            history_positions = np.searchsorted(history_dates, wanted_dates)
            valid = history_positions < len(history_dates)
            matched_positions = np.flatnonzero(valid)
            if len(matched_positions):
                valid[matched_positions] &= (
                    history_dates[history_positions[matched_positions]]
                    == wanted_dates[matched_positions]
                )
            values[f"lag_{days}"][positions[valid]] = audience[
                history_positions[valid]
            ]

        # Use observations in [date-window, date), excluding the target date.
        end_positions = np.searchsorted(history_dates, target_dates, side="left")
        for window in windows:
            start_dates = target_dates - np.timedelta64(window, "D")
            start_positions = np.searchsorted(
                history_dates, start_dates, side="left"
            )
            counts = end_positions - start_positions
            sums = cumulative[end_positions] - cumulative[start_positions]
            sums_squared = (
                cumulative_squared[end_positions]
                - cumulative_squared[start_positions]
            )
            valid_mean = counts > 0
            values[f"rolling_mean_{window}"][positions[valid_mean]] = (
                sums[valid_mean] / counts[valid_mean]
            )
            valid_std = counts > 1
            variance = np.zeros(len(counts), dtype=float)
            variance[valid_std] = (
                sums_squared[valid_std]
                - sums[valid_std] ** 2 / counts[valid_std]
            ) / (counts[valid_std] - 1)
            values[f"rolling_std_{window}"][positions[valid_std]] = np.sqrt(
                np.maximum(variance[valid_std], 0)
            )

    for name, column_values in values.items():
        result[name] = column_values
    return result


def make_training_frame(
    visits,
    theaters,
    bookings_daily=None,
    location_groups=None,
    bookings_by_location=None,
):
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
        location_groups=location_groups,
        bookings_by_location=bookings_by_location,
    )

    features[TARGET] = visits[TARGET].to_numpy()
    return features
