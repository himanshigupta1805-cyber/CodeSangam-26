"""Leakage-conscious daily features for one-step and recursive forecasts."""
import numpy as np
import pandas as pd


TARGET = "audience_count"
BOOKING_COLUMNS = ["tickets_booked", "tickets_sold"]


def make_features(history, rows, theaters, bookings_daily=None):
    """Build features for rows using only target history dated before each row.

    history must contain book_theater_id, show_date, audience_count. `rows`
    contains book_theater_id and show_date. Missing exact-day lags stay NaN.
    """
    history = history.copy()
    rows = rows.copy()
    history["show_date"] = pd.to_datetime(history["show_date"]).dt.normalize()
    rows["show_date"] = pd.to_datetime(rows["show_date"]).dt.normalize()
    theaters = theaters.drop_duplicates("book_theater_id").copy()
    # Build per-theater lookup series once, instead of scanning the full history
    # separately for every row.
    history_by_theater = {
        theater_id: group.drop_duplicates("show_date").set_index("show_date")[TARGET].sort_index()
        for theater_id, group in history.groupby("book_theater_id", sort=False)
    }
    if bookings_daily is not None and not bookings_daily.empty:
        booking_frame = bookings_daily.copy()
        booking_frame["show_date"] = pd.to_datetime(booking_frame["show_date"]).dt.normalize()
        booking_lookup = booking_frame.set_index(["book_theater_id", "show_date"])
    else:
        booking_lookup = None

    rows = rows.merge(theaters, on="book_theater_id", how="left", validate="many_to_one")
    output = []
    for row in rows.itertuples(index=False):
        theater_id, date = row.book_theater_id, row.show_date
        by_date = history_by_theater.get(theater_id, pd.Series(dtype=float))
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
        for lag in (1, 7, 14, 28):
            item[f"lag_{lag}"] = by_date.get(date - pd.Timedelta(days=lag), np.nan)
        for window in (7, 14, 28):
            recent = by_date[by_date.index >= date - pd.Timedelta(days=window)]
            item[f"rolling_mean_{window}"] = recent.mean() if len(recent) else np.nan
            item[f"rolling_std_{window}"] = recent.std() if len(recent) > 1 else np.nan
        # Prior day's ledger totals are available without using same-day bookings.
        for column in BOOKING_COLUMNS:
            key = (theater_id, date - pd.Timedelta(days=1))
            item[f"prior_{column}"] = (booking_lookup[column].get(key, 0)
                                        if booking_lookup is not None and column in booking_lookup
                                        else 0)
        output.append(item)
    return pd.DataFrame(output)


def make_training_frame(visits, theaters, bookings_daily=None):
    """Create one training row per observed label, using only earlier labels."""
    visits = visits.sort_values("show_date").reset_index(drop=True)
    x = make_features(visits, visits[["book_theater_id", "show_date"]], theaters, bookings_daily)
    x[TARGET] = visits[TARGET].to_numpy()
    return x
