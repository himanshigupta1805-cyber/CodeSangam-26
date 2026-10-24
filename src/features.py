"""Create calendar, history, cross-ledger, and theater-cluster features."""

import numpy as np
import pandas as pd


TARGET = "audience_count"
BOOKING_COLUMNS = ["tickets_booked", "tickets_sold"]
ADVANCE_COLUMNS = [
    "booknow_lead_24_72h",
    "booknow_lead_72_168h",
    "booknow_lead_168h_plus",
]
LEAD_BANDS = ("0_24h", "24_72h", "72_168h", "168h_plus")

# Broad India-wide calendar for the supplied dates. Theater areas are
# anonymized, so state-specific holiday calendars cannot be inferred.
PUBLIC_HOLIDAYS = set(pd.to_datetime([
    "2023-01-26", "2023-03-08", "2023-04-04", "2023-04-07",
    "2023-04-22", "2023-05-05", "2023-06-29", "2023-07-29",
    "2023-08-15", "2023-09-07", "2023-09-28", "2023-10-02",
    "2023-10-24", "2023-11-12", "2023-11-27", "2023-12-25",
    "2024-01-26",
]))
FESTIVAL_WINDOWS = {
    "sankranti_pongal": [("2023-01-13", "2023-01-17"), ("2024-01-13", "2024-01-17")],
    "maha_shivaratri": [("2023-02-18", "2023-02-18")],
    "holi": [("2023-03-07", "2023-03-09")],
    "gudi_padwa_ugadi": [("2023-03-21", "2023-03-23")],
    "ramadan_eid": [("2023-03-23", "2023-04-22")],
    "eid_al_adha": [("2023-06-28", "2023-06-30")],
    "muharram": [("2023-07-28", "2023-07-30")],
    "janmashtami": [("2023-09-06", "2023-09-08")],
    "ganesh_chaturthi": [("2023-09-19", "2023-09-28")],
    "navratri_dussehra": [("2023-10-15", "2023-10-24")],
    "diwali": [("2023-11-10", "2023-11-15")],
    "guru_nanak_jayanti": [("2023-11-26", "2023-11-28")],
    "christmas": [("2023-12-24", "2023-12-26")],
}
FESTIVAL_WINDOWS = {
    name: [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows]
    for name, windows in FESTIVAL_WINDOWS.items()
}


def _theater_cluster_map(theaters, n_clusters=10):
    """Build repeatable spatial cluster IDs from theater latitude/longitude."""
    metadata = theaters.drop_duplicates("book_theater_id").copy()
    coordinates = metadata[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float, copy=True)

    if not np.isfinite(coordinates).any():
        return dict.fromkeys(metadata["book_theater_id"], "cluster_00")

    medians = np.nanmedian(coordinates, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(coordinates)
    coordinates[missing] = np.broadcast_to(medians, coordinates.shape)[missing]

    scale = coordinates.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (coordinates - coordinates.mean(axis=0)) / scale
    unique = np.unique(scaled, axis=0)
    k = min(n_clusters, len(unique))
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    initial = np.linspace(0, len(ordered) - 1, k).round().astype(int)
    centers = ordered[initial].copy()

    for _ in range(50):
        distances = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = scaled[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated

    labels = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    center_order = np.lexsort((centers[:, 1], centers[:, 0]))
    canonical_label = {old: new for new, old in enumerate(center_order)}
    return {
        theater_id: f"cluster_{canonical_label[label]:02d}"
        for theater_id, label in zip(metadata["book_theater_id"], labels)
    }


def _date_indexed_groups(frame, value_columns, group_column="book_theater_id"):
    if frame is None or frame.empty:
        return {}
    return {
        key: group.set_index("show_date")[value_columns].sort_index()
        for key, group in frame.groupby(group_column, sort=False)
    }


def make_features(
    history,
    rows,
    theaters,
    bookings_daily=None,
    theater_relation=None,
):
    """Build leak-safe features for requested theater/date rows.

    For future test dates beyond the booking-data range, booking-derived
    features are frozen at the latest available booking date.
    """
    history = history.copy()
    rows = rows.copy()
    history["show_date"] = pd.to_datetime(history["show_date"]).dt.normalize()
    rows["show_date"] = pd.to_datetime(rows["show_date"]).dt.normalize()

    theaters = theaters.drop_duplicates("book_theater_id").copy()
    cluster_map = _theater_cluster_map(theaters)
    theaters["theater_cluster_id"] = theaters["book_theater_id"].map(cluster_map)

    history_by_theater = {
        theater_id: group.drop_duplicates("show_date").set_index("show_date")[TARGET].sort_index()
        for theater_id, group in history.groupby("book_theater_id", sort=False)
    }

    cluster_history = {}
    if not history.empty:
        history_clusters = history.merge(
            theaters[["book_theater_id", "theater_cluster_id"]],
            on="book_theater_id",
            how="left",
            validate="many_to_one",
        )
        cluster_daily = (
            history_clusters.groupby(["theater_cluster_id", "show_date"], as_index=False)[TARGET]
            .mean()
        )
        cluster_history = _date_indexed_groups(
            cluster_daily,
            [TARGET],
            group_column="theater_cluster_id",
        )

    booking_by_theater = {}
    booking_lookup = None
    latest_booking_date = None
    if bookings_daily is not None and not bookings_daily.empty:
        bookings_daily = bookings_daily.copy()
        bookings_daily["show_date"] = pd.to_datetime(
            bookings_daily["show_date"]
        ).dt.normalize()
        booking_lookup = bookings_daily.set_index(["book_theater_id", "show_date"])
        latest_booking_date = bookings_daily["show_date"].max()
        booking_by_theater = _date_indexed_groups(
            bookings_daily,
            list(bookings_daily.columns.difference(["book_theater_id", "show_date"])),
        )

    if theater_relation is not None:
        matched_cinepos_ids = set(theater_relation["book_theater_id"].dropna())
    elif bookings_daily is not None and "tickets_sold" in bookings_daily:
        matched_cinepos_ids = set(
            bookings_daily.loc[bookings_daily["tickets_sold"] > 0, "book_theater_id"]
        )
    else:
        matched_cinepos_ids = set()

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
        by_date = history_by_theater.get(
            theater_id,
            pd.Series(index=pd.DatetimeIndex([]), dtype=float),
        )
        by_date = by_date[by_date.index < date]

        year_length = 366 if date.is_leap_year else 365
        annual_angle = 2 * np.pi * (date.dayofyear - 1) / year_length
        item = {
            "book_theater_id": theater_id,
            "theater_area": getattr(row, "theater_area", None),
            "theater_type": getattr(row, "theater_type", None),
            "latitude": getattr(row, "latitude", np.nan),
            "longitude": getattr(row, "longitude", np.nan),
            "theater_cluster_id": getattr(row, "theater_cluster_id", "cluster_00"),
            "weekday": date.dayofweek,
            "month": date.month,
            "day_of_month": date.day,
            "day_of_year": date.dayofyear,
            "is_weekend": int(date.dayofweek >= 5),
            "is_public_holiday": int(date in PUBLIC_HOLIDAYS),
            "day_of_year_sin": np.sin(annual_angle),
            "day_of_year_cos": np.cos(annual_angle),
        }
        festival_flags = {
            f"festival_{name}": int(
                any(pd.Timestamp(start) <= date <= pd.Timestamp(end) for start, end in windows)
            )
            for name, windows in FESTIVAL_WINDOWS.items()
        }
        item.update(festival_flags)
        item["festival_season"] = int(any(festival_flags.values()))

        for lag in (1, 7, 14, 28):
            item[f"lag_{lag}"] = by_date.get(date - pd.Timedelta(days=lag), np.nan)

        for window in (7, 14, 28):
            recent = by_date[by_date.index >= date - pd.Timedelta(days=window)]
            item[f"rolling_mean_{window}"] = recent.mean() if not recent.empty else np.nan
            item[f"rolling_std_{window}"] = recent.std() if len(recent) > 1 else np.nan

        cluster_id = item["theater_cluster_id"]
        cluster_series = cluster_history.get(
            cluster_id,
            pd.DataFrame(columns=[TARGET], index=pd.DatetimeIndex([]))[TARGET],
        )
        cluster_series = cluster_series[cluster_series.index < date]
        for window in (7, 28):
            cluster_recent = cluster_series[
                cluster_series.index >= date - pd.Timedelta(days=window)
            ]
            item[f"cluster_attendance_mean_{window}d"] = (
                cluster_recent.mean() if not cluster_recent.empty else np.nan
            )

        # Stop booking-derived features at the latest date actually present in
        # the booking data, rather than letting them decay to zero in the test horizon.
        booking_as_of = date
        if latest_booking_date is not None:
            booking_as_of = min(date, latest_booking_date + pd.Timedelta(days=1))
        booking_history = booking_by_theater.get(theater_id)
        if booking_history is None:
            booking_history = pd.DataFrame(index=pd.DatetimeIndex([]))
        else:
            booking_history = booking_history[booking_history.index < booking_as_of]

        previous_date = date - pd.Timedelta(days=1)
        if latest_booking_date is not None:
            previous_date = min(previous_date, latest_booking_date)
        for column in BOOKING_COLUMNS:
            key = (theater_id, previous_date)
            item[f"prior_{column}"] = (
                booking_lookup[column].get(key, 0)
                if booking_lookup is not None and column in booking_lookup.columns
                else 0
            )

        matched = theater_id in matched_cinepos_ids
        item["has_cinepos_match"] = int(matched)
        window_frames = {}
        for window in (7, 28):
            start = booking_as_of - pd.Timedelta(days=window)
            frame = booking_history[booking_history.index >= start]
            window_frames[window] = frame
            online_advance = sum(
                frame[column].sum()
                for column in ADVANCE_COLUMNS
                if column in frame.columns
            )
            counter_sales = frame["tickets_sold"].sum() if "tickets_sold" in frame else 0
            item[f"advance_to_counter_ratio_{window}d"] = (
                (online_advance + 1) / (counter_sales + 1) if matched else np.nan
            )
            item[f"online_advance_share_{window}d"] = (
                online_advance / (online_advance + counter_sales + 1)
                if matched else np.nan
            )

        recent_week = window_frames[7]
        prior_week_start = booking_as_of - pd.Timedelta(days=14)
        prior_week_end = booking_as_of - pd.Timedelta(days=7)
        prior_week = booking_history[
            (booking_history.index >= prior_week_start)
            & (booking_history.index < prior_week_end)
        ]
        for column, name in (("tickets_booked", "booknow"), ("tickets_sold", "cinepos")):
            recent_total = recent_week[column].sum() if column in recent_week else 0
            previous_total = prior_week[column].sum() if column in prior_week else 0
            item[f"{name}_velocity_change_7d"] = (
                (recent_total - previous_total) / (previous_total + 1)
            )

        for ledger in ("booknow", "cinepos"):
            for band in LEAD_BANDS:
                column = f"{ledger}_lead_{band}"
                total = (
                    window_frames[28][column].sum()
                    if column in window_frames[28]
                    else 0
                )
                item[f"{ledger}_lead_{band}_avg_28d"] = total / 28

        feature_rows.append(item)

    return pd.DataFrame(feature_rows)


def make_training_frame(
    visits,
    theaters,
    bookings_daily=None,
    theater_relation=None,
):
    """Create model-training features for all observed visit rows."""
    visits = visits.copy()
    visits["show_date"] = pd.to_datetime(visits["show_date"]).dt.normalize()
    visits = visits.sort_values("show_date").reset_index(drop=True)

    features = make_features(
        history=visits,
        rows=visits[["book_theater_id", "show_date"]],
        theaters=theaters,
        bookings_daily=bookings_daily,
        theater_relation=theater_relation,
    )
    features[TARGET] = visits[TARGET].to_numpy()
    return features
