"""Reconcile theater IDs and aggregate daily booking and lead-time signals."""

from pathlib import Path

import numpy as np
import pandas as pd


LEAD_TIME_BINS = {
    "0_24h": (0, 24),
    "24_72h": (24, 72),
    "72_168h": (72, 168),
    "168h_plus": (168, np.inf),
}


def _aggregate_ledger(frame, ticket_column, prefix):
    """Aggregate tickets by show date and booking lead-time band."""
    ledger = frame.copy()
    ledger["show_datetime"] = pd.to_datetime(ledger["show_datetime"])
    ledger["booking_datetime"] = pd.to_datetime(ledger["booking_datetime"])
    ledger["show_date"] = ledger["show_datetime"].dt.normalize()
    ledger["lead_hours"] = (
        ledger["show_datetime"] - ledger["booking_datetime"]
    ).dt.total_seconds() / 3600

    # A transaction timestamped after its show is not an advance signal and
    # should not be used to construct leakage-safe lead-time features.
    ledger = ledger[ledger["lead_hours"] >= 0].copy()
    keys = ["book_theater_id", "show_date"]
    totals = (
        ledger.groupby(keys, as_index=False)[ticket_column]
        .sum()
    )

    for band, (lower, upper) in LEAD_TIME_BINS.items():
        in_band = ledger[ledger["lead_hours"].ge(lower)]
        if np.isfinite(upper):
            in_band = in_band[in_band["lead_hours"] < upper]
        band_totals = (
            in_band.groupby(keys, as_index=False)[ticket_column]
            .sum()
            .rename(columns={ticket_column: f"{prefix}_lead_{band}"})
        )
        totals = totals.merge(band_totals, on=keys, how="left", validate="one_to_one")

    band_columns = [f"{prefix}_lead_{band}" for band in LEAD_TIME_BINS]
    totals[band_columns] = totals[band_columns].fillna(0)
    return totals


def prepare_data(tables):
    """Prepare visit and booking tables for feature creation.

    Exact duplicate visit rows are dropped. If a theater/date still has
    multiple audience counts, those values are averaged because the source
    has no show ID that would tell us whether to sum them.
    """
    visits = tables["visits"].copy()
    visits["show_date"] = pd.to_datetime(visits["show_date"]).dt.normalize()
    visits = visits.drop_duplicates()
    visits = (
        visits.groupby(["book_theater_id", "show_date"], as_index=False)["audience_count"]
        .mean()
    )

    relation = tables["theater_relation"].drop_duplicates().copy()
    if relation["book_theater_id"].duplicated().any():
        raise ValueError("A BookNow theater maps to multiple CinePOS theaters.")

    booknow = tables["booknow_booking"].copy()
    booknow_daily = _aggregate_ledger(booknow, "tickets_booked", "booknow")

    cinepos = tables["cinepos_booking"].merge(
        relation,
        on="cine_theater_id",
        how="inner",
        validate="many_to_one",
    )
    cinepos_daily = _aggregate_ledger(cinepos, "tickets_sold", "cinepos")

    bookings_daily = booknow_daily.merge(
        cinepos_daily,
        on=["book_theater_id", "show_date"],
        how="outer",
        validate="one_to_one",
    )
    numeric_columns = bookings_daily.select_dtypes(include="number").columns
    bookings_daily[numeric_columns] = bookings_daily[numeric_columns].fillna(0)

    return {
        **tables,
        "visits": visits,
        "bookings_daily": bookings_daily,
        "theater_relation": relation,
    }


def save_processed_data(tables, output_dir):
    """Save prepared tables as CSV files in the processed-data directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables_to_save = {
        "visits_clean.csv": tables["visits"],
        "bookings_daily.csv": tables["bookings_daily"],
        "test_clean.csv": tables["test"],
        "booknow_theaters_clean.csv": tables["booknow_theaters"],
        "theater_relation_clean.csv": tables["theater_relation"],
    }

    for filename, dataframe in tables_to_save.items():
        dataframe.to_csv(output_dir / filename, index=False)

    return [output_dir / filename for filename in tables_to_save]
