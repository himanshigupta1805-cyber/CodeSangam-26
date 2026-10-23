"""Reconcile theater IDs and aggregate data into daily tables."""

from pathlib import Path

import pandas as pd


def prepare_data(tables):
    """Prepare visit and booking tables for feature creation.

    Exact duplicate visit rows are dropped. If a theater/date still has
    multiple audience counts, those values are averaged because the source
    has no show ID that would tell us whether to sum them.
    """
    # Prepare audience-count history.
    visits = tables["visits"].copy()
    visits["show_date"] = pd.to_datetime(visits["show_date"]).dt.normalize()

    # Remove rows identical across every column, then leave one value per
    # theater/date by averaging any remaining duplicate labels.
    visits = visits.drop_duplicates()
    visits = (
        visits.groupby(
            ["book_theater_id", "show_date"],
            as_index=False,
        )["audience_count"]
        .mean()
    )

    # Remove exact duplicate mappings. Each BookNow theater should map to
    # at most one CinePOS theater.
    relation = tables["theater_relation"].drop_duplicates().copy()

    if relation["book_theater_id"].duplicated().any():
        raise ValueError(
            "A BookNow theater maps to multiple CinePOS theaters."
        )

    # Aggregate BookNow bookings by theater and show date.
    booknow = tables["booknow_booking"].copy()
    booknow["show_date"] = (
        pd.to_datetime(booknow["show_datetime"]).dt.normalize()
    )
    booknow_daily = (
        booknow.groupby(
            ["book_theater_id", "show_date"],
            as_index=False,
        )["tickets_booked"] 
        .sum()
    )

    # Link CinePOS IDs to BookNow IDs, then aggregate by theater and show date.
    cinepos = tables["cinepos_booking"].copy()
    cinepos = cinepos.merge(
        relation,
        on="cine_theater_id",
        how="inner",
        validate="many_to_one",
    )
    cinepos["show_date"] = (
        pd.to_datetime(cinepos["show_datetime"]).dt.normalize()
    )
    cinepos_daily = (
        cinepos.groupby(
            ["book_theater_id", "show_date"],
            as_index=False,
        )["tickets_sold"]
        .sum()
    )

    # Combine both daily booking summaries. Fill a missing ledger total with 0.
    bookings_daily = booknow_daily.merge(
        cinepos_daily,
        on=["book_theater_id", "show_date"],
        how="outer",
    )
    bookings_daily[["tickets_booked", "tickets_sold"]] = (
        bookings_daily[["tickets_booked", "tickets_sold"]].fillna(0)
    )

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