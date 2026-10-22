"""Basic key checks and daily booking aggregation."""
import pandas as pd


def prepare_data(tables):
    """Check target/mapping keys and aggregate ledgers by theater and show date.

    Booking sums are historical inputs. The feature builder only uses their
    previous-day values, avoiding same-day booking lookahead.
    """
    visits = tables["visits"].copy()
    visits["show_date"] = pd.to_datetime(visits["show_date"])
    if visits.duplicated(["book_theater_id", "show_date"]).any():
        examples = visits.loc[visits.duplicated(["book_theater_id", "show_date"], keep=False),
                              ["book_theater_id", "show_date"]].head()
        raise ValueError(f"Visits has duplicate theater/date labels, e.g. {examples.to_dict('records')}")

    relation = tables["theater_relation"].drop_duplicates().copy()
    if relation["book_theater_id"].duplicated().any():
        raise ValueError("A BookNow theater maps to multiple CinePOS theaters.")

    book = tables["booknow_booking"].copy()
    book["show_date"] = pd.to_datetime(book["show_datetime"]).dt.normalize()
    book_daily = book.groupby(["book_theater_id", "show_date"], as_index=False)["tickets_booked"].sum()

    cine = tables["cinepos_booking"].copy()
    cine = cine.merge(relation, on="cine_theater_id", how="inner", validate="many_to_one")
    cine["show_date"] = pd.to_datetime(cine["show_datetime"]).dt.normalize()
    cine_daily = cine.groupby(["book_theater_id", "show_date"], as_index=False)["tickets_sold"].sum()

    bookings = book_daily.merge(cine_daily, on=["book_theater_id", "show_date"], how="outer")
    bookings = bookings.fillna({"tickets_booked": 0, "tickets_sold": 0})
    return {**tables, "visits": visits, "bookings_daily": bookings}
