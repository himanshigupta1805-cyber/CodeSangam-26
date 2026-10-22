"""Load and schema-check the raw Footfall Forecast CSV files."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILE_SPECS = {
    "visits": ("booknow_visits.csv", ["book_theater_id", "show_date", "audience_count"], ["show_date"]),
    "booknow_booking": ("booknow_booking.csv", ["book_theater_id", "show_datetime", "booking_datetime", "tickets_booked"], ["show_datetime", "booking_datetime"]),
    "cinepos_booking": ("cinePOS_booking.csv", ["cine_theater_id", "show_datetime", "booking_datetime", "tickets_sold"], ["show_datetime", "booking_datetime"]),
    "booknow_theaters": ("booknow_theaters.csv", ["book_theater_id", "theater_type", "theater_area", "latitude", "longitude"], []),
    "cinepos_theaters": ("cinePOS_theaters.csv", ["cine_theater_id", "theater_type", "theater_area", "latitude", "longitude"], []),
    "date_info": ("date_info.csv", ["show_date", "day_of_week"], ["show_date"]),
    "theater_relation": ("movie_theater_id_relation.csv", ["book_theater_id", "cine_theater_id"], []),
    "test": ("test.csv", ["ID", "book_theater_id", "show_date"], ["show_date"]),
    "sample_submission": ("sample_submission.csv", ["ID", "audience_count"], []),
}


def load_data(raw_data_dir=RAW_DATA_DIR):
    """Return all raw tables in a dictionary, keyed by short table name."""
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw-data directory does not exist: {raw_data_dir}")

    tables = {}
    for name, (filename, required, date_columns) in FILE_SPECS.items():
        path = raw_data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required CSV is missing: {path}")
        frame = pd.read_csv(path, parse_dates=date_columns)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")
        tables[name] = frame
    return tables


def print_data_summary(tables):
    """Print sizes, column names, missing counts, and date ranges."""
    for name, frame in tables.items():
        print(f"\n{name}: {len(frame):,} rows")
        print("Columns:", ", ".join(frame.columns))
        print("Missing values:", frame.isna().sum().to_dict())
        for column in ("show_date", "show_datetime", "booking_datetime"):
            if column in frame and not frame[column].empty:
                print(f"{column}: {frame[column].min()} to {frame[column].max()}")


if __name__ == "__main__":
    print_data_summary(load_data())
