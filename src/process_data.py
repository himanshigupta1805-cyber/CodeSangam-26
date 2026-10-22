"""Run the raw-data loading and reconciliation steps, then save processed CSVs."""
from pathlib import Path

from data_loader import load_data
from reconcile import prepare_data, save_processed_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    tables = prepare_data(load_data())
    saved_files = save_processed_data(tables, PROCESSED_DATA_DIR)

    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")
    for path in saved_files:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
