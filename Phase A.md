# Phase A Data and Model Walkthrough

This document explains the current project pipeline: the competition CSVs, the scripts that load and prepare them, the files written to `data/processed/`, and the feature, training, evaluation, and forecasting code. It describes the code as it currently exists; it also calls out assumptions that affect forecast reliability.

## 1. Pipeline at a glance

```text
data/raw/*.csv
       |
       v
src/data_loader.py       reads files and checks required columns
       |
       v
src/reconcile.py         cleans target rows, maps IDs, aggregates ledgers
       |
       +---- src/process_data.py saves CSVs under data/processed/
       |
       v
src/features.py          builds calendar, theatre, history, and ledger features
       |
       +---- src/train.py trains the Random Forest model
       +---- src/evaluate.py validates it on later historical dates
       +---- src/forecast.py predicts test.csv rows and writes submission.csv
```

`process_data.py` writes the processed CSV files. `forecast.py` does not read those saved CSVs: it loads the raw files again and calls the in-memory preparation functions before making predictions.

## 2. The raw competition files

The source CSVs are in `data/raw/`. The source files are not changed by the loader or reconciliation code.

| Raw file | Rows | Columns | Purpose |
|---|---:|---|---|
| `booknow_visits.csv` | 176,861 | `book_theater_id`, `show_date`, `audience_count` | Historical BookNow audience target, at theatre/date level. The project definition is total audience across all shows at that theatre on that date. |
| `booknow_booking.csv` | 46,735 | `book_theater_id`, `show_datetime`, `booking_datetime`, `tickets_booked` | BookNow online booking transactions. `show_datetime` identifies the show; `booking_datetime` records when a transaction was made. |
| `cinePOS_booking.csv` | 1,366,605 | `cine_theater_id`, `show_datetime`, `booking_datetime`, `tickets_sold` | CinePOS counter-sale transactions, with show and transaction timestamps. |
| `booknow_theaters.csv` | 829 | `book_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude` | BookNow theatre attributes. The file has repeated theatre IDs; the feature builder keeps the first metadata row for each ID. |
| `cinePOS_theaters.csv` | 4,690 | `cine_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude` | CinePOS theatre attributes. This table is loaded but is not currently used to build features or to derive new ID mappings. |
| `movie_theater_id_relation.csv` | 150 | `book_theater_id`, `cine_theater_id` | Supplied crosswalk between the two theatre ID systems. |
| `date_info.csv` | 547 | `show_date`, `day_of_week` | Calendar lookup. Loaded, but the feature code currently derives calendar fields directly from the date instead. |
| `test.csv` | 37,185 | `ID`, `book_theater_id`, `show_date` | Theatre/date rows for which the model must predict audience count. |
| `sample_submission.csv` | 37,185 | `ID`, `audience_count` | Submission template; the required output columns are `ID,audience_count`. |

## 3. How the data preparation scripts work

### `src/data_loader.py`

`FILE_SPECS` defines the required filename, columns, and date columns for each of the nine CSVs. `load_data()` then:

1. Locates `data/raw/` relative to the project directory.
2. Checks that each required file exists.
3. Reads each CSV into a pandas DataFrame and parses the configured dates/timestamps.
4. Checks for required columns and raises an error if any are missing.
5. Returns the tables in a dictionary under short names such as `visits`, `booknow_booking`, `cinepos_booking`, `theater_relation`, `test`, and `sample_submission`.

`print_data_summary()` can print row counts, column names, missing-value counts, and date ranges. The loader itself does not deduplicate or aggregate data.

### `src/reconcile.py`

`prepare_data()` takes the loaded tables and returns the original tables plus prepared `visits`, `theater_relation`, and `bookings_daily` tables.

#### Audience history

- Parses and normalizes `show_date`.
- Removes exact duplicate rows.
- Groups remaining records by `book_theater_id` and `show_date`.
- If a theatre/date still has multiple target values, it takes their **mean**.

In the current data, 10 exact duplicate visit rows are removed. The remaining repeated theatre/date keys are reduced by averaging, resulting in 176,686 rows in `visits_clean.csv`. The code chooses a mean because the source has no show ID to explain repeated keys. This is an assumption: if repeated rows represent separate show audiences, a mean would not represent the daily total.

#### Theatre ID crosswalk

- Removes exact duplicate rows from `movie_theater_id_relation.csv`.
- Raises an error if a BookNow ID maps to more than one CinePOS ID.
- Uses this relation to attach CinePOS transactions to BookNow IDs.

The current code does **not** match theatres by latitude/longitude. Coordinate inspection found many repeated coordinates and no shared coordinate pair that uniquely identifies one BookNow ID and one CinePOS ID. Therefore, coordinates cannot currently provide a safe automatic fallback. The relation file remains the trusted crosswalk.

#### Daily ledger aggregation

- BookNow rows are grouped by BookNow ID and the date portion of `show_datetime`; `tickets_booked` is summed.
- CinePOS rows are first inner-joined to the supplied crosswalk, then grouped by mapped BookNow ID and the date portion of `show_datetime`; `tickets_sold` is summed.
- The two daily summaries are outer-joined by BookNow ID and show date. Missing ledger totals are filled with zero.

The inner join means CinePOS transactions with no supplied mapping do not appear in `bookings_daily`. At the time inspected, this excludes 1,346,295 of 1,366,605 CinePOS transaction rows from that combined canonical table. The raw source CSV remains intact, but those rows are not currently saved in a separate processed output. Also, `booking_datetime` is parsed by the loader but is not used in this aggregation or filtered to a forecast cutoff.

### `src/process_data.py`

This is the command-line runner for the preparation stage. It calls `load_data()`, then `prepare_data()`, then `save_processed_data()` to write the selected tables to `data/processed/`.

## 4. Files in `data/processed/`

The current folder contains these files:

| Processed file | Rows | Columns | What it contains |
|---|---:|---|---|
| `visits_clean.csv` | 176,686 | `book_theater_id`, `show_date`, `audience_count` | Deduplicated and grouped historical target rows. Remaining repeated theatre/date target values have been averaged. |
| `bookings_daily.csv` | 25,398 | `book_theater_id`, `show_date`, `tickets_booked`, `tickets_sold` | Combined BookNow and *mapped* CinePOS totals by show date. Unmapped CinePOS rows are absent; a missing ledger total is represented as zero. |
| `test_clean.csv` | 37,185 | `ID`, `book_theater_id`, `show_date` | Test rows written from the loaded test table; it is not otherwise transformed. |
| `booknow_theaters_clean.csv` | 829 | `book_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude` | BookNow theatre metadata written from the loaded table. Despite the filename, `save_processed_data()` does not clean this table. |
| `theater_relation_clean.csv` | 150 | `book_theater_id`, `cine_theater_id` | Crosswalk after exact duplicate rows are removed and mapping uniqueness is checked. |

The preparation script does not write processed versions of `cinePOS_theaters.csv`, `date_info.csv`, or `sample_submission.csv`.

## 5. Feature creation — `src/features.py`

The target column is `audience_count`. `make_features(history, rows, theaters, bookings_daily)` creates one feature row for each requested theatre/date.

### Calendar and theatre features

- Theatre ID, area, type, latitude, and longitude.
- Day of week, month, day of month, day of year, and weekend flag.
- Theatre metadata is deduplicated by `book_theater_id` before joining; pandas keeps the first row for a repeated ID.

### Audience-history features

The function looks only at historical audience values with a date strictly before the requested date. It creates exact calendar-date lags of 1, 7, 14, and 28 days, plus rolling means and standard deviations over the previous 7, 14, and 28 calendar days. Missing lag/window values remain missing at this stage and are handled later by `train.py`.

These are calendar-day lookups, not “the Nth previous row” lookups. A missing date therefore produces a missing lag instead of shifting to a different day.

### Ledger features

It adds `prior_tickets_booked` and `prior_tickets_sold` from the **previous show date**. If there is no daily ledger record, it uses zero.

This is not yet the forecast-time logic you described: the feature code uses prior-date ledger totals, not online bookings for the requested future show date as of the current request time. It also does not filter ledger transactions by `booking_datetime`. Thus, ledger features need revision before they faithfully represent what would have been known when a forecast was requested.

`make_training_frame()` calls `make_features()` for each historical visit row and adds that row's observed audience count as the training label.

## 6. Model training — `src/train.py`

The current model is a **Random Forest regressor**, not LightGBM. LightGBM is listed as a dependency, but `train.py` imports and trains scikit-learn's `RandomForestRegressor`.

### What a decision tree does

A regression decision tree repeatedly splits training examples into smaller groups using rules about feature values. For example, it may split on whether the last-week audience count is above a threshold, whether the date is a weekend, or whether the theatre is in a particular area. Each final group (leaf) predicts a typical audience value learned from its training examples.

The forest contains 250 decision trees. Scikit-learn's default bootstrap sampling gives each tree a resampled training set; `max_features=0.8` also limits the feature candidates considered at each split. These sources of randomness make the trees differ. The Random Forest averages their predictions, which reduces the instability of relying on a single tree. It is a supervised regression model; it is not a clustering or theatre-segmentation algorithm.

### Encoding and settings

- `book_theater_id`, `theater_area`, and `theater_type` are one-hot encoded.
- At prediction time, feature columns are re-aligned to the training feature list; unseen category columns receive zero.
- Infinite values are replaced with missing values, then missing feature values are filled with `-1`.
- Random Forest settings: 250 trees (`n_estimators=250`), minimum leaf size 3, `max_features=0.8`, random seed 42, and all available CPU cores (`n_jobs=-1`).
- The bundle stores both the fitted model and its ordered feature-column list. `train.py` can pickle it to `models/footfall_model.pkl`.

## 7. Validation — `src/evaluate.py`

`temporal_validation()` sorts audience history by date and holds out the final 28 calendar days in the available history. It trains on earlier rows, then predicts each holdout date in sequence. After predicting a date, it appends those predictions (not actual values) to the history used for later holdout dates. This simulates multi-day forecasting without feeding later actual audience counts into the next day's lag features.

It reports:

- **RMSE**: square root of average squared error, measured in audience-count units. This is the competition's official score.
- **WAPE**: total absolute error divided by total actual audience.
- **MAE**: average absolute error, also in audience-count units.

The holdout is 28 days, while the competition test horizon is 53 days. More importantly, `bookings_daily` is prepared from all transaction rows and passed into validation without a simulated `booking_datetime` cutoff. Ledger features can therefore include information that would not have been available at the simulated forecast date. The current validation score is not a fully cutoff-safe estimate for ledger-enabled forecasting.

## 8. Test forecasting — `src/forecast.py`

`forecast(data, model_bundle=None)` performs the following:

1. Starts from prepared historical visits and the test rows.
2. Uses a supplied model bundle, or trains a new model on all available visit history if no bundle was supplied.
3. Processes test dates in order. For each date, it creates features, predicts all rows for that date, and clips negative predictions to zero.
4. Appends the predicted audience counts to history so later test dates can use earlier predictions as lags.
5. Returns predictions with IDs and dates.

The `main()` function loads and prepares the raw files, loads `models/footfall_model.pkl` if it exists (otherwise forecast will train a fresh model), aligns predictions to the IDs in `sample_submission.csv`, checks that every ID received a prediction, and writes `outputs/submission.csv` with the required `ID,audience_count` columns.

At the time of this document, `models/footfall_model.pkl` was not present. The existing `outputs/submission.csv` had 37,185 rows, no missing predictions or duplicate IDs, and predictions from about 3.07 to 210.68. These checks confirm output shape and basic numeric validity, not forecast accuracy.

## 9. Current implementation gaps to keep in mind

1. **Coordinate fallback:** not implemented; the supplied theatre coordinates are not unique enough for safe automatic mapping. Keep the official crosswalk authoritative and preserve unmapped POS data separately if it is to remain available for later analysis.
2. **Unmapped CinePOS transactions:** excluded from `bookings_daily` by an inner join and not currently written to a separate processed file.
3. **Forecast-time ledger cutoff:** `booking_datetime` is not used to filter features as of the prediction request time.
4. **Target duplicate rule:** remaining duplicate theatre/date audience values are averaged, which must be confirmed as the right interpretation for a total-across-shows target.
5. **Validation horizon and leakage:** validation holds out 28 days rather than the 53-day test horizon and does not reconstruct ledger availability at each historical cutoff.
6. **Metadata usage:** `cinePOS_theaters.csv` and `date_info.csv` are loaded but not used by the model feature function. Calendar features are derived directly from dates.
7. **Current algorithm:** Random Forest is used. LightGBM and CatBoost are not used by the present training script.

## 10. The two data caveats

### Caveat 1: repeated theatre/date audience rows are averaged

`reconcile.py` removes exact duplicate visit rows, then groups the remaining rows by BookNow theatre ID and show date. If a group still contains more than one `audience_count`, it takes the mean. This produces one row per theatre/date, but it does not add the values together. Since the intended target is total audience across every show at the theatre on that date, confirm that repeated rows are duplicate measurements rather than separate show counts before treating the mean as the correct cleaning rule. The source has no show ID to resolve that distinction automatically.

### Caveat 2: unmapped CinePOS transaction rows do not enter the combined daily ledger

`reconcile.py` inner-joins CinePOS bookings to the supplied 150-row ID crosswalk. Any CinePOS theatre ID absent from that crosswalk is excluded from `bookings_daily.csv`. In the inspected data, 1,346,295 of 1,366,605 CinePOS transaction rows were excluded from that canonical table. This does not delete or edit `data/raw/cinePOS_booking.csv`, but those rows are not preserved as a separate processed table by the current script.

Coordinate matching is not currently a safe automatic fix: the metadata contains many repeated coordinates, and inspection found no shared coordinate pair that maps exactly one BookNow ID to exactly one CinePOS ID. Keep the official crosswalk as the authoritative mapping. To retain the excluded sales for later analysis, preserve them in a separate CinePOS-ID/date table until a reliable mapping or a validated non-theatre-level use is available.

## 11. Running the existing scripts

From the project root in PowerShell, with the virtual environment active:

```powershell
python src/process_data.py
python src/evaluate.py
python src/forecast.py
```

The first command recreates the processed CSVs, the second prints historical validation metrics, and the third writes the competition-format forecast CSV.
