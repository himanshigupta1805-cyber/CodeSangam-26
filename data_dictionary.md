# Data dictionary and Phase A decisions

This document records what is known from the supplied project files and the competition overview screenshot. Items marked **Open** need confirmation; names and sample values alone are not treated as authoritative definitions.

## Competition decisions

| Decision | Current answer | Evidence | Status |
|---|---|---|---|
| Prediction target | Total number of audience members at one theatre across all shows on a particular date | User clarification; target column is `audience_count` in `booknow_visits.csv` | Confirmed |
| Prediction grain | One theatre/date pair | Competition overview says each (theatre, date) pair in `test.csv`; test columns are `book_theater_id` and `show_date` | Confirmed |
| Test horizon | 53 days following the training period | Competition overview screenshot | Confirmed for the supplied test set |
| Predicted value | `audience_count` for each theatre/date row | User clarification and competition overview | Confirmed |
| Submission format | CSV with `ID,audience_count`; `ID` must match `test.csv` in the form `{book_theater_id}_{YYYY-MM-DD}` | Kaggle Overview > Evaluation; matches `sample_submission.csv` | Confirmed |
| Data sources | BookNow is advance online bookings; CinePOS is on-site point-of-sale sales | Competition overview screenshot | Confirmed at source level |
| Cross-source theatre mapping | `movie_theater_id_relation.csv` links BookNow and CinePOS theatre IDs | Competition overview screenshot and mapping CSV | Confirmed |
| Forecast cutoff / information availability | Cutoff is the timestamp when the user requests a forecast. Use only data recorded by that time. For same-day forecasts, use online bookings and counter sales so far; for future dates, use online bookings so far and no counter sales not yet made. | User clarification | Confirmed |
| Official leaderboard metric | RMSE on `audience_count` | Kaggle Overview > Evaluation | Confirmed |
| Additional diagnostic metric | WAPE may be tracked for interpretation, but it is not the competition's stated score | User confirmed WAPE is among the discussed metrics; Kaggle Overview names RMSE as the score | Optional diagnostic |
| Exact meaning of `audience_count` | Total audience members at a particular theatre on a date, including all shows that day | User clarification | Confirmed |
| Stack / prototype dashboard | Python; Streamlit interactive dashboard | User instruction; `requirements.txt` includes Streamlit | Confirmed as project choice |

## Competition-specific rules (Kaggle Rules page)

- Only registered Logical Rhythm participants may enter; teams may have up to three members, and team merges require organizer approval.
- External data is prohibited. Training and prediction must use only the provided dataset files.
- Do not share code, notebooks, or predictions publicly or with other teams before the competition ends. Kaggle's foundational rules also prohibit private sharing of competition code/data outside the team.
- Final submission must be made through a committed Kaggle notebook using **Save and Run All**, not a quick-save.
- Final ranking combines leaderboard score with code quality, including preprocessing, visualization, and clarity of reasoning.
- The Overview > Evaluation section specifies **RMSE on `audience_count`** and a CSV with `ID,audience_count`; IDs must match `test.csv` exactly.
- The prediction-time cutoff is not specified on the Rules/Overview pages inspected; cutoff-safe ledger availability still needs confirmation.

## File inventory

Row counts below are data rows, excluding the header. Column meanings are best-effort descriptions inferred from names and examples unless explicitly confirmed above.

| File | Rows | Purpose / notes |
|---|---:|---|
| `data/raw/booknow_visits.csv` | 176,861 | BookNow historical target records |
| `data/raw/booknow_booking.csv` | 46,735 | BookNow booking transactions, with show and booking timestamps |
| `data/raw/booknow_theaters.csv` | 829 | BookNow theatre attributes |
| `data/raw/cinePOS_booking.csv` | 1,366,605 | CinePOS sales transactions, with show and booking timestamps |
| `data/raw/cinePOS_theaters.csv` | 4,690 | CinePOS theatre attributes |
| `data/raw/date_info.csv` | 547 | Calendar date and weekday lookup |
| `data/raw/movie_theater_id_relation.csv` | 150 | BookNow-to-CinePOS theatre ID mapping |
| `data/raw/test.csv` | 37,185 | Prediction rows for the competition test set |
| `data/raw/sample_submission.csv` | 37,185 | Example output with `ID` and `audience_count` |

## Column reference

### `booknow_visits.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `book_theater_id` | BookNow theatre identifier | String/category | `book_00001` | Yes, as a prediction key | Confirm uniqueness with `show_date` at the official grain |
| `show_date` | Date of the audience target | Date | `2023-01-13` | Yes | Date/timezone convention not documented |
| `audience_count` | Target audience value | Numeric | `50` | No for a future target | Exact operational definition is open |

### `booknow_booking.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `book_theater_id` | BookNow theatre identifier | String/category | `book_00244` | Yes | Join to BookNow theatre records; aggregation key still needs confirmation |
| `show_datetime` | Scheduled show date and time | Datetime | `2023-01-01 19:00:00` | Likely known | Confirm whether the prediction grain aggregates multiple shows |
| `booking_datetime` | Time the booking was recorded | Datetime | `2023-01-01 16:00:00` | Only if on or before forecast cutoff | Historical as-of snapshots/cutoff policy are unknown |
| `tickets_booked` | Ticket quantity in a booking record | Numeric/integer | `1` | Only records available by cutoff | Cancellation/refund/reversal fields are not present in this listed schema |

### `booknow_theaters.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `book_theater_id` | BookNow theatre identifier | String/category | `book_00093` | Yes | Check uniqueness |
| `theater_type` | Theatre type/category | String/category | `Drama` | Likely | Meaning and source definition are undocumented |
| `theater_area` | Theatre area/category | String/category | `Area_001` | Likely | Meaning and source definition are undocumented |
| `latitude` | Theatre latitude | Numeric | `22.61923281872136` | Likely | Validate ranges and missingness |
| `longitude` | Theatre longitude | Numeric | `78.11301670016887` | Likely | Validate ranges and missingness |

### `cinePOS_booking.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `cine_theater_id` | CinePOS theatre identifier | String/category | `cinePOS_00001` | Yes | Map through `movie_theater_id_relation.csv` where applicable |
| `show_datetime` | Scheduled show date and time | Datetime | `2023-01-01 11:00:00` | Likely known | Confirm relationship to prediction grain |
| `booking_datetime` | Time the POS sale was recorded | Datetime | `2023-01-01 09:00:00` | Only if on or before forecast cutoff | Historical snapshots/cutoff policy are unknown |
| `tickets_sold` | Ticket quantity in a POS sale record | Numeric/integer | `1` | Only records available by cutoff | Cancellation/refund/reversal fields are not present in this listed schema |

### `cinePOS_theaters.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `cine_theater_id` | CinePOS theatre identifier | String/category | `cinePOS_05466` | Yes | Check uniqueness |
| `theater_type` | Theatre type/category | String/category | `Other` | Likely | Meaning and source definition are undocumented |
| `theater_area` | Theatre area/category | String/category | `Area_104` | Likely | Meaning and source definition are undocumented |
| `latitude` | Theatre latitude | Numeric | `22.61923281872136` | Likely | Validate ranges and missingness |
| `longitude` | Theatre longitude | Numeric | `78.11301670016887` | Likely | Validate ranges and missingness |

### `date_info.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `show_date` | Calendar date | Date | `2023-01-01` | Yes | Check date coverage and duplicates |
| `day_of_week` | Weekday name for the date | String/category | `Sunday` | Yes | Can be derived from date; validate consistency |

### `movie_theater_id_relation.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `book_theater_id` | BookNow theatre identifier | String/category | `book_00509` | Yes | Mapping coverage and uniqueness need auditing |
| `cine_theater_id` | Corresponding CinePOS theatre identifier | String/category | `cinePOS_01261` | Yes | Do not assume all IDs map one-to-one until validated |

### `test.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `ID` | Competition row identifier | String | `book_00001_2024-01-07` | Yes | Exact construction/contract should be verified |
| `book_theater_id` | BookNow theatre key | String/category | `book_00001` | Yes | Part of prediction key per overview |
| `show_date` | Date to predict | Date | `2024-01-07` | Yes | Part of prediction key per overview |

### `sample_submission.csv`

| Column | Inferred meaning | Likely type | Example | Known at forecast time? | Notes |
|---|---|---|---|---|---|
| `ID` | Competition row identifier matching test rows | String | `book_00001_2024-01-07` | Yes | Verify exact output requirements in official rules |
| `audience_count` | Predicted target value | Numeric | `41.98` | It is the model output | Sample value is illustrative; formatting/rounding rules are unknown |

## Reconciliation and data quality decisions

- Prediction key from the supplied overview: theatre plus date; the project representation appears to be `book_theater_id` plus `show_date`.
- Keep BookNow and CinePOS transactions as separate aggregates before combining them, to avoid accidental double counting.
- Use historical audience counts and BookNow/CinePOS ledger information available at the forecast request time. Filter transactions by their recorded timestamp so no later information enters the prediction.
- For a target date equal to the forecast date, include BookNow and CinePOS/counter tickets recorded up to the forecast timestamp. For a future target date, include eligible BookNow advance bookings recorded up to the forecast timestamp; do not use future CinePOS/counter sales.
- For historical validation, reconstruct each simulated forecast using only transactions recorded by that simulated cutoff. Check whether transaction records are complete and whether cancellations/refunds/reversals need correction before treating the filtered history as a true as-of snapshot.
- No cancellation/refund/status columns appear in the transaction file schemas listed above. Confirm whether other files or source rules define reversals.
- Validate mapping uniqueness, unmatched IDs, duplicate rows, missing values, date parsing, negative/zero quantities, and row counts before/after joins.
- Distinguish a measured zero from missing or unavailable data; source semantics have not yet been confirmed.
- No canonical daily table or reproducible profiling report is present in the repository at the time this dictionary was drafted.

## Questions still needing confirmation

1. Raw-file Git policy: the user wants the provided files kept out of `.gitignore` and included in Git. Verified: `data/raw/` is not ignored, and all nine raw CSVs are currently tracked by Git. Keep repository access within the competition team and do not publish/share competition files or work outside the team before the competition ends.

## Provenance and scope

Competition-level facts above were taken from the supplied overview screenshot. Schema names, sample values, and row counts were inspected from the CSVs. Column meanings labeled “inferred” should be checked against the official data dictionary/rules before they are treated as authoritative.
