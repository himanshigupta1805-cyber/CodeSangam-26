# Audience forecasting: BookNow + CinePOS (LightGBM + CatBoost hybrid)

- `features.py`    cleaning, mapping-key merge, leak-free anchor-based lag/rolling features
- `validate.py`    first validation run (log-target models, showed log target hurts RMSE)
- `experiments.py` objective comparison (raw / Poisson / ratio / log)
- `final.py`       final 0.8 LightGBM + 0.2 CatBoost (Poisson) fit, writes submission.csv
- `submission.csv` predictions for 2024-01-07 .. 2024-02-28

Run order: `python features.py` then `python final.py`. Paths in the scripts point to
`/mnt/user-data/uploads/` (input CSVs) and `/home/claude/` (intermediates); edit them for local use.
Validation RMSE (train to 2023-11-11, predict next 53 days): blend ~26.60, baseline 26.73.
