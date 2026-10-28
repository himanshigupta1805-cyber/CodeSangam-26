from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from catboost import CatBoostRegressor
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT = ROOT / "submission.csv"
tr = pd.read_parquet(PROCESSED_DIR / "train.parquet")
te = pd.read_parquet(PROCESSED_DIR / "test.parquet")
ids = pd.read_csv(PROCESSED_DIR / "ids.csv").iloc[:, 0].values
days = pd.to_datetime(pd.read_csv(PROCESSED_DIR / "days.csv").iloc[:, 0])
test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["show_date"])
sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

te["book_theater_id"] = ids[te.book_theater_id.values]
te["show_date"] = days.values[te.date_idx.values]
te = test.merge(te, on=["book_theater_id", "show_date"], how="left")
assert len(te) == len(test) and te.h.notna().all()

drop = ["y", "anchor", "date_idx", "book_theater_id", "show_date", "doy", "woy", "ID"]
F = [c for c in tr.columns if c not in drop]
base = lambda d: d.wd_mean_8w.fillna(d.roll_mean_28).fillna(d.all_mean).fillna(42).values
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

def write(pred, tag):
    s = pd.DataFrame({"ID": te.ID.values, "audience_count": np.clip(pred, 2, None)})
    s = sample[["ID"]].merge(s, on="ID", how="left")
    assert s.audience_count.notna().all() and len(s) == len(sample)
    s.to_csv(OUT, index=False)
    print("wrote", tag, s.audience_count.describe().to_dict(), flush=True)

write(base(te), "baseline fallback")

def fit_lgb(d, seed=0):
    # More trees + smaller lr + wider leaves + full subsample for better convergence
    m = lgb.LGBMRegressor(objective="poisson", n_estimators=600, learning_rate=0.04, num_leaves=63,
                          min_child_samples=150, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=10, n_jobs=1, verbose=-1, random_state=seed)
    return m.fit(d[F], d.y.values)

def fit_cat(d, seed=0):
    # More iterations + lower lr for smoother convergence, same regularization
    m = CatBoostRegressor(loss_function="Poisson", iterations=600, learning_rate=0.07, depth=6,
                          l2_leaf_reg=10, thread_count=1, verbose=0, random_seed=seed,
                          allow_writing_files=False)
    return m.fit(d[F], d.y.values)

best_w = 0.75  # slight shift toward CatBoost which complements LGB on count data

# ---- Final fit on ALL training data (was 50% subsample — full data reduces variance)
full = tr  # use all data, not tr.sample(frac=0.5)
pl = fit_lgb(full).predict(te[F]); write(pl, "LGB only (interim)")
np.save(PROCESSED_DIR / "pl.npy", pl)
pc = fit_cat(full).predict(te[F]); np.save(PROCESSED_DIR / "pc.npy", pc)
final = best_w * pl + (1 - best_w) * pc
write(final, f"hybrid w_lgb={best_w:.2f}")
print("DONE", flush=True)
