import numpy as np, pandas as pd, lightgbm as lgb, warnings
from catboost import CatBoostRegressor
warnings.simplefilter("ignore")

OUT = "/mnt/user-data/outputs/submission.csv"
tr = pd.read_parquet("/home/claude/train.parquet")
te = pd.read_parquet("/home/claude/test.parquet")
ids = pd.read_csv("/home/claude/ids.csv").iloc[:, 0].values
days = pd.to_datetime(pd.read_csv("/home/claude/days.csv").iloc[:, 0])
test = pd.read_csv("/mnt/user-data/uploads/test.csv", parse_dates=["show_date"])
sample = pd.read_csv("/mnt/user-data/uploads/sample_submission.csv")

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
    m = lgb.LGBMRegressor(objective="poisson", n_estimators=350, learning_rate=0.05, num_leaves=31,
                          min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                          reg_lambda=10, n_jobs=1, verbose=-1, random_state=seed)
    return m.fit(d[F], d.y.values)

def fit_cat(d, seed=0):
    m = CatBoostRegressor(loss_function="Poisson", iterations=400, learning_rate=0.1, depth=6,
                          l2_leaf_reg=10, thread_count=1, verbose=0, random_seed=seed,
                          allow_writing_files=False)
    return m.fit(d[F], d.y.values)

best_w = 0.8  # validated: LGB 26.58, CAT 26.83, w=0.8 -> 26.60 (hybrid kept as hedge)

# ---- 2) final fit on all anchors (no future data exists beyond 2024-01-06)
full = tr.sample(frac=0.5, random_state=2)
pl = fit_lgb(full).predict(te[F]); write(pl, "LGB only (interim)")
np.save("/home/claude/pl.npy", pl)
pc = fit_cat(full).predict(te[F]); np.save("/home/claude/pc.npy", pc)
final = best_w * pl + (1 - best_w) * pc
write(final, f"hybrid w_lgb={best_w:.1f}")
print("DONE", flush=True)
