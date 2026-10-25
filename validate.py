import numpy as np, pandas as pd, lightgbm as lgb, warnings
from catboost import CatBoostRegressor
warnings.simplefilter("ignore")

tr = pd.read_parquet("/home/claude/train.parquet")
days = pd.read_csv("/home/claude/days.csv").iloc[:, 0]
days = pd.to_datetime(days)
val_anchor = int(np.where(days == "2023-11-11")[0][0])

drop = ["y", "anchor", "date_idx", "book_theater_id"]
def feats(use_doy):
    f = [c for c in tr.columns if c not in drop]
    if not use_doy:
        f = [c for c in f if c not in ("doy", "woy")]
    return f

fit = tr[(tr.date_idx <= val_anchor) & (tr.anchor < val_anchor)]
val = tr[tr.anchor == val_anchor]
print("fit", fit.shape, "val", val.shape)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def rmsle(a, b): return rmse(np.log1p(a), np.log1p(b))

res = {}
for use_doy in (False, True):
    F = feats(use_doy)
    ytr = np.log1p(fit.y.values)
    m1 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, n_jobs=1, verbose=-1)
    m1.fit(fit[F], ytr)
    p1 = np.expm1(m1.predict(val[F]))
    sub = fit.sample(frac=0.4, random_state=0)
    m2 = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=7, l2_leaf_reg=5, thread_count=1, verbose=0)
    m2.fit(sub[F], np.log1p(sub.y.values))
    p2 = np.expm1(m2.predict(val[F]))
    res[use_doy] = (p1, p2)
    y = val.y.values
    print(f"doy={use_doy} LGB rmse {rmse(y,p1):.3f} rmsle {rmsle(y,p1):.4f} | CAT rmse {rmse(y,p2):.3f} rmsle {rmsle(y,p2):.4f}")
    for w in (0.3, 0.5, 0.7):
        p = w * p1 + (1 - w) * p2
        print(f"   blend w_lgb={w}: rmse {rmse(y,p):.3f} rmsle {rmsle(y,p):.4f}")

# baselines
y = val.y.values
print("baseline wd_mean_8w", rmse(y, val.wd_mean_8w.fillna(val.roll_mean_28).fillna(40).values))
