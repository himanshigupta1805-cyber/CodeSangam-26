import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.simplefilter("ignore")
tr = pd.read_parquet("/home/claude/train.parquet")
days = pd.to_datetime(pd.read_csv("/home/claude/days.csv").iloc[:, 0])
va = int(np.where(days == "2023-11-11")[0][0])
drop = ["y", "anchor", "date_idx", "book_theater_id", "doy", "woy"]
F = [c for c in tr.columns if c not in drop]
fit = tr[(tr.date_idx <= va) & (tr.anchor < va)].sample(frac=0.5, random_state=1)
val = tr[tr.anchor == va]
y = val.y.values
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
base = lambda d: d.wd_mean_8w.fillna(d.roll_mean_28).fillna(d.all_mean).fillna(40).values
print("baseline", rmse(y, base(val)), flush=True)

def run(name, obj, target, inv, **kw):
    m = lgb.LGBMRegressor(objective=obj, n_estimators=300, learning_rate=0.06, num_leaves=31,
                          min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                          reg_lambda=10, n_jobs=1, verbose=-1, **kw)
    m.fit(fit[F], target(fit))
    p = inv(m.predict(val[F]), val)
    print(f"{name}: rmse {rmse(y, p):.3f}", flush=True)
    return p

run("L2 raw", "regression", lambda d: d.y.values, lambda p, d: np.clip(p, 1, None))
run("poisson", "poisson", lambda d: d.y.values, lambda p, d: p)
run("ratio-to-baseline L2", "regression", lambda d: d.y.values / base(d),
    lambda p, d: np.clip(p, 0.05, None) * base(d))
run("log L2", "regression", lambda d: np.log1p(d.y.values), lambda p, d: np.expm1(p))
