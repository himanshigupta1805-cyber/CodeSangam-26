"""Clean merge + leak-free feature builder.

Forecast design (direct multi-horizon, anchor based)
----------------------------------------------------
The test window (2024-01-07 .. 2024-02-28) is up to 53 days beyond the last
known day (2024-01-06). Plain lag_7/14/28 columns would be unknown for most test
rows, so every training sample is built exactly the way the test set is:

  * pick an anchor (cutoff) day c; only data with date <= c is visible
  * target day t = c + h, h in 1..53
  * lag_7 / lag_14 / lag_28 are the 7/14/28-day lags *shifted back by whole weeks
    until they are <= c*, i.e. y[t - 7k], y[t - 7k - 7], y[t - 7k - 21] with
    k = ceil(h/7).  For h <= 7 these are exactly the classic 7/14/28 day lags.
  * rolling mean / std windows (7, 14, 28, 56) end at the anchor c.

Anchors are all Saturdays (test anchor 2024-01-06 is a Saturday), so the
horizon <-> weekday relation is identical in train and test.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
H = 53


def load_clean():
    vis = pd.read_csv(RAW_DIR / "booknow_visits.csv", parse_dates=["show_date"])
    # 2023-02-28 is double-logged -> average duplicates (not sum)
    vis = vis.groupby(["book_theater_id", "show_date"], as_index=False).audience_count.mean()

    test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["show_date"])

    # theater metadata: drop rows with missing id, dedupe
    bt = pd.read_csv(RAW_DIR / "booknow_theaters.csv").dropna(subset=["book_theater_id"])
    bt = bt.drop_duplicates("book_theater_id")

    rel = pd.read_csv(RAW_DIR / "movie_theater_id_relation.csv").drop_duplicates()
    ct = pd.read_csv(RAW_DIR / "cinePOS_theaters.csv").drop_duplicates("cine_theater_id")

    bb = pd.read_csv(RAW_DIR / "booknow_booking.csv", parse_dates=["show_datetime", "booking_datetime"])
    cb = pd.read_csv(RAW_DIR / "cinePOS_booking.csv", parse_dates=["show_datetime", "booking_datetime"])
    # a handful of rows are booked after the show -> drop (can't be known pre-show)
    bb = bb[bb.booking_datetime <= bb.show_datetime]
    cb = cb[cb.booking_datetime <= cb.show_datetime]
    bb = bb.assign(show_date=bb.show_datetime.dt.normalize())
    cb = cb.assign(show_date=cb.show_datetime.dt.normalize())

    # BookNow daily booking aggregates per theater
    b_day = bb.groupby(["book_theater_id", "show_date"]).agg(
        bn_tix=("tickets_booked", "sum"), bn_n=("tickets_booked", "size")).reset_index()

    # CinePOS -> BookNow id via mapping key, then daily aggregates
    cb = cb.merge(rel, on="cine_theater_id", how="inner")  # only mapped cine theaters
    c_day = cb.groupby(["book_theater_id", "show_date"]).agg(
        cp_tix=("tickets_sold", "sum"), cp_n=("tickets_sold", "size")).reset_index()

    return vis, test, bt, rel, ct, b_day, c_day


def to_matrix(df, col, ids, days):
    m = df.pivot_table(index="book_theater_id", columns="show_date", values=col, aggfunc="sum")
    return m.reindex(index=ids, columns=days).values.astype(np.float32)


def nanstat(a, fn):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(a, axis=1)


def build_all():
    vis, test, bt, rel, ct, b_day, c_day = load_clean()
    ids = sorted(set(vis.book_theater_id) | set(test.book_theater_id))
    last = vis.show_date.max()                                  # 2024-01-06
    days = pd.date_range("2023-01-01", test.show_date.max())    # through 2024-02-28
    di = {d: i for i, d in enumerate(days)}
    Y = to_matrix(vis, "audience_count", ids, days)
    BT = to_matrix(b_day, "bn_tix", ids, days)
    CT = to_matrix(c_day, "cp_tix", ids, days)
    CN = to_matrix(c_day, "cp_n", ids, days)
    mapped = np.isin(ids, rel.book_theater_id.values)

    meta = pd.DataFrame({"book_theater_id": ids}).merge(bt, on="book_theater_id", how="left")
    for c in ["theater_type", "theater_area"]:
        meta[c] = meta[c].fillna("NA").astype("category").cat.codes
    n_ids = len(ids)
    dow_all = np.array(days.dayofweek)

    def anchor_feats(c):
        """features for every theater at anchor index c using only data <= c."""
        F = {}
        for w in (7, 14, 28, 56):
            win = Y[:, max(0, c - w + 1): c + 1]
            F[f"roll_mean_{w}"] = nanstat(win, np.nanmean)
            F[f"roll_std_{w}"] = nanstat(win, np.nanstd)
            F[f"roll_cnt_{w}"] = np.isfinite(win).sum(1)
            F[f"roll_max_{w}"] = nanstat(win, np.nanmax)
        win = Y[:, : c + 1]
        F["all_mean"] = nanstat(win, np.nanmean)
        F["all_std"] = nanstat(win, np.nanstd)
        F["all_median"] = nanstat(win, np.nanmedian)
        F["last_obs"] = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in win])
        # BookNow ledger (bookings) & CinePOS ledger (mapped theaters only)
        for w in (7, 28):
            F[f"bn_tix_{w}"] = nanstat(BT[:, max(0, c - w + 1): c + 1], np.nanmean)
            F[f"cp_tix_{w}"] = nanstat(CT[:, max(0, c - w + 1): c + 1], np.nanmean)
        F["cp_n_28"] = nanstat(CN[:, max(0, c - 27): c + 1], np.nanmean)
        F["cp_to_aud_28"] = F["cp_tix_28"] / (F["roll_mean_28"] + 1)
        F["bn_to_aud_28"] = F["bn_tix_28"] / (F["roll_mean_28"] + 1)
        # per-weekday means over last 8 weeks (index by weekday 0..6)
        wd = np.full((n_ids, 7), np.nan, np.float32)
        wd_std = np.full((n_ids, 7), np.nan, np.float32)
        for d in range(7):
            idx = [j for j in range(max(0, c - 55), c + 1) if dow_all[j] == d]
            wd[:, d] = nanstat(Y[:, idx], np.nanmean)
            wd_std[:, d] = nanstat(Y[:, idx], np.nanstd)
        return F, wd, wd_std

    def make_rows(c, hs):
        F, wd, wd_std = anchor_feats(c)
        rows = []
        for h in hs:
            t = c + h
            k = int(np.ceil(h / 7))
            tdow = dow_all[t]
            def lag(sh):
                j = t - sh
                return Y[:, j] if 0 <= j <= c else np.full(n_ids, np.nan, np.float32)
            def lagc(M, sh):
                j = t - sh
                return M[:, j] if 0 <= j <= c else np.full(n_ids, np.nan, np.float32)
            r = {k_: v for k_, v in F.items()}
            r["book_theater_id"] = np.arange(n_ids)
            r["h"] = np.full(n_ids, h)
            r["date_idx"] = np.full(n_ids, t)
            r["lag_7"] = lag(7 * k)
            r["lag_14"] = lag(7 * k + 7)
            r["lag_28"] = lag(7 * k + 21)
            r["lag_35"] = lag(7 * k + 28)
            r["cp_lag_7"] = lagc(CT, 7 * k)
            r["bn_lag_7"] = lagc(BT, 7 * k)
            r["wd_mean_8w"] = wd[:, tdow]
            r["wd_std_8w"] = wd_std[:, tdow]
            r["lag_mean3"] = np.nanmean(np.stack([r["lag_7"], r["lag_14"], r["lag_28"]]), 0) \
                if True else None
            r["dow"] = np.full(n_ids, tdow)
            r["dom"] = np.full(n_ids, days[t].day)
            r["month"] = np.full(n_ids, days[t].month)
            r["woy"] = np.full(n_ids, days[t].isocalendar()[1])
            r["doy"] = np.full(n_ids, days[t].dayofyear)
            r["y"] = Y[:, t] if t <= di[last] else np.full(n_ids, np.nan, np.float32)
            rows.append(pd.DataFrame(r))
        df = pd.concat(rows, ignore_index=True)
        df["anchor"] = c
        return df

    import warnings
    warnings.simplefilter("ignore")
    # Saturday anchors
    sat = [i for i, d in enumerate(days) if d.dayofweek == 5 and d <= last]
    train_parts = []
    for c in sat:
        if c < 6:
            continue
        hs = [h for h in range(1, H + 1) if c + h <= di[last]]
        if not hs:
            continue
        p = make_rows(c, hs)
        p = p[p.y.notna()]
        train_parts.append(p)
    train = pd.concat(train_parts, ignore_index=True)

    ca = di[last]
    test_df = make_rows(ca, list(range(1, (days[-1] - last).days + 1)))
    meta_cols = meta.drop(columns="book_theater_id").assign(tid=np.arange(n_ids))
    for df in (train, test_df):
        m = df.merge(meta_cols, left_on="book_theater_id", right_on="tid", how="left").drop(columns="tid")
        df[m.columns] = m
    return train, test_df, ids, days, mapped


if __name__ == "__main__":
    tr, te, ids, days, mapped = build_all()
    print(tr.shape, te.shape)
    tr.to_parquet(PROCESSED_DIR / "train.parquet")
    te.to_parquet(PROCESSED_DIR / "test.parquet")
    pd.Series(ids).to_csv(PROCESSED_DIR / "ids.csv", index=False)
    pd.Series(days).to_csv(PROCESSED_DIR / "days.csv", index=False)
