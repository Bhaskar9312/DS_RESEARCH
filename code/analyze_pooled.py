"""
analyze_pooled.py -- compare runs on POOLED out-of-fold predictions.

Per-fold few-shot sets contain only ~10-20 ECGs, so per-fold few-shot numbers are
noise-dominated. Pooling all folds gives ~140 few-shot ECGs, and a paired bootstrap
over ECGs gives a confidence interval for the difference.

usage:
  python analyze_pooled.py --baseline runs_v2/preds_lite_baseline.csv \
         --proposed runs_v2/preds_lite_gai_bal.csv runs_v2/preds_lite_samp_aug.csv --shots 10
Negative difference = proposed is better (lower error).
Only ECGs present in BOTH files (same folds) are compared.
"""
import argparse
import numpy as np
import pandas as pd


def metrics(e):
    a = np.abs(e)
    return {"MSE": np.mean(e ** 2), "MAE": a.mean(), "GM": np.exp(np.mean(np.log(a + 1e-6)))}


def boot_diff(eb, ep, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eb)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for name, fn in [("MSE", lambda e: np.mean(e ** 2, axis=1)),
                     ("MAE", lambda e: np.mean(np.abs(e), axis=1)),
                     ("GM", lambda e: np.exp(np.mean(np.log(np.abs(e) + 1e-6), axis=1)))]:
        d = fn(ep[idx]) - fn(eb[idx])
        out[name] = (np.percentile(d, 2.5), np.percentile(d, 97.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--proposed", nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=10)
    a = ap.parse_args()

    base = pd.read_csv(a.baseline)
    base = base[base["shots"] == a.shots].set_index("ecg_id")
    for path in a.proposed:
        prop = pd.read_csv(path)
        prop = prop[prop["shots"] == a.shots].set_index("ecg_id")
        ids = base.index.intersection(prop.index)
        b, p = base.loc[ids], prop.loc[ids]
        eb, ep = (b["pred"] - b["age"]).to_numpy(), (p["pred"] - p["age"]).to_numpy()
        grp = b["group"].to_numpy()
        print(f"\n##### {path}  vs  {a.baseline}   ({len(ids)} ECGs, shots={a.shots})")
        rows = []
        for g in ["all", "many", "median", "few"]:
            m = np.ones(len(ids), bool) if g == "all" else grp == g
            if m.sum() < 2:
                continue
            mb, mp = metrics(eb[m]), metrics(ep[m])
            ci = boot_diff(eb[m], ep[m])
            for k in ("MSE", "MAE", "GM"):
                lo, hi = ci[k]
                sig = "yes" if (lo > 0 or hi < 0) else "no"
                rows.append({"group": g, "n": int(m.sum()), "metric": k,
                             "baseline": round(mb[k], 2), "proposed": round(mp[k], 2),
                             "diff(prop-base)": round(mp[k] - mb[k], 2),
                             "95% CI": f"[{lo:.2f}, {hi:.2f}]", "CI excludes 0": sig})
        print(pd.DataFrame(rows).to_string(index=False))

        # signed bias by age band: regression-to-the-mean check
        bins = [0, 18, 40, 60, 80, 101]
        lab = ["<18", "18-39", "40-59", "60-79", "80+"]
        band = pd.cut(b["age"], bins=bins, labels=lab, right=False)
        bias = pd.DataFrame({"band": band.to_numpy(), "base_bias": eb, "prop_bias": ep,
                             "base_MAE": np.abs(eb), "prop_MAE": np.abs(ep)})
        print("\nMean signed error (pred-true, years) and MAE by age band:")
        print(bias.groupby("band", observed=True).agg(n=("base_bias", "size"),
              base_bias=("base_bias", "mean"), prop_bias=("prop_bias", "mean"),
              base_MAE=("base_MAE", "mean"), prop_MAE=("prop_MAE", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
