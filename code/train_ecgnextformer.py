"""
Training / evaluation pipeline for ECGNextFormer age estimation on PTB-XL.

Follows the paper's protocol (Sec. III-B/C):
  * 100 Hz signals, MSE loss, 200 epochs, batch 96, lr 1e-3, /10 every 75 epochs
  * 8-second random crops for training
  * best checkpoint chosen by validation MSE
  * 10 sliding 8-second windows averaged at inference ("multi-shot")
  * official PTB-XL strat_fold column; metrics MSE / MAE / GM on all, many-shot
    (>120 samples per age), median-shot (40-120) and few-shot (<40) subsets
  * 10-fold cross-validation (or any subset of folds)

Two loss options, everything else identical, so the comparison is fair:
  --loss mse     standard MSE (paper baseline)
  --loss tamse   Tail-Aware Weighted MSE (see TailAwareWeights below)

Usage examples
--------------
  # sanity check without any data (synthetic signals)
  python train_ecgnextformer.py --synthetic --epochs 3 --folds 10 --variant lite

  # one fold, baseline vs proposed, on your extracted PTB-XL folder
  python train_ecgnextformer.py --dataset_dir "C:/.../ptb-xl-...-1.0.1" --loss mse   --folds 10
  python train_ecgnextformer.py --dataset_dir "C:/.../ptb-xl-...-1.0.1" --loss tamse --folds 10

  # full 10-fold CV (final numbers)
  python train_ecgnextformer.py --dataset_dir ... --loss mse   --folds all
  python train_ecgnextformer.py --dataset_dir ... --loss tamse --folds all
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ecgnextformer import ECGNextFormer, count_parameters

SIG_LEN = 1000   # 10 s @ 100 Hz
CROP = 800       # 8 s
MAX_AGE_VALID = 100  # PTB-XL stores ages > 89 as 300 (anonymised); drop them


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_ptbxl(dataset_dir, cache_dir, max_records=None):
    """Return (metadata_df, signals[N,1000,12] float32). Cached as .npy."""
    import wfdb  # imported lazily so --synthetic works without it

    dataset_dir, cache_dir = Path(dataset_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(dataset_dir / "ptbxl_database.csv")
    meta = meta.dropna(subset=["age", "filename_lr", "strat_fold"])
    meta = meta[(meta["age"] > 0) & (meta["age"] <= MAX_AGE_VALID)].reset_index(drop=True)
    if max_records:
        meta = meta.iloc[:max_records].copy()

    sig_path = cache_dir / f"ptbxl_lr_signals_{len(meta)}.npy"
    ids_path = cache_dir / f"ptbxl_lr_ids_{len(meta)}.npy"
    if sig_path.exists() and ids_path.exists() and \
            (np.load(ids_path) == meta["ecg_id"].to_numpy()).all():
        print(f"Loading cached signals: {sig_path}")
        return meta, np.load(sig_path, mmap_mode=None)

    print(f"Reading {len(meta)} waveforms with wfdb (one-time, then cached)...")
    sigs = np.zeros((len(meta), SIG_LEN, 12), dtype=np.float32)
    t0 = time.time()
    for i, fn in enumerate(meta["filename_lr"]):
        s, f = wfdb.rdsamp(str(dataset_dir / fn))
        assert s.shape == (SIG_LEN, 12) and f["fs"] == 100, (fn, s.shape, f["fs"])
        sigs[i] = s
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(meta)}  ({time.time() - t0:.0f}s)")
    sigs = np.nan_to_num(sigs)
    np.save(sig_path, sigs)
    np.save(ids_path, meta["ecg_id"].to_numpy())
    return meta, sigs


def make_synthetic(n=3000, seed=0):
    """Fake ECG-like data whose waveform depends weakly on age (pipeline test)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(60, 17, n), 2, 95).round()
    t = np.arange(SIG_LEN) / 100.0
    sigs = np.zeros((n, SIG_LEN, 12), dtype=np.float32)
    for i in range(n):
        hr = 1.0 + 0.003 * (age[i] - 60) + 0.05 * rng.standard_normal()
        base = np.sin(2 * np.pi * hr * t) ** 15
        for l in range(12):
            sigs[i, :, l] = (0.5 + 0.02 * l) * base * (1 + 0.01 * (age[i] - 60) / 10) \
                            + 0.05 * rng.standard_normal(SIG_LEN)
    meta = pd.DataFrame({
        "ecg_id": np.arange(n), "age": age,
        "strat_fold": (np.arange(n) % 10) + 1,
    })
    return meta, sigs


# ----------------------------------------------------------------------------
# Shot groups (paper: few <40, median 40-120, many >120 instances per age)
# ----------------------------------------------------------------------------
def shot_group_of_ages(ages, counts_by_age):
    c = np.array([counts_by_age.get(int(a), 0) for a in ages])
    g = np.where(c < 40, "few", np.where(c <= 120, "median", "many"))
    return g


# ----------------------------------------------------------------------------
# Tail-Aware Weighted MSE
# ----------------------------------------------------------------------------
class TailAwareWeights:
    """Gaussian-smoothed inverse-frequency weights for training ages.

    The raw inverse-frequency weights are normalized to have mean 1 across
    training examples, then clipped to [1/clip, clip]. Because clipping can
    change the mean, the final weights are not guaranteed to have mean exactly
    1; this version prioritizes enforcing the requested bounds.

    Weights are derived from training-fold ages only, avoiding validation/test
    label leakage. This is a proposed tail-aware weighting method, not a claim
    that the original ECGNextFormer paper uses this loss.
    """

    def __init__(self, train_ages, alpha=0.5, sigma=2.0, clip=5.0, max_age=120):
        train_ages = np.asarray(train_ages).astype(int)
        if train_ages.size == 0:
            raise ValueError("train_ages must not be empty")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if clip < 1:
            raise ValueError("clip must be >= 1")
        if np.any(train_ages < 0) or np.any(train_ages > max_age):
            raise ValueError(f"Training ages must be in [0, {max_age}]")

        counts = np.bincount(train_ages, minlength=max_age + 1).astype(float)

        # Smooth counts across neighbouring ages with a Gaussian kernel.
        radius = int(4 * sigma)
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(counts, kernel, mode="same")
        smooth = np.maximum(smooth, 1e-3)

        weights = smooth ** (-alpha)

        # Normalize the average weight over training examples to 1 first.
        mean_weight = float(weights[train_ages].mean())
        weights = weights / max(mean_weight, 1e-12)

        # Clip AFTER normalization. Do not renormalize afterward, because that
        # could move values outside the specified clipping interval.
        weights = np.clip(weights, 1.0 / clip, clip)

        self.table = torch.tensor(weights, dtype=torch.float32)
        train_weights = weights[train_ages]
        print(
            "TAMSE weights | "
            f"train mean={train_weights.mean():.4f} | "
            f"train min={train_weights.min():.4f} | "
            f"train max={train_weights.max():.4f} | "
            f"configured bounds=[{1.0 / clip:.4f}, {clip:.4f}]"
        )

    def weights_for(self, ages_tensor):
        indices = ages_tensor.long().clamp(0, len(self.table) - 1)
        return self.table.to(ages_tensor.device)[indices]


# ----------------------------------------------------------------------------
# Batching (all data lives in RAM; crops done on the fly)
# ----------------------------------------------------------------------------
def batch_iterator(X, y, idx, batch_size, crop_mode, rng=None):
    """X: torch (N,1000,12). crop_mode: 'random' or 'center'."""
    ar = torch.arange(CROP)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        xb = X[b]  # (B,1000,12) copy
        if crop_mode == "random":
            starts = torch.from_numpy(rng.integers(0, SIG_LEN - CROP + 1, size=len(b)))
        else:
            starts = torch.full((len(b),), (SIG_LEN - CROP) // 2)
        gather_idx = (starts[:, None] + ar[None, :])[:, :, None].expand(-1, -1, 12)
        xb = torch.gather(xb, 1, gather_idx).transpose(1, 2).contiguous()  # B,12,800
        yield xb, y[b]


@torch.no_grad()
def predict(model, X, idx, device, n_shots=1, batch_size=256):
    """n_shots=1 -> centre crop; n_shots>1 -> evenly spaced sliding windows, averaged."""
    model.eval()
    ar = torch.arange(CROP)
    starts_list = ([(SIG_LEN - CROP) // 2] if n_shots == 1
                   else np.linspace(0, SIG_LEN - CROP, n_shots).round().astype(int).tolist())
    out = []
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        xb_full = X[b]
        acc = 0.0
        for st in starts_list:
            xb = xb_full[:, st:st + CROP, :].transpose(1, 2).contiguous().to(device)
            acc = acc + model(xb).float().cpu()
        out.append(acc / len(starts_list))
    return torch.cat(out)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def age_metrics(y_true, y_pred, groups):
    err = y_pred - y_true
    res = {}
    for name, mask in [("all", np.ones_like(err, dtype=bool)),
                       ("many", groups == "many"),
                       ("median", groups == "median"),
                       ("few", groups == "few")]:
        if mask.sum() == 0:
            res.update({f"MSE_{name}": np.nan, f"MAE_{name}": np.nan, f"GM_{name}": np.nan,
                        f"n_{name}": 0})
            continue
        e = err[mask]
        res[f"MSE_{name}"] = float(np.mean(e ** 2))
        res[f"MAE_{name}"] = float(np.mean(np.abs(e)))
        res[f"GM_{name}"] = float(np.exp(np.mean(np.log(np.abs(e) + 1e-6))))
        res[f"n_{name}"] = int(mask.sum())
    return res


# ----------------------------------------------------------------------------
# One fold
# ----------------------------------------------------------------------------
def run_fold(args, meta, X_all, test_fold, device, out_dir):
    folds = meta["strat_fold"].to_numpy().astype(int)
    val_fold = (test_fold - 2) % 10 + 1  # fold 10 -> val 9, fold 1 -> val 10
    tr_idx = np.where((folds != test_fold) & (folds != val_fold))[0]
    va_idx = np.where(folds == val_fold)[0]
    te_idx = np.where(folds == test_fold)[0]
    ages = meta["age"].to_numpy().astype(np.float32)

    # Patient-level leakage check (official folds should already guarantee it)
    if "patient_id" in meta.columns:
        pid = meta["patient_id"].to_numpy()
        shared = set(pid[tr_idx]) & (set(pid[va_idx]) | set(pid[te_idx]))
        assert not shared, f"patient overlap across splits: {len(shared)}"

    # Normalisation + target scaling from TRAINING data only
    mu = X_all[tr_idx].reshape(-1, 12).mean(0)
    sd = X_all[tr_idx].reshape(-1, 12).std(0) + 1e-6
    Xn = ((X_all - mu) / sd).astype(np.float32)
    X = torch.from_numpy(Xn)
    age_mu, age_sd = float(ages[tr_idx].mean()), float(ages[tr_idx].std())
    y = torch.from_numpy((ages - age_mu) / age_sd)  # standardised target
    y_years = torch.from_numpy(ages)

    # Shot groups: counts over the whole dataset (paper) or training split only
    count_src = ages if args.shot_counts == "all" else ages[tr_idx]
    vals, cnts = np.unique(count_src.astype(int), return_counts=True)
    counts_by_age = dict(zip(vals.tolist(), cnts.tolist()))
    groups_test = shot_group_of_ages(ages[te_idx], counts_by_age)
    groups_val = shot_group_of_ages(ages[va_idx], counts_by_age)

    weigher = None
    if args.loss == "tamse":
        weigher = TailAwareWeights(ages[tr_idx], alpha=args.alpha, sigma=args.sigma,
                                   clip=args.clip)

    torch.manual_seed(args.seed + test_fold)
    rng = np.random.default_rng(args.seed + test_fold)
    model = ECGNextFormer(args.variant, dropkey=args.dropkey).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.1)
    print(f"\n=== fold {test_fold} | val {val_fold} | loss={args.loss} | "
          f"train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} | "
          f"params {count_parameters(model):,} ===")

    best_val, best_state, hist = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        tot, nseen = 0.0, 0
        for xb, yb in batch_iterator(X, y, perm, args.batch_size, "random", rng):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            per = (pred - yb) ** 2
            if weigher is not None:
                yrs = yb * age_sd + age_mu
                per = per * weigher.weights_for(torch.round(yrs))
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            nseen += len(yb)
        sched.step()

        # validation (plain MSE in years, centre crop) -> checkpoint selection
        vp = predict(model, X, va_idx, device, n_shots=1) * age_sd + age_mu
        val_mse = float(((vp - y_years[va_idx]) ** 2).mean())
        hist.append({"epoch": epoch, "train_loss": tot / nseen, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d} | train {tot / nseen:8.4f} | val MSE {val_mse:7.2f} "
                  f"| best {best_val:7.2f}")

    model.load_state_dict(best_state)
    results = []
    for shots in (1, 10):
        tp = predict(model, X, te_idx, device, n_shots=shots) * age_sd + age_mu
        m = age_metrics(ages[te_idx], tp.numpy(), groups_test)
        m.update({"fold": test_fold, "loss": args.loss, "variant": args.variant,
                  "shots": shots, "best_val_mse": best_val,
                  "params": count_parameters(model)})
        results.append(m)
        print(f"test | {shots:2d}-shot | MSE all {m['MSE_all']:.2f} | many {m['MSE_many']:.2f} "
              f"| median {m['MSE_median']:.2f} | few {m['MSE_few']:.2f} | MAE {m['MAE_all']:.2f}")

    tag = f"{args.variant}_{args.loss}_fold{test_fold}"
    torch.save({"state_dict": best_state, "age_mu": age_mu, "age_sd": age_sd,
                "x_mu": mu, "x_sd": sd, "args": vars(args)}, out_dir / f"{tag}_best.pt")
    pd.DataFrame(hist).to_csv(out_dir / f"{tag}_history.csv", index=False)
    return results


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="folder containing ptbxl_database.csv and records100/")
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--synthetic", action="store_true", help="use fake data (pipeline test)")
    p.add_argument("--max_records", type=int, default=None, help="debug: limit records")
    p.add_argument("--variant", default="lite", choices=["lite", "base", "plus", "max"])
    p.add_argument("--loss", default="mse", choices=["mse", "tamse"])
    p.add_argument("--folds", default="10", help="e.g. 10 | 9,10 | all")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_step", type=int, default=75)
    p.add_argument("--dropkey", type=float, default=0.6)
    p.add_argument("--shot_counts", default="all", choices=["all", "train"],
                   help="count ages over whole dataset (paper) or the training split")
    p.add_argument("--alpha", type=float, default=0.5, help="tamse: weight exponent")
    p.add_argument("--sigma", type=float, default=2.0, help="tamse: kernel smoothing (years)")
    p.add_argument("--clip", type=float, default=5.0, help="tamse: max weight ratio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        meta, sigs = make_synthetic()
    else:
        assert args.dataset_dir, "--dataset_dir is required (or use --synthetic)"
        meta, sigs = load_ptbxl(args.dataset_dir, args.cache_dir, args.max_records)
    print(f"records: {len(meta)} | signals {sigs.shape} | "
          f"age range {meta['age'].min():.0f}-{meta['age'].max():.0f}")

    folds = list(range(1, 11)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    all_res = []
    for f in folds:
        all_res += run_fold(args, meta, sigs, f, device, out_dir)
        pd.DataFrame(all_res).to_csv(out_dir / f"results_{args.variant}_{args.loss}.csv",
                                     index=False)

    df = pd.DataFrame(all_res)
    cols = [c for c in df.columns if c.startswith(("MSE", "MAE", "GM"))]
    summary = df.groupby("shots")[cols].agg(["mean", "std"])
    summary.to_csv(out_dir / f"summary_{args.variant}_{args.loss}.csv")
    print("\n=== Mean over folds ===")
    print(df.groupby("shots")[cols].mean().round(2).T)


if __name__ == "__main__":
    main()
