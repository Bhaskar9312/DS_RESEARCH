"""
train_ecgnextformer_v2.py  --  rare-age (few-shot) improvements for ECGNextFormer.

Drop-in companion to train_ecgnextformer.py (keep that file in the same folder;
this one imports its data loading, batching, prediction and metric functions so
the baseline is untouched).

What is new
-----------
--loss      mse | tamse | gai | bmc
              gai = Balanced MSE, GAI form (Ren et al., CVPR 2022): uses the
                    training-age prior (smoothed) inside a softmax over ages.
              bmc = Balanced MSE, batch-based form (no prior needed; use batch>=64).
--sampler   uniform | balanced     balanced = draw training ECGs with prob ~ count^-beta
--select    mse | balanced         checkpoint selection metric. 'balanced' = mean of
              per-shot-group validation MAE (many / median / few). Plain val MSE is
              dominated by many-shot ages and therefore prefers checkpoints that
              under-serve the tails.
--aug       light ECG augmentation (amplitude scale, noise, baseline wander).
              Recommended together with --sampler balanced (otherwise the same rare
              ECGs are replayed identically and the model memorises them).
Test-set predictions are saved (preds_*.csv) so results can be POOLED across folds
(see analyze_pooled.py). Per-fold few-shot sets are only ~10-20 ECGs: far too small
to compare methods on.

Examples
--------
  python train_ecgnextformer_v2.py --dataset_dir ... --loss gai  --select balanced --folds all --epochs 100 --batch_size 96 --run_name gai_bal
  python train_ecgnextformer_v2.py --dataset_dir ... --loss mse  --sampler balanced --aug --select balanced --folds all --epochs 100 --batch_size 96 --run_name samp_aug
  python train_ecgnextformer_v2.py --dataset_dir ... --loss mse  --select mse --folds all --epochs 100 --batch_size 96 --run_name baseline
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgnextformer import ECGNextFormer, count_parameters
from train_ecgnextformer import (  # reuse, do not duplicate
    load_ptbxl, make_synthetic, shot_group_of_ages, TailAwareWeights,
    batch_iterator, predict, age_metrics,
)

MAX_AGE = 120


# ----------------------------------------------------------------------------
def smooth_age_counts(train_ages, sigma=2.0, max_age=MAX_AGE):
    """Gaussian-smoothed per-integer-age counts from TRAINING ages only."""
    counts = np.bincount(np.asarray(train_ages).astype(int), minlength=max_age + 1).astype(float)
    radius = int(4 * sigma)
    off = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (off / sigma) ** 2)
    k /= k.sum()
    return np.maximum(np.convolve(counts, k, mode="same"), 1e-3)


# ----------------------------------------------------------------------------
# Balanced MSE (Ren et al. 2022)
# ----------------------------------------------------------------------------
class BalancedMSE(nn.Module):
    """
    gai: logits_j = -(pred - y_j)^2 / (2 s^2) + log p_train(y_j), CE against the true age bin.
    bmc: same, but the 'classes' are the targets in the current batch (no prior).
    Loss is multiplied by (2 s^2).detach() as in the reference implementation.
    Everything is in standardised-target units. s is learned (its own optimiser group).
    """

    def __init__(self, mode, age_mu, age_sd, train_ages, prior_sigma=2.0, init_sigma=0.5):
        super().__init__()
        assert mode in ("gai", "bmc")
        self.mode = mode
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(init_sigma))))
        if mode == "gai":
            sm = smooth_age_counts(train_ages, prior_sigma)
            prior = sm / sm.sum()
            self.register_buffer("log_prior", torch.log(torch.tensor(prior, dtype=torch.float32)))
            bins = (np.arange(MAX_AGE + 1) - age_mu) / age_sd
            self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32))

    def forward(self, pred, target_std, target_age):
        var2 = 2.0 * torch.exp(2.0 * self.log_sigma)
        if self.mode == "gai":
            logits = -(pred[:, None] - self.bins[None, :]) ** 2 / var2 + self.log_prior[None, :]
            labels = target_age.long().clamp(0, MAX_AGE)
        else:
            logits = -(pred[:, None] - target_std[None, :]) ** 2 / var2
            labels = torch.arange(len(pred), device=pred.device)
        return F.cross_entropy(logits, labels) * var2.detach()


# ----------------------------------------------------------------------------
# Augmentation (training batches only; xb is standardised, shape B,12,L)
# ----------------------------------------------------------------------------
def augment(xb, strength=1.0):
    B, C, L = xb.shape
    dev = xb.device
    xb = xb * (1.0 + 0.10 * strength * (torch.rand(B, 1, 1, device=dev) * 2 - 1))  # amplitude
    xb = xb + 0.03 * strength * torch.randn_like(xb)                                  # sensor noise
    t = torch.linspace(0, 1, L, device=dev)[None, None, :]
    cycles = 0.4 + 3.6 * torch.rand(B, 1, 1, device=dev)          # ~0.05-0.5 Hz over 8 s
    phase = 2 * np.pi * torch.rand(B, 1, 1, device=dev)
    lead_gain = 0.5 + 0.5 * torch.rand(B, C, 1, device=dev)
    return xb + 0.10 * strength * lead_gain * torch.sin(2 * np.pi * cycles * t + phase)  # wander


def balanced_mae(abs_err, groups):
    vals = [abs_err[groups == g].mean() for g in ("many", "median", "few") if (groups == g).any()]
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
def run_fold(args, meta, X_all, test_fold, device, out_dir):
    folds = meta["strat_fold"].to_numpy().astype(int)
    val_fold = (test_fold - 2) % 10 + 1
    tr_idx = np.where((folds != test_fold) & (folds != val_fold))[0]
    va_idx = np.where(folds == val_fold)[0]
    te_idx = np.where(folds == test_fold)[0]
    ages = meta["age"].to_numpy().astype(np.float32)

    if "patient_id" in meta.columns:
        pid = meta["patient_id"].to_numpy()
        shared = set(pid[tr_idx]) & (set(pid[va_idx]) | set(pid[te_idx]))
        assert not shared, f"patient overlap across splits: {len(shared)}"

    mu = X_all[tr_idx].reshape(-1, 12).mean(0)
    sd = X_all[tr_idx].reshape(-1, 12).std(0) + 1e-6
    X = torch.from_numpy(((X_all - mu) / sd).astype(np.float32))
    age_mu, age_sd = float(ages[tr_idx].mean()), float(ages[tr_idx].std())
    y = torch.from_numpy((ages - age_mu) / age_sd)
    y_years = torch.from_numpy(ages)

    count_src = ages if args.shot_counts == "all" else ages[tr_idx]
    vals, cnts = np.unique(count_src.astype(int), return_counts=True)
    counts_by_age = dict(zip(vals.tolist(), cnts.tolist()))
    groups_test = shot_group_of_ages(ages[te_idx], counts_by_age)
    groups_val = shot_group_of_ages(ages[va_idx], counts_by_age)

    # ---- loss
    weigher, crit = None, None
    if args.loss == "tamse":
        weigher = TailAwareWeights(ages[tr_idx], alpha=args.alpha, sigma=args.sigma, clip=args.clip)
    elif args.loss in ("gai", "bmc"):
        crit = BalancedMSE(args.loss, age_mu, age_sd, ages[tr_idx], args.sigma,
                           args.noise_sigma).to(device)
    if args.sampler == "balanced" and args.loss != "mse":
        print("WARNING: balanced sampler + a re-balancing loss double-corrects the imbalance. "
              "Use the sampler with --loss mse.")

    # ---- sampler probabilities (training ages only)
    p_samp = None
    if args.sampler == "balanced":
        sm = smooth_age_counts(ages[tr_idx], args.sigma)
        w = sm[ages[tr_idx].astype(int)] ** (-args.beta)
        p_samp = w / w.sum()

    torch.manual_seed(args.seed + test_fold)
    rng = np.random.default_rng(args.seed + test_fold)
    model = ECGNextFormer(args.variant, dropkey=args.dropkey).to(device)
    groups_opt = [{"params": model.parameters(), "lr": args.lr}]
    if crit is not None:
        groups_opt.append({"params": crit.parameters(), "lr": 1e-2})
    opt = torch.optim.Adam(groups_opt)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.1)
    print(f"\n=== fold {test_fold} | val {val_fold} | run={args.run_name} loss={args.loss} "
          f"sampler={args.sampler} select={args.select} aug={args.aug} | "
          f"train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} | "
          f"params {count_parameters(model):,} ===")

    best_score, best_state, hist = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = (rng.permutation(tr_idx) if p_samp is None
                else rng.choice(tr_idx, size=len(tr_idx), replace=True, p=p_samp))
        tot, nseen = 0.0, 0
        for xb, yb in batch_iterator(X, y, perm, args.batch_size, "random", rng):
            xb, yb = xb.to(device), yb.to(device)
            if args.aug:
                xb = augment(xb, args.aug_strength)
            pred = model(xb)
            yrs = torch.round(yb * age_sd + age_mu)
            if crit is not None:
                loss = crit(pred, yb, yrs)
            else:
                per = (pred - yb) ** 2
                if weigher is not None:
                    per = per * weigher.weights_for(yrs)
                loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            nseen += len(yb)
        sched.step()

        vp = (predict(model, X, va_idx, device, n_shots=1) * age_sd + age_mu).numpy()
        verr = vp - ages[va_idx]
        val_mse = float(np.mean(verr ** 2))
        val_bal = balanced_mae(np.abs(verr), groups_val)
        score = val_mse if args.select == "mse" else val_bal
        hist.append({"epoch": epoch, "train_loss": tot / nseen, "val_mse": val_mse,
                     "val_balanced_mae": val_bal})
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d} | train {tot / nseen:8.4f} | val MSE {val_mse:7.2f} "
                  f"| val balMAE {val_bal:6.2f} | best({args.select}) {best_score:7.2f}")

    model.load_state_dict(best_state)
    results, pred_rows = [], []
    for shots in (1, 10):
        tp = (predict(model, X, te_idx, device, n_shots=shots) * age_sd + age_mu).numpy()
        m = age_metrics(ages[te_idx], tp, groups_test)
        m.update({"fold": test_fold, "run": args.run_name, "loss": args.loss, "variant": args.variant,
                  "shots": shots, "best_score": best_score, "params": count_parameters(model)})
        results.append(m)
        print(f"test | {shots:2d}-shot | MSE all {m['MSE_all']:.2f} | many {m['MSE_many']:.2f} "
              f"| median {m['MSE_median']:.2f} | few {m['MSE_few']:.2f} | MAE {m['MAE_all']:.2f} "
              f"| MAE few {m['MAE_few']:.2f} (n_few={m['n_few']})")
        pred_rows.append(pd.DataFrame({
            "ecg_id": meta["ecg_id"].to_numpy()[te_idx], "fold": test_fold, "shots": shots,
            "age": ages[te_idx], "pred": tp, "group": groups_test}))

    tag = f"{args.variant}_{args.run_name}_fold{test_fold}"
    torch.save({"state_dict": best_state, "age_mu": age_mu, "age_sd": age_sd,
                "x_mu": mu, "x_sd": sd, "args": vars(args)}, out_dir / f"{tag}_best.pt")
    pd.DataFrame(hist).to_csv(out_dir / f"{tag}_history.csv", index=False)
    return results, pd.concat(pred_rows)


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--out_dir", type=str, default="./runs_v2")
    p.add_argument("--run_name", type=str, default=None, help="default: the loss name")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max_records", type=int, default=None)
    p.add_argument("--variant", default="lite", choices=["lite", "base", "plus", "max"])
    p.add_argument("--loss", default="mse", choices=["mse", "tamse", "gai", "bmc"])
    p.add_argument("--sampler", default="uniform", choices=["uniform", "balanced"])
    p.add_argument("--beta", type=float, default=0.5, help="sampler: prob ~ count^-beta (1 = fully balanced)")
    p.add_argument("--select", default="balanced", choices=["mse", "balanced"])
    p.add_argument("--aug", action="store_true")
    p.add_argument("--aug_strength", type=float, default=1.0)
    p.add_argument("--folds", default="10")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_step", type=int, default=40, help="LR /10 every N epochs (paper: 75 of 200)")
    p.add_argument("--dropkey", type=float, default=0.6)
    p.add_argument("--shot_counts", default="all", choices=["all", "train"])
    p.add_argument("--alpha", type=float, default=0.5, help="tamse: weight exponent")
    p.add_argument("--sigma", type=float, default=2.0, help="label smoothing width (years)")
    p.add_argument("--clip", type=float, default=5.0, help="tamse: max weight ratio")
    p.add_argument("--noise_sigma", type=float, default=0.5, help="gai/bmc: initial noise std (standardised units)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    a = p.parse_args()
    a.run_name = a.run_name or a.loss
    return a


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
    print(f"records: {len(meta)} | signals {sigs.shape}")

    folds = list(range(1, 11)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    all_res, all_pred = [], []
    for f in folds:
        r, pr = run_fold(args, meta, sigs, f, device, out_dir)
        all_res += r
        all_pred.append(pr)
        pd.DataFrame(all_res).to_csv(out_dir / f"results_{args.variant}_{args.run_name}.csv", index=False)
        pd.concat(all_pred).to_csv(out_dir / f"preds_{args.variant}_{args.run_name}.csv", index=False)

    df = pd.DataFrame(all_res)
    cols = [c for c in df.columns if c.startswith(("MSE", "MAE", "GM"))]
    print("\n=== Mean over folds ===")
    print(df.groupby("shots")[cols].mean().round(2).T)


if __name__ == "__main__":
    main()
