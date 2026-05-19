"""
Ensemble script - run on master (VPS 1) after all VPS complete.
Pulls results from GitHub, combines predictions, submits.

Usage: python ensemble.py
"""

import os
import json
import pickle
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

BASE = Path(__file__).parent
CKPT = BASE / "checkpoints"
RESULTS = BASE / "results"
DATA = BASE / "data"

with open(BASE / "config.json") as f:
    CFG = json.load(f)

TARGET = CFG["target"]
N_VPS = CFG["total_vps"]


def load_vps(vps_id):
    """Load one VPS results."""
    r = {"id": vps_id}
    
    # State
    sp = CKPT / f"vps{vps_id}_state.json"
    if sp.exists():
        with open(sp) as f:
            r["state"] = json.load(f)
    else:
        return None
    
    # Predictions
    for name in ["test_preds", "oof_preds", "final_auc"]:
        p = CKPT / f"vps{vps_id}_{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                r[name] = pickle.load(f)
    
    r["auc"] = r.get("final_auc", 0) or r.get("state", {}).get("final_auc", 0)
    r["trials"] = r.get("state", {}).get("trials_done", 0)
    
    return r


def optimize_weights(oof_list, y):
    """Find optimal ensemble weights."""
    n = len(oof_list)
    
    def neg_auc(w):
        w = np.abs(w) / np.sum(np.abs(w))
        blend = sum(wi * oi for wi, oi in zip(w, oof_list))
        return -roc_auc_score(y, blend)
    
    res = minimize(neg_auc, np.ones(n) / n, method="Nelder-Mead",
                   options={"maxiter": 1000, "xatol": 1e-8})
    w = np.abs(res.x) / np.sum(np.abs(res.x))
    return w


def main():
    print("=" * 60)
    print("  ENSEMBLE - Combining all VPS results")
    print("=" * 60)
    
    # Pull latest from GitHub
    print("\nPulling latest from GitHub...")
    subprocess.run(["git", "pull"], capture_output=True)
    
    # Load results
    vps_list = []
    for vid in range(1, N_VPS + 1):
        r = load_vps(vid)
        if r and "test_preds" in r and "oof_preds" in r:
            vps_list.append(r)
            print(f"  VPS {vid}: AUC={r['auc']:.5f}, Trials={r['trials']}")
        else:
            print(f"  VPS {vid}: incomplete (skipped)")
    
    if len(vps_list) < 2:
        print(f"\nERROR: Need at least 2 complete VPS, got {len(vps_list)}")
        return
    
    # Ground truth
    # Load ground truth - ROGII data is in directories
    from pathlib import Path
    train_dfs = []
    for well_dir in sorted((DATA / "train").iterdir()):
        if not well_dir.is_dir():
            continue
        for csv_file in well_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            train_dfs.append(df)
    train = pd.concat(train_dfs, ignore_index=True)
    y = train[TARGET].values
    
    oof_list = [r["oof_preds"] for r in vps_list]
    test_list = [r["test_preds"] for r in vps_list]
    
    # Method 1: Simple average
    avg_oof = np.mean(oof_list, axis=0)
    avg_test = np.mean(test_list, axis=0)
    avg_auc = roc_auc_score(y, avg_oof)
    
    # Method 2: AUC-weighted
    aucs = [r["auc"] for r in vps_list]
    wa = np.array(aucs) / sum(aucs)
    wa_oof = sum(w * o for w, o in zip(wa, oof_list))
    wa_test = sum(w * t for w, t in zip(wa, test_list))
    wa_auc = roc_auc_score(y, wa_oof)
    
    # Method 3: Optimized
    opt_w = optimize_weights(oof_list, y)
    opt_oof = sum(w * o for w, o in zip(opt_w, oof_list))
    opt_test = sum(w * t for w, t in zip(opt_w, test_list))
    opt_auc = roc_auc_score(y, opt_oof)
    
    # Pick best
    methods = {
        "simple_avg": (avg_auc, avg_test),
        "auc_weighted": (wa_auc, wa_test),
        "optimized": (opt_auc, opt_test),
    }
    best_name = max(methods, key=lambda k: methods[k][0])
    best_auc, best_test = methods[best_name]
    
    print(f"\nResults:")
    print(f"  Simple Average:  {avg_auc:.5f}")
    print(f"  AUC-Weighted:    {wa_auc:.5f}")
    print(f"  Optimized:       {opt_auc:.5f}")
    print(f"  Best: {best_name} ({best_auc:.5f})")
    
    print(f"\nWeights:")
    for i, r in enumerate(vps_list):
        print(f"  VPS {r['id']}: opt={opt_w[i]:.3f}, auc_weight={wa[i]:.3f}")
    
    # Save
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = best_test
    out = RESULTS / "final_ensemble.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    
    # Push to GitHub
    subprocess.run(["git", "add", "results/"], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"ensemble {best_name} AUC={best_auc:.5f}"],
                   capture_output=True)
    subprocess.run(["git", "push"], capture_output=True)
    
    # Submit
    answer = input(f"\nSubmit to Kaggle (AUC={best_auc:.5f})? [y/N]: ")
    if answer.lower() == "y":
        subprocess.run([
            "kaggle", "competitions", "submit",
            "-c", CFG["competition"],
            "-f", str(out),
            "-m", f"5xVPS ensemble {best_name} AUC={best_auc:.5f}"
        ])
        subprocess.run(["kaggle", "competitions", "submissions", "-c", CFG["competition"]])


if __name__ == "__main__":
    main()
