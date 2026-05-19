"""
Kaggle F1 Pit Stops - Distributed ML Pipeline
Auto-resumes across 1-hour sessions. Reads config from config.json.

Usage: python run.py
"""

import os
import sys
import json
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# LOAD CONFIG
# ============================================================
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

VPS_ID = CFG["vps_id"]
TOTAL_VPS = CFG["total_vps"]
COMPETITION = CFG["competition"]
TARGET = CFG["target"]
N_TRIALS = CFG["optuna_trials"]
CV_FOLDS = CFG["cv_folds"]
MAX_MINUTES = CFG["max_session_minutes"]
SEED = CFG["seed_base"] + VPS_ID * 1000

# Directories
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
CKPT_DIR = BASE / "checkpoints"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

for d in [DATA_DIR, CKPT_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================
# CHECKPOINT UTILS
# ============================================================
def state_path():
    return CKPT_DIR / f"vps{VPS_ID}_state.json"

def load_state():
    if state_path().exists():
        with open(state_path()) as f:
            return json.load(f)
    return {
        "vps_id": VPS_ID,
        "step": "init",
        "completed": [],
        "best_auc": 0,
        "best_params": {},
        "trials_done": 0,
        "started": datetime.now().isoformat(),
    }

def save_state(state):
    state["updated"] = datetime.now().isoformat()
    with open(state_path(), "w") as f:
        json.dump(state, f, indent=2)

def done(state, step):
    if step not in state["completed"]:
        state["completed"].append(step)
    state["step"] = step
    save_state(state)

def is_done(state, step):
    return step in state["completed"]

def save_pkl(name, obj):
    with open(CKPT_DIR / f"vps{VPS_ID}_{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def load_pkl(name):
    p = CKPT_DIR / f"vps{VPS_ID}_{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ============================================================
# STEP 1: INSTALL & DOWNLOAD
# ============================================================
def step_setup(state):
    print("\n[1/5] Setup & download data...")
    
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "kaggle", "lightgbm", "xgboost", "optuna", "pandas", "numpy",
        "scikit-learn", "scipy"
    ], check=True)
    
    if not (DATA_DIR / "train.csv").exists():
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", COMPETITION, "-p", str(DATA_DIR), "--force"
        ], check=True)
        import zipfile
        zip_path = DATA_DIR / f"{COMPETITION}.zip"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
    
    done(state, "setup")
    print("  Done.")


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
def step_features(state):
    print("\n[2/5] Feature engineering...")
    
    import pandas as pd
    import numpy as np
    
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    from features import build_features
    train_fe = build_features(train)
    test_fe = build_features(test)
    
    save_pkl("train_fe", train_fe)
    save_pkl("test_fe", test_fe)
    done(state, "features")
    print(f"  Train: {train.shape} -> {train_fe.shape}")
    print(f"  Test:  {test.shape} -> {test_fe.shape}")


# ============================================================
# STEP 3: PREPARE MATRICES
# ============================================================
def step_prepare(state):
    print("\n[3/5] Prepare matrices...")
    
    import pandas as pd
    import numpy as np
    
    train_fe = load_pkl("train_fe")
    test_fe = load_pkl("test_fe")
    
    drop = ["id", TARGET]
    feats = [c for c in train_fe.columns if c not in drop]
    common = [f for f in feats if f in test_fe.columns]
    
    X = train_fe[common].copy()
    y = train_fe[TARGET].copy()
    X_test = test_fe[common].copy()
    
    # Encode categoricals
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cats:
        X[c] = X[c].astype("category").cat.codes
        X_test[c] = X_test[c].astype("category").cat.codes
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(-999)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-999)
    
    save_pkl("X", X)
    save_pkl("y", y)
    save_pkl("X_test", X_test)
    save_pkl("features", common)
    done(state, "prepare")
    print(f"  Features: {len(common)}")


# ============================================================
# STEP 4: OPTUNA TUNING (main work)
# ============================================================
def step_tune(state, session_start):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import optuna
    import numpy as np
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    X = load_pkl("X")
    y = load_pkl("y")
    
    # Resume from checkpoint
    prev_trials = state.get("trials_done", 0)
    best_auc = state.get("best_auc", 0)
    best_params = state.get("best_params", {})
    
    print(f"\n[4/5] Optuna tuning (VPS {VPS_ID}, seed={SEED})...")
    if prev_trials > 0:
        print(f"  Resuming: {prev_trials} trials done, best AUC={best_auc:.5f}")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 512),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 200),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 100, 500),
            "n_jobs": -1,
            "random_state": SEED,
            "verbose": -1,
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X))
        
        for tr_idx, val_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        
        auc = roc_auc_score(y, oof)
        
        # Save checkpoint every 50 trials
        if trial.number % 50 == 0 and trial.number > 0:
            state["trials_done"] = prev_trials + trial.number
            state["best_auc"] = trial.study.best_value
            state["best_params"] = trial.study.best_params
            save_state(state)
        
        return auc
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    
    remaining_sec = MAX_MINUTES * 60 - (time.time() - session_start)
    n_trials = max(50, min(N_TRIALS, int(remaining_sec / 15)))
    
    print(f"  Running {n_trials} trials on {os.cpu_count()} cores...")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    state["trials_done"] = prev_trials + len(study.trials)
    state["best_auc"] = max(best_auc, study.best_value)
    state["best_params"] = study.best_params if study.best_value > best_auc else best_params
    
    save_pkl("study_best_params", state["best_params"])
    save_pkl("study_best_auc", state["best_auc"])
    done(state, "tune")
    
    print(f"  Best AUC: {state['best_auc']:.5f}")
    print(f"  Total trials: {state['trials_done']}")


# ============================================================
# STEP 5: FINAL MODEL & PREDICT
# ============================================================
def step_final(state):
    print("\n[5/5] Final model & predict...")
    
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    
    X = load_pkl("X")
    y = load_pkl("y")
    X_test = load_pkl("X_test")
    
    best_params = load_pkl("study_best_params")
    best_params.update({"n_jobs": -1, "random_state": SEED, "verbose": -1})
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / CV_FOLDS
        
        with open(MODEL_DIR / f"vps{VPS_ID}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
    
    final_auc = roc_auc_score(y, oof_preds)
    
    save_pkl("test_preds", test_preds)
    save_pkl("oof_preds", oof_preds)
    save_pkl("final_auc", final_auc)
    
    # Save submission CSV
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = test_preds
    sub.to_csv(RESULTS_DIR / f"vps{VPS_ID}_submission.csv", index=False)
    
    state["final_auc"] = final_auc
    done(state, "final")
    print(f"  Final {CV_FOLDS}-fold AUC: {final_auc:.5f}")
    print(f"  Submission: results/vps{VPS_ID}_submission.csv")


# ============================================================
# GIT SYNC
# ============================================================
def git_sync(message="auto-sync"):
    """Push results to GitHub."""
    os.chdir(BASE)
    subprocess.run(["git", "add", "checkpoints/", "results/"], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"[VPS {VPS_ID}] {message}"], capture_output=True)
    subprocess.run(["git", "push"], capture_output=True)
    print(f"  Synced to GitHub.")


# ============================================================
# MAIN
# ============================================================
def main():
    session_start = time.time()
    state = load_state()
    
    print("=" * 60)
    print(f"  VPS {VPS_ID}/{TOTAL_VPS} | CPU: {os.cpu_count()} cores")
    print(f"  Competition: {COMPETITION}")
    print(f"  Seed: {SEED}")
    print(f"  Resume from: {state['step']}")
    print("=" * 60)
    
    # Run pipeline
    if not is_done(state, "setup"):
        step_setup(state)
    
    if time.time() - session_start > MAX_MINUTES * 60:
        git_sync("timeout after setup"); return
    
    if not is_done(state, "features"):
        step_features(state)
    
    if time.time() - session_start > MAX_MINUTES * 60:
        git_sync("timeout after features"); return
    
    if not is_done(state, "prepare"):
        step_prepare(state)
    
    if time.time() - session_start > MAX_MINUTES * 60:
        git_sync("timeout after prepare"); return
    
    step_tune(state, session_start)  # always re-run (accumulates trials)
    
    if time.time() - session_start > MAX_MINUTES * 60:
        git_sync("timeout after tune"); return
    
    if not is_done(state, "final"):
        step_final(state)
    
    git_sync("pipeline complete")
    
    elapsed = (time.time() - session_start) / 60
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f} min")
    print(f"  Best AUC: {state.get('best_auc', 0):.5f}")
    print(f"  Final AUC: {state.get('final_auc', 0):.5f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
