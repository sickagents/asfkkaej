"""
Feature engineering for F1 Pit Stops prediction.
Shared across all VPS - edit this to add custom features.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    for skip in ["id", "PitNextLap"]:
        if skip in num_cols:
            num_cols.remove(skip)
    
    # Top numeric by variance (avoid feature explosion)
    var = df[num_cols].var().sort_values(ascending=False)
    top = var.head(15).index.tolist()
    
    # --- 1. Interactions (top pairs) ---
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, c2 = top[i], top[j]
            if df[c1].std() > 0 and df[c2].std() > 0:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                df[f"{c1}_d_{c2}"] = df[c1] / (df[c2] + 1e-8)
    
    # --- 2. Group aggregations ---
    groups = [c for c in cat_cols if c != "id"]
    for g in groups:
        for n in top[:10]:
            grp = df.groupby(g)[n]
            df[f"{n}_gmean_{g}"] = grp.transform("mean")
            df[f"{n}_gstd_{g}"] = grp.transform("std").fillna(0)
            df[f"{n}_grank_{g}"] = grp.rank(pct=True)
    
    # --- 3. Rolling features (if ordered) ---
    for sort_col in ["LapNumber", "Lap", "lap"]:
        if sort_col in df.columns:
            df = df.sort_values(sort_col).reset_index(drop=True)
            for n in top[:8]:
                for w in [3, 5, 10]:
                    df[f"{n}_r{w}m"] = df[n].rolling(w, min_periods=1).mean()
                    df[f"{n}_r{w}s"] = df[n].rolling(w, min_periods=1).std().fillna(0)
                    df[f"{n}_d{w}"] = df[n].diff(w).fillna(0)
            break
    
    # --- 4. Binning ---
    for n in top:
        if df[n].nunique() > 20:
            df[f"{n}_q10"] = pd.qcut(df[n], q=10, labels=False, duplicates="drop")
    
    # --- 5. Row-wise stats ---
    if len(top) >= 3:
        nd = df[top]
        df["_mean"] = nd.mean(axis=1)
        df["_std"] = nd.std(axis=1).fillna(0)
        df["_range"] = nd.max(axis=1) - nd.min(axis=1)
        df["_skew"] = nd.skew(axis=1)
    
    # --- 6. Frequency encoding ---
    for c in cat_cols:
        freq = df[c].value_counts(normalize=True)
        df[f"{c}_freq"] = df[c].map(freq).fillna(0)
    
    # --- 7. Polynomials (top 5) ---
    for n in top[:5]:
        df[f"{n}_sq"] = df[n] ** 2
        df[f"{n}_log"] = np.log1p(np.abs(df[n]))
    
    return df
