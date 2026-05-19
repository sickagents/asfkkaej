"""
Feature engineering for ROGII Wellbore Geology Prediction.
Predict TVT (True Vertical Thickness) from well trajectory data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_well_data(data_dir: str, split: str = "train"):
    """Load all wells from train/ or test/ directory."""
    base = Path(data_dir) / split
    wells = []
    
    for well_dir in sorted(base.iterdir()):
        if not well_dir.is_dir():
            continue
        well_name = well_dir.name
        
        # Load all CSVs for this well
        for csv_file in well_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df["well_name"] = well_name
            df["source_file"] = csv_file.stem
            wells.append(df)
    
    return wells


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from well data."""
    df = df.copy()
    
    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    for skip in ["id", "TVT", "well_name", "source_file"]:
        if skip in num_cols:
            num_cols.remove(skip)
        if skip in cat_cols:
            cat_cols.remove(skip)
    
    # Top numeric by variance
    var = df[num_cols].var().sort_values(ascending=False)
    top = var.head(20).index.tolist()
    
    # --- 1. Depth-based features ---
    if "MD" in df.columns:
        df["MD_log"] = np.log1p(df["MD"])
        df["MD_bin"] = pd.cut(df["MD"], bins=20, labels=False)
    
    if "Z" in df.columns:
        df["Z_log"] = np.log1p(np.abs(df["Z"]))
    
    # --- 2. Spatial features ---
    if "X" in df.columns and "Y" in df.columns:
        df["XY_dist"] = np.sqrt(df["X"]**2 + df["Y"]**2)
        df["XY_angle"] = np.arctan2(df["Y"], df["X"])
    
    if "X" in df.columns and "Z" in df.columns:
        df["XZ_dist"] = np.sqrt(df["X"]**2 + df["Z"]**2)
    
    # --- 3. Interactions ---
    for i in range(min(len(top), 10)):
        for j in range(i + 1, min(len(top), 10)):
            c1, c2 = top[i], top[j]
            if df[c1].std() > 0 and df[c2].std() > 0:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                df[f"{c1}_d_{c2}"] = df[c1] / (df[c2] + 1e-8)
    
    # --- 4. Rolling features ---
    for n in top[:10]:
        for w in [3, 5, 10]:
            df[f"{n}_r{w}m"] = df[n].rolling(w, min_periods=1).mean()
            df[f"{n}_r{w}s"] = df[n].rolling(w, min_periods=1).std().fillna(0)
            df[f"{n}_d{w}"] = df[n].diff(w).fillna(0)
    
    # --- 5. Row-wise stats ---
    if len(top) >= 3:
        nd = df[top]
        df["_mean"] = nd.mean(axis=1)
        df["_std"] = nd.std(axis=1).fillna(0)
        df["_range"] = nd.max(axis=1) - nd.min(axis=1)
    
    # --- 6. Frequency encoding ---
    for c in cat_cols:
        freq = df[c].value_counts(normalize=True)
        df[f"{c}_freq"] = df[c].map(freq).fillna(0)
    
    # --- 7. Polynomials ---
    for n in top[:5]:
        df[f"{n}_sq"] = df[n] ** 2
        df[f"{n}_log"] = np.log1p(np.abs(df[n]))
    
    return df
