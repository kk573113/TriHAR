# -*- coding: utf-8 -*-
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import wfdb

from config import (
    BATCH_SIZE, CAT_COLS, CXR_BASE, CXR_CSV_PATH, ECG_BASE, ECG_CSV_PATH,
    MAX_ECG_LEN, N_CLASSES, NUM_WORKERS, TRIMODAL_CSV_PATH,
)

class TrimodalHFSDataset(Dataset):
    def __init__(self, x_ecg: np.ndarray, cxr_paths: np.ndarray, x_tab: np.ndarray, y: np.ndarray):
        # ECG: input numpy shape (N, L, 12), output torch shape (N, 12, L)
        self.x_ecg = torch.FloatTensor(x_ecg).permute(0, 2, 1)
        self.cxr_paths = np.asarray(cxr_paths)
        self.x_tab = torch.FloatTensor(x_tab)
        self.y = torch.LongTensor(y)

        self.cxr_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.cxr_paths[idx]).convert("RGB")
        img = self.cxr_transform(img)
        return self.x_ecg[idx], img, self.x_tab[idx], self.y[idx]


def get_loader(x_ecg, cxr_paths, x_tab, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TrimodalHFSDataset(x_ecg, cxr_paths, x_tab, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================

# ============================================================
def load_or_merge_trimodal_csv() -> pd.DataFrame:
    if os.path.exists(TRIMODAL_CSV_PATH):
        print(f"Loading trimodal CSV: {TRIMODAL_CSV_PATH}")
        return pd.read_csv(TRIMODAL_CSV_PATH)

    print("Trimodal CSV not found. Merging ECG and CXR CSV files.")
    print(f"ECG CSV: {ECG_CSV_PATH}")
    print(f"CXR CSV: {CXR_CSV_PATH}")

    df_ecg = pd.read_csv(ECG_CSV_PATH)
    df_cxr = pd.read_csv(CXR_CSV_PATH)

    candidate_keys = ["subject_id", "hadm_id", "stay_id"]
    keys = [k for k in candidate_keys if k in df_ecg.columns and k in df_cxr.columns]
    if len(keys) == 0:
        raise ValueError("No common merge keys found among subject_id, hadm_id, stay_id.")

    if "cxr_path" not in df_cxr.columns:
        raise ValueError("CXR CSV must contain cxr_path.")
    if "ecg_path" not in df_ecg.columns:
        raise ValueError("ECG CSV must contain ecg_path.")

    # Keep clinical columns from ECG CSV and only add cxr_path from CXR CSV.
    cxr_keep = keys + ["cxr_path"]
    df = df_ecg.merge(df_cxr[cxr_keep].drop_duplicates(), on=keys, how="inner")
    print(f"Merged rows: {len(df)}")
    return df


def resolve_cxr_path(rel_path: str) -> str:
    # Existing CXR script used replace('.0/', '/') because of CSV path formatting.
    rel_path = str(rel_path).replace(".0/", "/")
    return os.path.join(CXR_BASE, rel_path)


def resolve_ecg_path(rel_path: str) -> str:
    return os.path.join(ECG_BASE, str(rel_path))


def prepare_data():
    df = load_or_merge_trimodal_csv()

    df["icu_los_class"] = df["icu_los_days"].apply(lambda x: 1 if x > 3 else 0)
    target_col = "icu_los_class"

    ignore_cols = [
        "subject_id", "hadm_id", "stay_id",
        "intime", "outtime", "admittime", "dischtime",
        "ecg_path", "cxr_path",
        "mortality", "icu_mortality", "hosp_mortality", "readmission_30d",
        "sapsii", "icu_los_days", "icu_los_class",
        "anchor_year", "hosp_los_days", "AP", "ViewPosition", "cxr_time"
    ]

    available_cols = [c for c in df.columns if c not in ignore_cols + [target_col]]
    cat_features = [c for c in CAT_COLS if c in available_cols]
    num_features = [c for c in available_cols if c not in cat_features]

    print(f"Numeric features: {len(num_features)}")
    print(f"Categorical features: {cat_features}")

    cat_dims = []
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_dims.append(len(le.classes_))
    cat_dims = tuple(cat_dims)

    X_num = df[num_features].values.astype(np.float32) if len(num_features) > 0 else np.empty((len(df), 0), dtype=np.float32)
    X_cat = df[cat_features].values.astype(np.float32) if len(cat_features) > 0 else np.empty((len(df), 0), dtype=np.float32)
    X_tab_all = np.hstack([X_num, X_cat]).astype(np.float32)

    x_ecg_all, cxr_paths, valid_indices = [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading ECG + checking CXR"):
        try:
            ecg_path = resolve_ecg_path(row["ecg_path"])
            cxr_path = resolve_cxr_path(row["cxr_path"])

            if not os.path.exists(ecg_path + ".dat"):
                continue
            if not os.path.exists(cxr_path):
                continue

            sig, _ = wfdb.rdsamp(ecg_path)
            if len(sig) < MAX_ECG_LEN:
                sig = np.pad(sig, ((0, MAX_ECG_LEN - len(sig)), (0, 0)), "constant")
            else:
                sig = sig[:MAX_ECG_LEN]

            if np.isnan(sig).any():
                continue

            x_ecg_all.append(sig.astype(np.float32))
            cxr_paths.append(cxr_path)
            valid_indices.append(idx)
        except Exception:
            continue

    x_ecg_all = np.asarray(x_ecg_all, dtype=np.float32)
    cxr_paths = np.asarray(cxr_paths)
    y_all = df.loc[valid_indices, target_col].values.astype(int)
    X_tab_all = X_tab_all[valid_indices]

    print(f"Valid trimodal samples: {len(y_all)}")
    print(f"Class counts: {np.bincount(y_all, minlength=N_CLASSES)}")

    return x_ecg_all, cxr_paths, X_tab_all, y_all, num_features, cat_features, cat_dims


# ============================================================


def create_repeat_loaders(
    x_ecg_all: np.ndarray,
    cxr_paths: np.ndarray,
    x_tab_all: np.ndarray,
    y_all: np.ndarray,
    num_count: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the original stratified 60/20/20 split and train-fitted scalers."""
    (
        x_tab_train, x_tab_temp,
        x_ecg_train, x_ecg_temp,
        x_cxr_train, x_cxr_temp,
        y_train, y_temp,
    ) = train_test_split(
        x_tab_all, x_ecg_all, cxr_paths, y_all,
        test_size=0.4, stratify=y_all, random_state=seed,
    )

    (
        x_tab_val, x_tab_test,
        x_ecg_val, x_ecg_test,
        x_cxr_val, x_cxr_test,
        y_val, y_test,
    ) = train_test_split(
        x_tab_temp, x_ecg_temp, x_cxr_temp, y_temp,
        test_size=0.5, stratify=y_temp, random_state=seed,
    )

    x_tab_train, x_tab_val, x_tab_test = [x.copy() for x in (x_tab_train, x_tab_val, x_tab_test)]
    x_ecg_train, x_ecg_val, x_ecg_test = [x.copy() for x in (x_ecg_train, x_ecg_val, x_ecg_test)]

    scaler_tab = StandardScaler()
    if num_count > 0:
        x_tab_train[:, :num_count] = scaler_tab.fit_transform(x_tab_train[:, :num_count])
        x_tab_val[:, :num_count] = scaler_tab.transform(x_tab_val[:, :num_count])
        x_tab_test[:, :num_count] = scaler_tab.transform(x_tab_test[:, :num_count])

    scaler_ecg = StandardScaler()
    n_train, seq_len, n_channels = x_ecg_train.shape
    x_ecg_train = scaler_ecg.fit_transform(
        x_ecg_train.reshape(-1, n_channels)
    ).reshape(n_train, seq_len, n_channels)
    x_ecg_val = scaler_ecg.transform(
        x_ecg_val.reshape(-1, n_channels)
    ).reshape(x_ecg_val.shape[0], seq_len, n_channels)
    x_ecg_test = scaler_ecg.transform(
        x_ecg_test.reshape(-1, n_channels)
    ).reshape(x_ecg_test.shape[0], seq_len, n_channels)

    return (
        get_loader(x_ecg_train, x_cxr_train, x_tab_train, y_train, BATCH_SIZE, True),
        get_loader(x_ecg_val, x_cxr_val, x_tab_val, y_val, BATCH_SIZE, False),
        get_loader(x_ecg_test, x_cxr_test, x_tab_test, y_test, BATCH_SIZE, False),
    )
