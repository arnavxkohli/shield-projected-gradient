from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.parser import (
    parse_constraints_file,
    remap_constraint_variables,
)


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.info(f"Env  → Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda or 'CPU'}")
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(precedence: tuple[str, ...] = ("cuda", "mps", "cpu")) -> torch.device:
    for dev in precedence:
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_data(
    df: pd.DataFrame,
    constraints_file: Path,
    test_size: float,
    val_size: float,
    seed: int,
    scale_targets: bool,
    train_fraction: float,
):

    for col in df.select_dtypes(object).columns:
        df[col] = df[col].apply(lambda x: hash(x) % 10**6)

    ordering, constraints = parse_constraints_file(constraints_file)
    ordering, constraints, rev_map = remap_constraint_variables(ordering, constraints)
    target_cols = [df.columns[int(var.split("_")[1])] for var in rev_map.values()]
    feature_cols = [c for c in df.columns if c not in target_cols]

    if "status" in df and df["status"].dtype == object:
        df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

    X, y = df[feature_cols], df[target_cols]

    if train_fraction < 1.0:
        idx = X.sample(frac=train_fraction, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    cont_feats = [c for c in feature_cols if df[c].nunique() > 2]
    cat_feats = [c for c in feature_cols if c not in cont_feats]
    preproc = ColumnTransformer(
        [("num", StandardScaler(), cont_feats), ("cat", "passthrough", cat_feats)]
    )

    if scale_targets:
        target_scaler = StandardScaler()
        y_train = pd.DataFrame(target_scaler.fit_transform(y_train), columns=target_cols)
        y_val = pd.DataFrame(target_scaler.transform(y_val), columns=target_cols)
        y_test = pd.DataFrame(target_scaler.transform(y_test), columns=target_cols)
    else:
        target_scaler = None

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    X_train_t = to_tensor(preproc.fit_transform(X_train))
    X_val_t = to_tensor(preproc.transform(X_val))
    X_test_t = to_tensor(preproc.transform(X_test))
    y_train_t, y_val_t, y_test_t = map(to_tensor, (y_train.values, y_val.values, y_test.values))

    return (
        (X_train_t, y_train_t),
        (X_val_t, y_val_t),
        (X_test_t, y_test_t),
        constraints,
        len(feature_cols),
        len(target_cols),
        target_scaler,
    )


def parse_numpy_data(logger: logging.Logger, numpy_data: bool, 
                     data_list: list[str]) -> pd.DataFrame:
    if not numpy_data:
        logger.warning("Numpy data not requested, returning empty DataFrame.")
        return pd.DataFrame()

    if len(data_list) != 4:
        raise ValueError("Expected exactly 4 numpy files: X_train, X_test, y_train, y_test")
    
    X_train = np.load(data_list[0])
    X_test = np.load(data_list[1])
    y_train = np.load(data_list[2])
    y_test = np.load(data_list[3])

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    training_data = np.hstack((X_train, y_train))
    test_data = np.hstack((X_test, y_test))
    all_data = np.vstack((training_data, test_data))

    return pd.DataFrame(all_data, columns=[f"y_{i}" for i in range(all_data.shape[1])])


def make_dataloaders(data, batch_size: int) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, (X, y) in zip(("train", "val", "test"), data):
        generator = torch.Generator()
        generator.manual_seed(0)
        loaders[name] = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            generator=generator
        )
    return loaders
