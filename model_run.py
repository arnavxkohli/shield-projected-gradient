#!/usr/bin/env python

from __future__ import annotations
import argparse, logging, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.parser import (
    parse_constraints_file,
    remap_constraint_variables,
)
from pishield.linear_requirements.shield_layer import ShieldLayer
from kkt_ste import KKTShieldSTE, build_constraint_matrix


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device(precedence: tuple[str, ...] = ("cuda", "mps", "cpu")) -> torch.device:
    for dev in precedence:
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_data(
    csv_path: Path,
    constraints_file: Path,
    *,
    test_size: float,
    val_size: float,
    seed: int,
    scale_targets: bool,
    train_fraction: float,
):
    df = pd.read_csv(csv_path)

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


def make_dataloaders(data, batch_size: int) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, (X, y) in zip(("train", "val", "test"), data):
        loaders[name] = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False
        )
    return loaders


def validate_constraints(
    constraints: list[Constraint],
    preds: torch.Tensor,
    logger: logging.Logger,
    verbose: bool,
) -> bool:
    all_ok = True
    if verbose:
        logger.info("Starting constraint check...")
    for c in constraints:
        ok = c.check_satisfaction(preds)
        if not ok and verbose:
            logger.warning(f"Constraint violated: {c.readable()}")
        all_ok &= ok
    if verbose:
        logger.info("Constraint check complete.")
    return all_ok


class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


ARCH_MAP = {
    "shallow": ShallowMLP,
    "deep": DeepMLP,
}


class ShieldedMLP(nn.Module):
    def __init__(self, base_net: ShallowMLP | DeepMLP, output_dim: int, constraints_file: Path):
        super().__init__()
        self.mlp = base_net
        self.shield = ShieldLayer(output_dim, constraints_file)

    def forward(self, x):
        return self.shield(self.mlp(x))


class ShieldedMLPWithKKTSTE(nn.Module):
    def __init__(self, base_net: ShallowMLP | DeepMLP, output_dim: int, constraints_file: Path):
        super().__init__()
        self.mlp = base_net
        self.shield = ShieldLayer(output_dim, constraints_file)

        A, b = build_constraint_matrix(self.shield.constraints, output_dim)
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x):
        preds = self.mlp(x)
        return KKTShieldSTE.apply(preds, self.A, self.b, self.shield)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    constraints: list[Constraint],
    device: torch.device,
):
    model.train()
    tot_loss, batches, sat_batches = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = F.mse_loss(out, yb)
        loss.backward()
        opt.step()

        sat_batches += validate_constraints(constraints, out, logging.getLogger(), verbose=False)
        tot_loss += loss.item()
        batches += 1
    return tot_loss / batches, sat_batches, batches


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    constraints: list[Constraint],
    scaler: StandardScaler | None,
    device: torch.device,
):
    model.eval()
    preds, targs = [], []
    sat_batches, batches = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds.append(out.cpu())
        targs.append(yb.cpu())
        sat_batches += validate_constraints(constraints, out, logging.getLogger(), verbose=False)
        batches += 1

    preds_t = torch.cat(preds)
    targs_t = torch.cat(targs)

    if scaler is not None:
        preds_t = torch.tensor(scaler.inverse_transform(preds_t), dtype=torch.float32)
        targs_t = torch.tensor(scaler.inverse_transform(targs_t), dtype=torch.float32)

    rmse = torch.sqrt(F.mse_loss(preds_t, targs_t)).item()
    return rmse, sat_batches, batches


def main(args):
    device = get_device()
    out_dir = Path("out") / args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / f"{args.base_arch}_log.txt")
    logger.info(f"Using device: {device}")

    results_csv = out_dir / f"final_rmses_{args.base_arch}.csv"
    results_csv.write_text("model,trial,test_rmse,test_sat\n")

    base_cls = ARCH_MAP[args.base_arch]

    for trial in range(args.trials):
        seed = args.seed + trial
        set_seed(seed)
        logger.info(f"\n=== Trial {trial + 1}/{args.trials} | Seed={seed} ===")

        data = load_data(
            Path("data") / args.data_dir / "data.csv",
            Path("data") / args.data_dir / "constraints.txt",
            test_size=args.test_size,
            val_size=args.val_size,
            seed=seed,
            scale_targets=args.scale_targets,
            train_fraction=args.train_fraction,
        )
        (train_d, val_d, test_d, constraints, in_dim, out_dim, scaler) = data
        loaders = make_dataloaders((train_d, val_d, test_d), args.batch_size)

        for name, ctor in [
            (base_cls.__name__.capitalize(), lambda: base_cls(in_dim, out_dim)),
            ("ShieldedMLP", lambda: ShieldedMLP(base_cls(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt")),
            ("ShieldedMLPWithKKTSTE", lambda: ShieldedMLPWithKKTSTE(base_cls(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt")),
        ]:
            model = ctor().to(device)
            opt = (optim.Adam if args.optimizer == "adam" else optim.SGD)(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                tloss, tsat, _ = train_epoch(model, loaders["train"], opt, constraints, device)
                vrmse, vsat, _ = eval_epoch(model, loaders["val"], constraints, scaler, device)
                logger.info(f"{name:>24} | Epoch {epoch:02d}/{args.epochs} | "
                            f"loss {tloss:.4f} | val RMSE {vrmse:.4f} | val sat {vsat}")

            trmse, tsat, _ = eval_epoch(model, loaders["test"], constraints, scaler, device)
            logger.info(f"{name:>24} | TEST RMSE {trmse:.4f} | test sat {tsat}")
            with results_csv.open("a") as f:
                f.write(f"{name},{trial},{trmse:.4f},{tsat}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="url")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.125)
    p.add_argument("--scale-targets", action="store_true")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--base-arch", choices=["shallow", "deep"], default="shallow")
    main(p.parse_args())
