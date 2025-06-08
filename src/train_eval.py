from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.adjusted_constraint_loss import adjusted_constraint_loss


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    constraints: list[Constraint],
    device: torch.device,
    logger: logging.Logger,
    mask_method: bool = False,
):
    model.train()
    tot_loss, batches, sat_batches = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        if mask_method:
            out = model(xb, yb)
            shield_masks = model.shield.masks
            loss = adjusted_constraint_loss(out, yb, shield_masks)
        else:
            out = model(xb)
            loss = F.mse_loss(out, yb)
        loss.backward()
        opt.step()

        sat_batches += validate_constraints(constraints, out, logger, verbose=False)
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
    logger: logging.Logger,
):
    model.eval()
    preds, targs = [], []
    sat_batches, batches = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds.append(out.cpu())
        targs.append(yb.cpu())
        sat_batches += validate_constraints(constraints, out, logger, verbose=False)
        batches += 1

    preds_t = torch.cat(preds)
    targs_t = torch.cat(targs)

    if scaler is not None:
        preds_t = torch.tensor(scaler.inverse_transform(preds_t), dtype=torch.float32)
        targs_t = torch.tensor(scaler.inverse_transform(targs_t), dtype=torch.float32)

    rmse = torch.sqrt(F.mse_loss(preds_t, targs_t)).item()
    return rmse, sat_batches, batches
