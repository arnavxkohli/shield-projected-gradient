#!/usr/bin/env python

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from train_eval import validate_constraints, train_epoch, eval_epoch
from setup import setup_logging, get_device, set_seed, load_data, parse_numpy_data, make_dataloaders
from models import DeepMLP, ShieldedMLP, ShieldWithProjGrad

from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.adjusted_constraint_loss import adjusted_constraint_loss


class GradientAnalyzer:
    def __init__(self, device: torch.device, mask_zero: bool = True):
        self.device = device
        self.mask_zero = mask_zero

    def extract_mlp_gradients(self, model: nn.Module) -> torch.Tensor:
        gradients = []

        if hasattr(model, 'mlp'):
            mlp = model.mlp
        else:
            mlp = model

        for param in mlp.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        return torch.cat(gradients) if gradients else torch.tensor([])

    def compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        if grad1.numel() == 0 or grad2.numel() == 0:
            return 0.0

        if self.mask_zero:
            mask = (grad1 != 0) | (grad2 != 0)
            grad1, grad2 = grad1[mask], grad2[mask]
            if grad1.numel() == 0 or grad2.numel() == 0:
                return 0.0

        cos_sim = F.cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0)).item()
        return cos_sim

    def sync_model_parameters(self, baseline_model: nn.Module, 
                              shielded_model: nn.Module,
                              masked_model: nn.Module,
                              proj_grad_model: nn.Module):
        baseline_params = list(baseline_model.parameters())
        shielded_params = list(shielded_model.mlp.parameters())
        masked_params = list(masked_model.mlp.parameters())
        proj_params = list(proj_grad_model.mlp.parameters())

        for bp, sp, mp, pjp in zip(baseline_params, shielded_params, masked_params, proj_params):
            sp.data.copy_(bp.data)
            mp.data.copy_(bp.data)
            pjp.data.copy_(bp.data)

    def extract_gradient_predictions(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                                     model: nn.Module, 
                                     is_masked: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        model.zero_grad()
        if is_masked:
            predictions = model(batch_x, batch_y)
            shield_masks = model.shield.masks
            loss = adjusted_constraint_loss(predictions, batch_y, shield_masks)
        else:
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
        loss.backward()
        gradients = self.extract_mlp_gradients(model)
        return gradients, predictions

    def analyze_batch_gradients(
        self, 
        models: Dict[str, nn.Module],
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        constraints: List[Constraint]
    ) -> Dict[str, float]:

        baseline_model = models["baseline"]
        shielded_model = models["shielded"]
        masked_model = models["masked"]
        proj_model = models["proj"]

        self.sync_model_parameters(baseline_model, shielded_model, masked_model, proj_model)

        results = {}

        baseline_grad, baseline_pred = self.extract_gradient_predictions(batch_x, batch_y, baseline_model)
        shielded_grad, shielded_pred = self.extract_gradient_predictions(batch_x, batch_y, shielded_model)
        masked_grad, masked_pred = self.extract_gradient_predictions(batch_x, batch_y, masked_model, is_masked=True)
        proj_grad, proj_pred = self.extract_gradient_predictions(batch_x, batch_y, proj_model)

        results["baseline_vs_shielded"] = self.compute_cosine_similarity(baseline_grad, shielded_grad)
        results["baseline_vs_masked"] = self.compute_cosine_similarity(baseline_grad, masked_grad)
        results["baseline_vs_proj"] = self.compute_cosine_similarity(baseline_grad, proj_grad)

        results["baseline_violations"] = sum(1 for c in constraints if not c.check_satisfaction(baseline_pred))
        results["shielded_violations"] = sum(1 for c in constraints if not c.check_satisfaction(shielded_pred))
        results["masked_violations"] = sum(1 for c in constraints if not c.check_satisfaction(masked_pred))
        results["proj_violations"] = sum(1 for c in constraints if not c.check_satisfaction(proj_pred))

        return results

    def analyze_dataset(
        self,
        models: Dict[str, nn.Module],
        data_loader,
        constraints: List[Constraint],
        max_batches: int = 100
    ) -> pd.DataFrame:
        results = []

        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader, desc="Analyzing gradients")):
            if batch_idx >= max_batches:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            batch_results = self.analyze_batch_gradients(models, batch_x, batch_y, constraints)
            batch_results["batch_idx"] = batch_idx
            results.append(batch_results)

        return pd.DataFrame(results)
