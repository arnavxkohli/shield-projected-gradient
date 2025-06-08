import torch
import torch.nn as nn
from proj_grad_ste import ProjGradSTE, build_constraint_matrix
from pathlib import Path

from pishield.linear_requirements.shield_layer import ShieldLayer

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


class ShieldedMLP(nn.Module):
    def __init__(self, base_net: ShallowMLP | DeepMLP, output_dim: int, 
                 constraints_file: Path):
        super().__init__()
        self.mlp = base_net
        self.shield = ShieldLayer(output_dim, constraints_file)

    def forward(self, x, y=None):
        return self.shield(self.mlp(x), y)


class ShieldWithProjGrad(nn.Module):
    def __init__(self, base_net: ShallowMLP | DeepMLP, output_dim: int, 
                 constraints_file: Path):
        super().__init__()
        self.mlp = base_net
        self.shield = ShieldLayer(output_dim, constraints_file)

        A, b = build_constraint_matrix(self.shield.constraints, output_dim)
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x):
        preds = self.mlp(x)
        return ProjGradSTE.apply(preds, self.A, self.b, self.shield)
