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

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.ln1 = nn.LayerNorm(64)

        self.act2 = nn.LeakyReLU(inplace=True)
        self.ln2 = nn.LayerNorm(128)

        self.act3 = nn.LeakyReLU(inplace=True)
        self.ln3 = nn.LayerNorm(128)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.orthogonal_(self.fc3.weight)
        torch.nn.init.orthogonal_(self.fc4.weight)

        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()

    def forward(self, x):
        x = self.ln1(self.act1(self.fc1(x)))
        x = self.ln2(self.act2(self.fc2(x)))
        x = self.ln3(self.act3(self.fc3(x)))
        return self.fc4(x)


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
