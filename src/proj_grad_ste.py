import torch
from torch.autograd import Function
from typing import List, Tuple

from pishield.linear_requirements.shield_layer import ShieldLayer
from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.constants import EPSILON


def build_constraint_matrix(
    constraints: List[Constraint],
    num_vars: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble the global constraint matrix and bias vector such that:
        constraint_matrix @ y <= bias

    Args:
        constraints: list of PiShield Constraint objects
        num_vars:    number of output variables in y

    Returns:
        constraint_matrix: shape (num_constraints, num_vars)
        bias: shape (num_constraints,)
    """
    rows, rhs = [], []
    for constr in constraints:
        row = torch.zeros(num_vars, dtype=torch.float32)  # shape: (num_vars,)
        ineq = constr.single_inequality

        for atom in ineq.body:
            sign = 1.0 if atom.positive_sign else -1.0
            row[atom.variable.id] = atom.coefficient * sign

        constant = float(ineq.constant)
        if ineq.ineq_sign == ">":
            constant += EPSILON
        elif ineq.ineq_sign == "<":
            constant -= EPSILON

        if ineq.ineq_sign in (">", ">="):
            row = -row
            constant = -constant

        rows.append(row)
        rhs.append(constant)

    constraint_matrix = torch.stack(rows, dim=0)  # shape: (num_constraints, num_vars)
    bias = torch.tensor(rhs, dtype=torch.float32)  # shape: (num_constraints,)
    return constraint_matrix, bias


class ProjGradSTE(Function):
    """
    Custom autograd Function that applies PiShield's constraint clamping in the forward pass
    and uses a tangent projection in the backward pass to allow gradient flow.
    """

    @staticmethod
    def forward(
        ctx,
        predictions: torch.Tensor,
        constraint_matrix: torch.Tensor,
        bias: torch.Tensor,
        shield_layer: ShieldLayer
    ) -> torch.Tensor:
        """
        Forward pass that applies a hard projection via PiShield’s shield layer.
        
        Given a set of linear inequality constraints of the form:
            constraint_matrix @ y <= bias,
        the shield layer modifies `predictions` to ensure constraint satisfaction.
        
        This is done under `torch.no_grad()` to block gradient propagation through
        the projection step itself. Instead, a custom backward pass is defined to 
        provide a meaningful Jacobian-like approximation.

        Args:
            predictions (torch.Tensor): Unconstrained model output of shape (B, D)
            constraint_matrix (torch.Tensor): LHS matrix A in Ay <= b, shape (m, D)
            bias (torch.Tensor): RHS vector b in Ay <= b, shape (m,)
            shield_layer (ShieldLayer): Enforces constraints during inference

        Returns:
            torch.Tensor: Constraint-satisfying output of shape (B, D)
        """

        with torch.no_grad():
            corrected = shield_layer(predictions)  # shape: (batch_size, num_vars)

        constraint_matrix = constraint_matrix.to(predictions.device)
        bias = bias.to(predictions.device)

        ctx.save_for_backward(corrected, constraint_matrix, bias)
        return corrected  # shape: (batch_size, num_vars)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        tolerance: float = 1e-7,
        condition_threshold: float = 1e7
    ) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass that projects the gradient onto the feasible tangent space
        defined by the active constraints using a KKT-style projection.

        The shield layer performs a non-differentiable projection in the forward
        pass. To allow gradient-based optimization, we approximate the Jacobian 
        using a local constrained quadratic projection:

            argmin_v  ||v - grad_output||^2 s.t. A_active @ v = 0

        This ensures that the updated gradient direction lies in the tangent space of
        the active constraint boundary. Using Lagrange multipliers, this leads to:

            v = grad_output - A_active.T @ lambda

        where:
            (A_active @ A_active.T + λI) lambda = A_active @ grad_output
            v = grad_output - A_active.T @ lambda

        This system is solved using either torch.linalg.solve (fast and exact) 
        or torch.linalg.lstsq (robust fallback for ill-conditioned matrices),
        depending on the condition number.

        Args:
            grad_output (torch.Tensor): Gradient from upstream
            tolerance (float): Ridge regularization coefficient (default: 1e-7)

        Returns:
            Tuple[torch.Tensor, None, None, None]: 
                - Gradient with respect to predictions (projected)
                - None for constraint_matrix, bias, and shield_layer (non-differentiable)
        """

        corrected, constraint_matrix, bias = ctx.saved_tensors

        batch_size, _ = corrected.shape
        num_constraints, _ = constraint_matrix.shape
        device = corrected.device

        identity_cache = torch.eye(num_constraints, device=device, dtype=constraint_matrix.dtype)  # shape: (num_constraints, num_constraints)
        grad_input = grad_output.clone()  # shape: (batch_size, num_vars)

        for i in range(batch_size):
            y_i = corrected[i]  # shape: (num_vars,)
            grad_i = grad_input[i]  # shape: (num_vars,)

            active_mask = bias - (constraint_matrix @ y_i) <= EPSILON  # shape: (num_constraints,)
            num_active = int(active_mask.sum().item())

            if num_active > 0:
                active_matrix = constraint_matrix[active_mask]  # shape: (num_active, num_vars)
                lhs = active_matrix @ active_matrix.T  # shape: (num_active, num_active)

                lhs += (tolerance * lhs.diag().mean()).clamp(min=tolerance) * identity_cache[:num_active, :num_active]  # shape: (num_active, num_active)
                rhs = active_matrix @ grad_i.unsqueeze(-1)  # shape: (num_active, 1)

                try:
                    if torch.linalg.cond(lhs) > condition_threshold:
                        lagrangian = torch.linalg.lstsq(lhs, rhs, driver='gelsd').solution  # shape: (num_active, 1)
                    else:
                        lagrangian = torch.linalg.solve(lhs, rhs)  # shape: (num_active, 1)
                except RuntimeError:
                    lagrangian = torch.linalg.lstsq(lhs, rhs, driver='gelsd').solution  # shape: (num_active, 1)

                projection = active_matrix.T @ lagrangian  # shape: (num_vars, 1)
                grad_input[i] -= projection.squeeze(1)  # shape: (num_vars,)

        return grad_input, None, None, None
