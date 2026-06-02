"""
Neutrino projects the momentum matrix onto a thin k-dimensional random basis
regenerated locally from a shared seed, and preserves the unprojected
residual as error feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


try:
    from emerging_optimizers.orthogonalized_optimizers import (
        get_muon_scale_factor as _emerging_get_muon_scale_factor,
    )
except ImportError:
    _emerging_get_muon_scale_factor = None


def get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    if mode == "shape_up":
        return max(size_out / size_in, size_in / size_out) ** 0.5
    if mode == "none":
        return 1.0
    if _emerging_get_muon_scale_factor is not None:
        return _emerging_get_muon_scale_factor(size_out, size_in, mode=mode)
    # kimi's muon fallback
    return 0.2 * (max(size_out, size_in) ** 0.5)


@torch.no_grad()
def newton_schulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Important: Different from NS5 from muown.py !

    Quintic iteration with the canonical Muon coefficients (3.4445, -4.7750, 2.0315)
    in float32/bf16. Supports 2D and 3D tensors.
    """
    assert G.ndim in (2, 3), f"newton_schulz5 expects 2D or 3D, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class Neutrino(Optimizer):
    """Neutrino Optimizer

    For 2D matrix parameters and 3D expert parameters, applies Muon with projection of the momentum matrix + EF.
    Other parameters (embeddings, biases, 1D gains) are routed to AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        k: int = 512,  # random basis vectors for proj
        no_momentum: bool = False,  # not to allocate buffer
        scale_mode: str = "spectral",  # same as  --muon-scale-mode, we recommend a "shape_up" mode; see get_muon_scale_factor
        basis_init: str = "gaussian",  # gaussian (default), rademacher, uniform, orthonormal -- gaussian might be the best from math persepctive
        pg_collection: Optional[
            Any
        ] = None,  # holds all the distributed process communication groups, needed for TP and EP
        tp_mode: str = "duplicated",  # forced to be like that
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            k=k,
            no_momentum=no_momentum,
            scale_mode=scale_mode,
            basis_init=basis_init,
        )
        super().__init__(params, defaults)
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        # assign a deterministic unique param_id to each parameter in the optimizer
        # this is used to form the random basis generation seed
        param_id = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['param_id'] = param_id
                param_id += 1

    def _generate_basis(
        self,
        p: Tensor,
        N: int,
        k: int,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
        partition_dim: int | None = None,
        tp_group: Optional[Any] = None,
        basis_init: str = "gaussian",
    ) -> Tensor:
        """Deterministic basis matrix generation using step seed and param_id."""
        state = self.state[p]
        param_id = state['param_id']

        tp_rank = 0
        N_global = N
        if partition_dim == 1 and tp_group is not None and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            N_global = N * torch.distributed.get_world_size(group=tp_group)

        # combine step, param_id, and tp_rank deterministically for a 32-bit uint seed
        # hashing formula with prime multipliers
        seed = (step * 997 + param_id * 1000003 + tp_rank * 17) & 0xFFFFFFFF

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        if basis_init == "rademacher":
            # draw random signs +1 / -1
            rand_u = torch.rand((N, k), generator=generator, device=device, dtype=dtype)
            V_raw = torch.where(rand_u > 0.5, torch.ones_like(rand_u), -torch.ones_like(rand_u))
        elif basis_init == "uniform":
            # draw Uniform[-sqrt(3), sqrt(3)] (variance 1)
            rand_u = torch.rand((N, k), generator=generator, device=device, dtype=dtype)
            V_raw = (rand_u * 2.0 - 1.0) * (3.0**0.5)
        elif basis_init == "orthonormal":
            # Haar random orthonormal matrix via QR
            G = torch.randn((N, k), generator=generator, device=device, dtype=torch.float32)
            Q, R = torch.linalg.qr(G)
            # correct the sign of Q columns to make it deterministic across ranks
            d = torch.diagonal(R, dim1=-2, dim2=-1)
            ph = d.sign()
            Q = Q * ph.unsqueeze(-2)
            V_raw = Q.to(dtype)
            # orthonormal Q already has columns of norm 1 (expected squared element value is 1/N)
            # no further scaling is needed since Q^T Q = I.
            return V_raw
        else:
            # gaussian (default)
            V_raw = torch.randn((N, k), generator=generator, device=device, dtype=dtype)

        return V_raw / (N_global**0.5)

    def _cholesky_qr(self, Gram: Tensor) -> Tensor:
        """Cholesky factorization with adaptive jitter and SVD fallback."""
        k = Gram.shape[-1]
        device = Gram.device
        dtype = Gram.dtype

        if Gram.ndim == 3:
            # batched over experts
            E = Gram.shape[0]
            diag = torch.diagonal(Gram, dim1=-2, dim2=-1)  # [E, k]
            mean_diag = diag.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [E, 1, 1]
            jitter = 1e-4 * mean_diag
            eye = torch.eye(k, device=device, dtype=dtype).unsqueeze(0)
            Gram_jittered = Gram + jitter * eye

            try:
                L = torch.linalg.cholesky(Gram_jittered)
            except RuntimeError:
                # fallback to stronger jitter
                Gram_jittered = Gram + 1e-2 * mean_diag * eye
                try:
                    L = torch.linalg.cholesky(Gram_jittered)
                except RuntimeError:
                    L = self._svd_fallback(Gram)
        else:
            diag = torch.diagonal(Gram)
            mean_diag = diag.mean()
            jitter = 1e-4 * mean_diag
            eye = torch.eye(k, device=device, dtype=dtype)
            Gram_jittered = Gram + jitter * eye

            try:
                L = torch.linalg.cholesky(Gram_jittered)
            except RuntimeError:
                Gram_jittered = Gram + 1e-2 * mean_diag * eye
                try:
                    L = torch.linalg.cholesky(Gram_jittered)
                except RuntimeError:
                    L = self._svd_fallback(Gram)
        return L

    def _svd_fallback(self, Gram: Tensor) -> Tensor:
        """SVD / Eigenvalue decomposition fallback for singular Gram matrices."""
        eigenvalues, eigenvectors = torch.linalg.eigh(Gram)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        return eigenvectors @ torch.diag_embed(eigenvalues.sqrt())

    def _solve_triangular(self, L: Tensor, Y: Tensor) -> Tensor:
        """Solve lower triangular system L @ X = Y.T to obtain U = X.T."""
        Y_T = Y.transpose(-2, -1)
        X = torch.linalg.solve_triangular(L, Y_T, upper=False, left=True)
        return X.transpose(-2, -1)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_metrics = {}
        total_error_ratio = 0.0
        total_update_ratio = 0.0
        total_orth_deviation = 0.0
        total_gram_cond = 0.0
        num_parameters = 0
        total_neutrino_bytes = 0
        total_muon_bytes = 0

        try:
            import os

            is_rank_0 = int(os.environ.get("RANK", "0")) == 0
        except Exception:
            is_rank_0 = True

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            k = group['k']
            no_momentum = group['no_momentum']
            scale_mode = group['scale_mode']
            basis_init = group.get('basis_init', 'gaussian')

            for p in group['params']:
                if p.grad is None:
                    continue

                # operations in float32 for numerical stability
                grad = p.grad.data.float()

                if p.ndim not in (2, 3):
                    p.data.add_(grad.to(p.dtype), alpha=-lr)
                    continue

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    if not no_momentum:
                        state['momentum_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['error_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)

                state['step'] += 1
                step = state['step']

                if p.ndim == 3:
                    E, M, N = p.shape
                else:
                    E = None
                    M, N = p.shape

                tp_group = None
                if self.pg_collection:
                    tp_group = (
                        self.pg_collection.expt_tp
                        if getattr(p, 'expert_tp', False)
                        else self.pg_collection.tp
                    )

                partition_dim = getattr(p, 'partition_dim', None)
                if partition_dim == -1:
                    partition_dim = None

                # Compute global dimensions for aspect scaling
                M_global = M
                N_global = N
                if tp_group is not None and torch.distributed.is_initialized():
                    tp_size = torch.distributed.get_world_size(group=tp_group)
                    if partition_dim == 0:
                        M_global = M * tp_size
                    elif partition_dim == 1:
                        N_global = N * tp_size

                # fallback: if min(M_global, N_global) <= k, use full-rank Muon
                if min(M_global, N_global) <= k:
                    if no_momentum:
                        g_adj = grad
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                        if nesterov:
                            g_adj = grad.mul(1.0 - momentum).add_(buf, alpha=momentum)
                        else:
                            g_adj = buf

                    U = newton_schulz5(g_adj, steps=5)

                    if wd != 0:
                        p.data.mul_(1.0 - lr * wd)

                    scale = get_muon_scale_factor(M_global, N_global, mode=scale_mode)
                    p.data.add_(U.to(p.dtype), alpha=-lr * scale)
                    continue

                if no_momentum:
                    g_hat = grad
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    if nesterov:
                        g_hat = grad.mul(1.0 - momentum).add_(buf, alpha=momentum)
                    else:
                        g_hat = buf

                # error feedback adjustment
                e_prev = state['error_buffer']
                g_adj = g_hat + e_prev

                # basis generation
                V = self._generate_basis(
                    p,
                    N,
                    k,
                    step,
                    grad.device,
                    grad.dtype,
                    partition_dim=partition_dim,
                    tp_group=tp_group,
                    basis_init=basis_init,
                )

                # subspace projection
                if E is not None:
                    Y_local = torch.matmul(g_adj, V)
                else:
                    Y_local = g_adj @ V

                # TP collective communication & Cholesky-QR
                if (
                    tp_group is not None
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_world_size(group=tp_group) > 1
                ):
                    if partition_dim == 0:
                        # Row dimension is partitioned
                        if E is not None:
                            Gram_local = torch.matmul(Y_local.transpose(-2, -1), Y_local)
                        else:
                            Gram_local = Y_local.T @ Y_local

                        torch.distributed.all_reduce(Gram_local, group=tp_group)
                        Gram = Gram_local

                        L = self._cholesky_qr(Gram)
                        U_local = self._solve_triangular(L, Y_local)

                        if E is not None:
                            update = torch.matmul(U_local, V.transpose(-2, -1))
                            YV = torch.matmul(Y_local, V.transpose(-2, -1))
                        else:
                            update = U_local @ V.T
                            YV = Y_local @ V.T

                    elif partition_dim == 1:
                        # column dimension is partitioned
                        torch.distributed.all_reduce(Y_local, group=tp_group)
                        Y = Y_local

                        if E is not None:
                            Gram = torch.matmul(Y.transpose(-2, -1), Y)
                        else:
                            Gram = Y.T @ Y

                        L = self._cholesky_qr(Gram)
                        U = self._solve_triangular(L, Y)

                        if E is not None:
                            update = torch.matmul(U, V.transpose(-2, -1))
                            YV = torch.matmul(Y, V.transpose(-2, -1))
                        else:
                            update = U @ V.T
                            YV = Y @ V.T

                    else:
                        # duplicated TP
                        if E is not None:
                            Gram = torch.matmul(Y_local.transpose(-2, -1), Y_local)
                        else:
                            Gram = Y_local.T @ Y_local

                        L = self._cholesky_qr(Gram)
                        U = self._solve_triangular(L, Y_local)

                        if E is not None:
                            update = torch.matmul(U, V.transpose(-2, -1))
                            YV = torch.matmul(Y_local, V.transpose(-2, -1))
                        else:
                            update = U @ V.T
                            YV = Y_local @ V.T
                else:
                    # single rank / non-distributed group
                    if E is not None:
                        Gram = torch.matmul(Y_local.transpose(-2, -1), Y_local)
                    else:
                        Gram = Y_local.T @ Y_local

                    L = self._cholesky_qr(Gram)
                    U = self._solve_triangular(L, Y_local)

                    if E is not None:
                        update = torch.matmul(U, V.transpose(-2, -1))
                        YV = torch.matmul(Y_local, V.transpose(-2, -1))
                    else:
                        update = U @ V.T
                        YV = Y_local @ V.T

                # update
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                scale = get_muon_scale_factor(M_global, N_global, mode=scale_mode)
                p.data.add_(update.to(p.dtype), alpha=-lr * scale)

                # error feedback update
                state['error_buffer'].copy_(g_adj - YV)

                # --- metrics (rank-0 only) ---
                if is_rank_0:
                    err_norm = torch.linalg.vector_norm(g_adj - YV)
                    g_adj_norm = torch.linalg.vector_norm(g_adj)
                    error_ratio = (err_norm / (g_adj_norm + 1e-8)).item()

                    if E is not None:
                        UTU = torch.matmul(U.transpose(-2, -1), U)
                        eye = torch.eye(k, device=U.device, dtype=U.dtype).unsqueeze(0)
                        orth_dev = torch.linalg.vector_norm(UTU - eye).mean().item()
                    else:
                        UTU = U.T @ U
                        eye = torch.eye(k, device=U.device, dtype=U.dtype)
                        orth_dev = torch.linalg.vector_norm(UTU - eye).item()

                    update_norm = torch.linalg.vector_norm(update)
                    w_norm = torch.linalg.vector_norm(p.data)
                    update_ratio = (lr * scale * update_norm / (w_norm + 1e-8)).item()

                    if E is not None:
                        eigvals = torch.linalg.eigvalsh(Gram)
                        cond_per_expert = eigvals.max(dim=-1).values / (
                            eigvals.min(dim=-1).values + 1e-8
                        )
                        gram_cond = cond_per_expert.mean().item()
                    else:
                        eigvals = torch.linalg.eigvalsh(Gram)
                        gram_cond = (eigvals.max() / (eigvals.min() + 1e-8)).item()

                    tp_size = 1
                    if tp_group is not None and torch.distributed.is_initialized():
                        tp_size = torch.distributed.get_world_size(group=tp_group)

                    num_experts = E if E is not None else 1
                    if tp_size > 1:
                        if partition_dim == 0:
                            p_neutrino_bytes = num_experts * k * k * 4
                        elif partition_dim == 1:
                            p_neutrino_bytes = num_experts * M * k * 4
                        else:
                            p_neutrino_bytes = 0
                        p_muon_bytes = num_experts * M * N * 2
                    else:
                        p_neutrino_bytes = 0
                        p_muon_bytes = 0

                    total_neutrino_bytes += p_neutrino_bytes
                    total_muon_bytes += p_muon_bytes

                    p_name = getattr(p, 'param_name', f'param_{state["param_id"]}')
                    p_name_clean = p_name.replace("model.decoder.layers.", "layer_")

                    step_metrics[f"neutrino/{p_name_clean}/error_ratio"] = error_ratio
                    step_metrics[f"neutrino/{p_name_clean}/gram_cond"] = gram_cond
                    step_metrics[f"neutrino/{p_name_clean}/orth_dev"] = orth_dev
                    step_metrics[f"neutrino/{p_name_clean}/update_ratio"] = update_ratio

                    total_error_ratio += error_ratio
                    total_update_ratio += update_ratio
                    total_orth_deviation += orth_dev
                    total_gram_cond += gram_cond
                    num_parameters += 1

        # logging to wandb (rank-0 only)
        if is_rank_0 and num_parameters > 0:
            global_metrics = {
                "neutrino/global_mean_error_ratio": total_error_ratio / num_parameters,
                "neutrino/global_mean_update_ratio": total_update_ratio / num_parameters,
                "neutrino/global_mean_orth_deviation": total_orth_deviation / num_parameters,
                "neutrino/global_mean_gram_cond": total_gram_cond / num_parameters,
                "neutrino/allreduce_bytes_per_step": float(total_neutrino_bytes),
                "neutrino/comm_savings_multiplier": total_muon_bytes
                / (total_neutrino_bytes + 1e-8),
            }
            global_metrics.update(step_metrics)

            try:
                import wandb

                if wandb.run is not None:
                    param_step = 1
                    for group in self.param_groups:
                        for p in group['params']:
                            if p in self.state and 'step' in self.state[p]:
                                param_step = self.state[p]['step']
                                break
                        break
                    wandb.log(global_metrics, step=param_step)
            except Exception:
                pass

        return loss
