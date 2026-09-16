"""
Dion: a warm-started, power-iterated low-rank orthonormalized update.

Reference: Ahn, Xu, Abreu, Fan, Magakyan, Sharma, Zhan, Langford, "Dion: Distributed
Orthonormalized Updates" (arXiv:2504.05295), code: https://github.com/microsoft/dion. This is a
from-scratch Megatron-native port (the upstream code is built on PyTorch DTensor/FSDP2, which does
not map onto Megatron's TP/EP process-group conventions) — it follows the same shape-bucketed,
TP/EP-aware architecture as ``neutrino.py`` in this package, not an import of the upstream package.

Algorithmic difference from Neutrino (see neutrino.py's module docstring for that side): Neutrino
regenerates a fresh *random* basis every step from a shared seed, so the basis costs zero bytes to
communicate but only weakly compresses the gradient's information (needs error feedback to
recover, and only mildly-compressed subspaces perform well without it). Dion instead keeps a
persistent, *warm-started* basis Q that is refined by exactly one power-iteration step per training
step. Because Q tracks the true dominant subspace of the momentum (which drifts slowly across
steps, not randomly), the resulting projection is higher-fidelity per rank — the trade-off is that
Q is now real optimizer *state* (not free to regenerate), so it must be checkpointed like a
momentum buffer, and any DP-projection communication saving requires actually exchanging the thin
P/R factors on the wire, not a zero-cost seed.

Per (batched) parameter of shape [M, N] with rank k:
  1. Accumulate momentum: M_buf += G  (no decay factor; decay happens implicitly via step 4 below)
  2. Project onto the *previous* step's basis: P = M_buf @ Q
  3. Orthonormalize P via QR (this step's U-like factor)
  4. Error feedback: M_buf -= (1 - mu) * P @ R^T  (subtract the part now captured; see below for R)
  5. R = M_buf^T @ P (after step 4's subtraction — mirrors the reference implementation's ordering)
  6. Q <- column-normalize(R)  (the new, refined basis for *next* step's projection); with
     orth='qr' (Orth-Dion, arXiv:2605.16341) Q <- qr(R) instead, so Q has orthonormal columns
  7. Apply the weight update: W -= lr * shape_scale * (P @ Q^T)   (using the JUST-refreshed Q)

Communication (default / safe mode, matching neutrino.py's non-dp_projection default): this
implementation assumes Megatron's DDP has already all-reduced the dense gradient before
``step()`` runs (the standard path), so M_buf/P/Q are already DP-consistent with no extra
collective needed here — same reasoning as Neutrino without ``--neutrino-dp-projection``.
A ``dion_dp_projection`` flag exists for symmetry with Neutrino's design but is NOT yet wired to
a DDP-level skip-hook in this tree (see neutrino.py's module docstring on the same limitation) —
leaving it False is always safe, just not communication-optimal.

Scope note (V1): TP-sharded params are handled in "duplicated" mode only (same simplification
Neutrino's own tp_mode default uses) — full row/col-sharded Dion (reconstructing P/R exactly
across TP shards) is future work, not required for the DP=N, TP=1 comparisons this port targets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer, get_standard_config_overrides
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

logger = logging.getLogger(__name__)


def _shape_scale(size_out: int, size_in: int) -> float:
    """Aspect-ratio scale factor, matching the reference dion_update's sqrt(fan_out/fan_in)."""
    return (size_out / size_in) ** 0.5


class Dion(Optimizer):
    """Dion Optimizer.

    For 2D matrix parameters and 3D expert parameters, maintains a persistent warm-started
    low-rank basis refined by one power-iteration step per training step. Other parameters
    (embeddings, biases, 1D gains) are routed to AdamW by the factory function below.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        mu: float = 0.95,  # error-feedback retention: M -= (1 - mu) * reconstructed part
        weight_decay: float = 0.0,
        rank: int = 256,
        eps: float = 1e-8,
        dp_projection: bool = False,
        pg_collection: Optional[Any] = None,
        tp_mode: str = "duplicated",
        orth: str = "colnorm",
    ) -> None:
        defaults = dict(lr=lr, mu=mu, weight_decay=weight_decay, rank=rank, eps=eps)
        super().__init__(params, defaults)
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.orth = orth
        self.dp_projection = dp_projection
        self._global_step = 0

        param_id = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['param_id'] = param_id
                param_id += 1
                if self.dp_projection and p.ndim in (2, 3):
                    p.dion_managed = True

    # ------------------------------------------------------------------
    # checkpoint safety — same fix as Neutrino.state_dict(): the layer-wise
    # distributed optimizer shards param_groups per rank, but self.state (a
    # defaultdict) can retain entries for params this rank doesn't own; torch's
    # base state_dict() would KeyError on those. Filter to owned params only.
    # ------------------------------------------------------------------
    def build_sharded_optimizer_state(self, model_param, value, state_key, prefix):
        """Keep non-model-shaped optimizer state out of the sharded checkpoint.

        Hook called by dist_checkpointing.optimizer.optim_state_to_sharding_state. A non-None
        return is used as-is (no shape assertion); None falls back to the model-shaped path.

        Dion keeps two things in self.state that are not model-shaped, and each one on its own
        is enough to kill save_checkpoint after a run has trained perfectly to its last
        iteration: ``param_id`` is a bare int, and ``Q`` is [n, k] while the parameter is
        [m, n]. make_sharded_optimizer_tensor asserts the value's shape against the model
        param, so both blow up.

        Both are reconstructible. ``param_id`` is reassigned in __init__. ``Q`` is a power
        iteration basis that is refreshed every step from the momentum; restarting it from the
        same random init it gets at step 0 costs a few steps of refinement, not correctness.
        """
        from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject

        # Claim by key name, never by shape introspection: model_param may be a
        # ShardedTensorFactory with no .local_shape, and a shape-based test then falls
        # through to the asserting path and fails later inside validate_metadata_integrity.
        if state_key == "param_id" or state_key == "Q":
            return LocalNonpersistentObject(value)
        return None

    def state_dict(self):
        live = {id(p) for group in self.param_groups for p in group['params']}
        full_state = self.state
        self.state = defaultdict(dict, {p: s for p, s in full_state.items() if id(p) in live})
        try:
            return super().state_dict()
        finally:
            self.state = full_state

    def _tp_group_for(self, p: Tensor):
        if not self.pg_collection:
            return None
        return self.pg_collection.expt_tp if getattr(p, 'expert_tp', False) else self.pg_collection.tp

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            wd = group['weight_decay']
            rank = group['rank']
            eps = group['eps']

            buckets: dict[tuple, list] = {}
            experts_3d: list = []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim not in (2, 3):
                    p.data.add_(p.grad.data.to(p.dtype), alpha=-lr)
                    continue
                if p.ndim == 3:
                    experts_3d.append(p)
                else:
                    partition_dim = getattr(p, 'partition_dim', None)
                    if partition_dim == -1:
                        partition_dim = None
                    key = (tuple(p.shape), partition_dim, bool(getattr(p, 'expert_tp', False)))
                    buckets.setdefault(key, []).append(p)

            for key, items in buckets.items():
                self._process_bucket(items, key, lr, mu, wd, rank, eps)
            for p in experts_3d:
                self._process_expert(p, lr, mu, wd, rank, eps)

        return loss

    def _init_state(self, p: Tensor, N: int, k: int, P_batch: int, device, dtype) -> Dict[str, Tensor]:
        state = self.state[p]
        if 'momentum' not in state:
            state['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
        if 'Q' not in state:
            # Q persists across steps (warm start) — initialized once from a random basis;
            # subsequent steps refine it via the power-iteration update in _dion_step.
            state['Q'] = torch.randn(N, k, device=device, dtype=torch.float32) / (N ** 0.5)
        return state

    def _dion_step(self, X: Tensor, M: Tensor, Q: Tensor, lr: float, mu: float, wd: float,
                   eps: float, tp_group, partition_dim: int, expert: bool) -> None:
        """In-place Dion update for a single (possibly batched-over-bucket) tensor.

        X: [.., M, N] (the param data, batched or not); M/Q: matching momentum/basis state.
        Shapes: X/M are [P, m, n] or [m, n]; Q is [P, n, k] or [n, k] — bmm is used when batched
        (bucket of P same-shape params), matmul otherwise (single 3D expert tensor [E, m, n]).
        """
        batched = X.dim() == 3
        mm = torch.bmm if batched else torch.matmul

        P_ = mm(M, Q)  # project onto the (still-old) basis
        if tp_group is not None and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(group=tp_group)
            if tp_size > 1 and partition_dim == 1:
                # column-sharded: each TP rank holds a disjoint slice of the contraction
                # dimension, so local P is a partial sum — reconstruct the global P.
                torch.distributed.all_reduce(P_, group=tp_group)
        P32, _ = torch.linalg.qr(P_.to(torch.float32))

        R = mm(M.transpose(-2, -1), P32)
        M.add_(mm(P32, R.transpose(-2, -1)), alpha=-(1 - mu))  # error feedback (in-place)

        R32 = R.to(torch.float32)
        if self.orth == 'qr':
            Q.copy_(torch.linalg.qr(R32)[0])  # Orth-Dion: orthonormal columns, not unit columns
        else:
            denom = R32.norm(dim=-2, keepdim=True) + eps
            Q.copy_(R32 / denom)  # refreshed basis for next step

        m_dim, n_dim = X.shape[-2], X.shape[-1]
        scale = _shape_scale(m_dim, n_dim)
        if wd != 0:
            X.mul_(1.0 - lr * wd)
        X.add_(mm(P32, Q.transpose(-2, -1)).to(X.dtype), alpha=-lr * scale)

    def _process_bucket(self, items, key, lr, mu, wd, rank, eps):
        shape, partition_dim, expert_tp = key
        M_dim, N_dim = shape
        k_eff = max(1, min(rank, min(M_dim, N_dim)))
        p0 = items[0]
        device = p0.device
        tp_group = self._tp_group_for(p0)

        Ms, Qs, Xs = [], [], []
        for p in items:
            grad = p.grad.data.float()
            state = self._init_state(p, N_dim, k_eff, len(items), device, torch.float32)
            state['momentum'].add_(grad)
            Ms.append(state['momentum'])
            Qs.append(state['Q'])
            Xs.append(p.data)

        M_stack = torch.stack(Ms, dim=0)
        Q_stack = torch.stack(Qs, dim=0)
        X_stack = torch.stack([x.float() for x in Xs], dim=0)

        self._dion_step(X_stack, M_stack, Q_stack, lr, mu, wd, eps, tp_group, partition_dim,
                        bool(expert_tp))

        for i, p in enumerate(items):
            p.data.copy_(X_stack[i].to(p.dtype))
            self.state[p]['Q'].copy_(Q_stack[i])
            self.state[p]['momentum'].copy_(M_stack[i])

    def _process_expert(self, p, lr, mu, wd, rank, eps):
        E, M_dim, N_dim = p.shape
        k_eff = max(1, min(rank, min(M_dim, N_dim)))
        device = p.device
        tp_group = self._tp_group_for(p)
        partition_dim = getattr(p, 'partition_dim', None)
        if partition_dim == -1:
            partition_dim = None

        grad = p.grad.data.float()
        state = self.state[p]
        if 'momentum' not in state:
            state['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
        if 'Q' not in state:
            # Shared basis across experts of identical shape: one [N, k] matrix, broadcast
            # over the expert batch dim (matches Neutrino's expert-basis sharing rationale).
            state['Q'] = torch.randn(N_dim, k_eff, device=device, dtype=torch.float32) / (N_dim ** 0.5)
        state['momentum'].add_(grad)

        M_buf = state['momentum']
        Q = state['Q']
        X = p.data.float()

        Q_batched = Q.unsqueeze(0).expand(E, -1, -1).contiguous()
        self._dion_step(X, M_buf, Q_batched, lr, mu, wd, eps, tp_group, partition_dim, True)
        p.data.copy_(X.to(p.dtype))
        # Collapse the (expanded, now-refined-per-expert) Q back to a single shared basis by
        # averaging over experts — keeps state size O(N*k) instead of O(E*N*k).
        Q.copy_(Q_batched.mean(dim=0))


def get_megatron_dion_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build the Dion optimizer for the given model chunks. Mirrors
    :func:`megatron.core.optimizer.neutrino.get_megatron_neutrino_optimizer`: 2D matrix params and
    3D expert-batched params are Dion-managed; everything else is delegated to a chained external
    AdamW via :func:`get_megatron_optimizer`.
    """
    if config.use_distributed_optimizer:
        raise Exception(
            'dion with the standard distributed optimizer is not supported; '
            'use --use-layer-wise-distributed-optimizer to shard optimizer state.'
        )
    if config.fp16:
        raise Exception('dion with fp16 is not supported (use bf16 or fp32).')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up Dion optimizer with config {config}')

    base_overrides = (
        config_overrides if config_overrides is not None else get_standard_config_overrides(config)
    )

    def dion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if 'momentum' not in opt.state[p]:
                    opt.state[p]['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
                if 'Q' not in opt.state[p] and p.ndim in (2, 3):
                    n = p.shape[-1]
                    k = max(1, min(group['rank'], min(p.shape[-2], p.shape[-1])))
                    opt.state[p]['Q'] = torch.randn(n, k, device=p.device, dtype=torch.float32) / (n ** 0.5)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True

    linear_params: List[torch.nn.Parameter] = []
    nonlinear_params: List[torch.nn.Parameter] = []
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            is_emb = getattr(param, 'is_embedding_or_output_parameter', False)
            if len(param.shape) in (2, 3) and not is_emb:
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    dion_kwargs = dict(
        lr=(config.matrix_lr if config.matrix_lr is not None
            else config.muon_lr_factor * (config.lr or 0.0)),
        mu=config.dion_mu,
        weight_decay=config.weight_decay,
        rank=config.dion_rank,
        eps=config.dion_eps,
        dp_projection=config.dion_dp_projection,
        pg_collection=pg_collection,
        tp_mode='duplicated',
        orth=config.dion_orth,
    )

    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, base_overrides)
    if config.matrix_lr is not None or config.muon_lr_factor != 1.0:
        floor = config.min_lr if config.min_lr is not None else 0.0
        matrix_lr = dion_kwargs['lr']
        if config.min_lr_mode == 'absolute':
            matrix_min_lr = min(floor, matrix_lr)
        else:
            ratio = (floor / config.lr) if (config.lr and config.lr > 0) else 0.0
            matrix_min_lr = matrix_lr * ratio
        for group in linear_param_groups:
            if group.get('default_config', False):
                group['max_lr'] = matrix_lr
                group['min_lr'] = matrix_min_lr
                group['default_config'] = False

    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in list(linear_param_groups):
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = Dion(linear_param_groups, **dion_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            config.bf16 = False
            reset_config_bf16 = True
        else:
            optimizer = Float16OptimizerWithFloat16Params(optimizer, config, None, dion_init_state_fn)
    else:
        optimizer = FP32Optimizer(optimizer, config, dion_init_state_fn)

    optimizers = [optimizer]

    if len(expert_param_groups) > 0:
        expert_optimizer = Dion(expert_param_groups, **dion_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, dion_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, dion_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    prev_optimizer = config.optimizer
    config.optimizer = 'adam'
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=base_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    config.optimizer = prev_optimizer

    for param in linear_params:
        param.requires_grad = True

    init_fns = [dion_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Dion')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            model_chunks=model_chunks,
            async_allgather=config.overlap_param_gather,
        )
    return ChainedOptimizer(optimizers)
