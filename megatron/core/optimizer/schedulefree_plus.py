# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.distributed as dist

import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

try:
    import emerging_optimizers
    from emerging_optimizers.orthogonalized_optimizers import (
        get_muon_scale_factor as _emerging_get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
except ImportError:
    emerging_optimizers = None

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size
from .master import _get_muon_scale_factor, _split_qkv, _merge_qkv


class AdamCScheduleFreePlusPaper(torch.optim.Optimizer):
    r"""
    AdamC + Schedule-Free + Polyak (Reference / Research Implementation)
    Extended with Hypersphere Constraints and Orthogonal Updates (Muon-style).

    This code uses a non-standard optimizer step interface: it requires the
    rank-local function value to be passed into the step. This, together
    with gradient norm information, will be all-reduced to compute the
    Polyak step size.
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 sf_beta1: float = 0.9,
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0,
                 polyak_beta: float = 0,
                 c_warmup: int = 0,
                 sf_beta1_anneal_steps: int = 0,
                 sf_beta1_max: float = 0.965,
                 weight_lr_power: float = 2,
                 # Hypersphere options (L2, post-step weight projection only).
                 hypersphere_mode: Optional[str] = None,
                 hypersphere_embedding_mode: Optional[str] = None,
                 hypersphere_router_mode: Optional[str] = None,
                 hypersphere_eps: float = 1e-8,
                 hypersphere_tangential_grad: bool = False,
                 hypersphere_preserve_init: bool = False,
                 hypersphere_scale_out_proj_init: bool = False,
                 num_layers: Optional[int] = None,
                 # Muon (orthogonalized updates).
                 use_orthogonal_updates: bool = False,
                 momentum_beta: float = 0.95,
                 use_nesterov: bool = True,
                 split_qkv: bool = True,
                 qkv_split_shapes: Optional[tuple[int, int, int]] = None,
                 qkv_dim: Optional[int] = None,
                 is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
                 fp32_matmul_prec: str = "medium",
                 coefficient_type: str = "quintic",
                 num_ns_steps: int = 5,
                 scale_mode: str = "spectral",
                 extra_scale_factor: float = 1.0,
                 # NorMuon options
                 use_normuon: bool = False,
                 normuon_beta2: float = 0.95,
                 normuon_eps: float = 1e-8,
                 pg_collection: Optional[ProcessGroupCollection] = None,
                 tp_mode: str = "duplicated",
                 ):

        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov
        self.momentum_beta = momentum_beta

        self.use_normuon = use_normuon
        self.normuon_beta2 = normuon_beta2
        self.normuon_eps = normuon_eps

        self.hypersphere_mode = hypersphere_mode
        self.hypersphere_embedding_mode = hypersphere_embedding_mode
        self.hypersphere_router_mode = hypersphere_router_mode
        self.hypersphere_eps = hypersphere_eps
        self.hypersphere_tangential_grad = hypersphere_tangential_grad
        self.hypersphere_preserve_init = hypersphere_preserve_init
        if hypersphere_scale_out_proj_init:
            assert num_layers is not None and num_layers > 0, (
                "hypersphere_scale_out_proj_init=True requires num_layers"
            )
            self.out_proj_radius_scale = 1.0 / math.sqrt(2 * num_layers)
        else:
            self.out_proj_radius_scale = 1.0

        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn if is_qkv_fn is not None else (lambda p: False)
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_dim = qkv_dim

        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        defaults = dict(lr=lr,
                        betas=betas,
                        sf_beta1=sf_beta1,
                        eps=eps,
                        r=r,
                        k=0,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=eps,
                        scheduled_lr=0.0,
                        polyak_beta=polyak_beta,
                        sf_beta1_anneal_steps=sf_beta1_anneal_steps,
                        sf_beta1_max=sf_beta1_max,
                        grad_l1_ema=0.0,
                        polyak_denom_ema=0.0,
                        c_warmup=c_warmup,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        use_orthogonal_updates=use_orthogonal_updates)
        super().__init__(params, defaults)
        self.loss_val = None
        self.process_group = None
        self.grad_norm = None
        self.clip_grad = 0.0

        # Normalize parameters at init so the first forward sees on-sphere weights.
        if (not self.hypersphere_preserve_init
                and (self.hypersphere_mode is not None
                     or self.hypersphere_embedding_mode is not None
                     or self.hypersphere_router_mode is not None)):
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.ndim != 2:
                            continue
                        is_qkv = self.is_qkv_fn(p)
                        is_out_proj = getattr(p, "is_out_proj", False)
                        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                        is_router = getattr(p, "is_router", False)
                        self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router)

    def _init_group(self, group, skip_non_grad_params=False):
        for p in group['params']:
            if p.grad is None and skip_non_grad_params:
                continue
            state = self.state[p]
            if 'z' not in state:
                state['z'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                state['x'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                state['y'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p.detach(), memory_format=torch.preserve_format)
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                is_qkv = self.is_qkv_fn(p)
                is_out_proj = getattr(p, "is_out_proj", False)
                is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                is_router = getattr(p, "is_router", False)

                if (not self.hypersphere_preserve_init
                        and p.ndim == 2
                        and self._resolve_mode(is_embedding, is_router) is not None):
                    self._normalize(p, state['z'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)
                    self._normalize(p, state['x'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)
                    self._normalize(p, state['y'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)

    def _normuon_rescale(self, update, state):
        avg_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
        if "normuon_v" not in state:
            buf_shape = list(update.shape)
            buf_shape[avg_dim] = 1
            state["normuon_v"] = update.new_zeros(buf_shape)
        moment2 = state["normuon_v"]
        v_mean = update.square().mean(dim=avg_dim, keepdim=True)
        moment2.lerp_(v_mean, 1 - self.normuon_beta2)
        res = update * moment2.clamp_min(self.normuon_eps).rsqrt()

        vnorm_new = res.norm(dim=(-2, -1), keepdim=True).clamp_min(self.normuon_eps)
        shape_scaling = min(update.size(-2), update.size(-1)) ** 0.5
        res = res * shape_scaling / vnorm_new

        scaling_factor = _get_muon_scale_factor(update.size(-2), update.size(-1), mode=self.scale_mode)
        return res * scaling_factor

    def _orthogonalize_param(self, p, grad, is_qkv: bool = False):
        if self.pg_collection is not None:
            tp_group = (self.pg_collection.expt_tp
                        if getattr(p, "expert_tp", False)
                        else self.pg_collection.tp)
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and is_qkv:
            qs, ks, vs = _split_qkv(grad, self.qkv_split_shapes)
            qs = self._orthogonalize_tensor(qs, tp_group, partition_dim)
            ks = self._orthogonalize_tensor(ks, tp_group, partition_dim)
            vs = self._orthogonalize_tensor(vs, tp_group, partition_dim)
            return _merge_qkv((qs, ks, vs), grad.shape, self.qkv_split_shapes)
        return self._orthogonalize_tensor(grad, tp_group, partition_dim)

    def _orthogonalize_tensor(self, grad, tp_group, partition_dim):
        assert grad.ndim == 2
        size = [grad.size(-2), grad.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=("duplicated" if self.tp_mode == "blockwise" else self.tp_mode),
        )
        scale = _get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        return orth * scale * self.extra_scale_factor

    def _resolve_mode(self, is_embedding: bool, is_router: bool = False):
        if is_router and self.hypersphere_router_mode is not None:
            mode = self.hypersphere_router_mode
        elif is_embedding and self.hypersphere_embedding_mode is not None:
            mode = self.hypersphere_embedding_mode
        else:
            mode = self.hypersphere_mode
        return None if mode == "none" else mode

    def _resolve_radius_scale(self, is_out_proj: bool) -> float:
        if not is_out_proj or self.out_proj_radius_scale == 1.0:
            return 1.0
        return self.out_proj_radius_scale

    def _project_tangent_inplace(self, p, grad, is_qkv: bool = False, is_out_proj: bool = False,
                                  is_embedding: bool = False, is_router: bool = False):
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return
        if is_qkv and self.split_qkv:
            ps = _split_qkv(p, self.qkv_split_shapes)
            gs = _split_qkv(grad, self.qkv_split_shapes)
            for pi, gi in zip(ps, gs):
                self._project_tangent_single_(pi, gi, is_out_proj, mode)
            grad.copy_(_merge_qkv(gs, grad.size(), self.qkv_split_shapes))
            return
        self._project_tangent_single_(p, grad, is_out_proj, mode)

    def _project_tangent_single_(self, p, grad, is_out_proj: bool, mode: str):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        elif mode == "flat":
            dim = None
        else:
            return
        if dim is None:
            p_norm_sq = (p * p).sum().clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum() / p_norm_sq
        else:
            p_norm_sq = (p * p).sum(dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum(dim=dim, keepdim=True) / p_norm_sq
        grad.sub_(p * radial)

    def _normalize(self, p, x, is_qkv: bool = False, is_out_proj: bool = False,
                   is_embedding: bool = False, is_router: bool = False):
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return

        radius_scale = self._resolve_radius_scale(is_out_proj)

        if is_qkv and self.split_qkv:
            qs, ks, vs = _split_qkv(x, self.qkv_split_shapes)
            self._normalize_single(qs, is_out_proj, mode, radius_scale)
            self._normalize_single(ks, is_out_proj, mode, radius_scale)
            self._normalize_single(vs, is_out_proj, mode, radius_scale)
            x.copy_(_merge_qkv((qs, ks, vs), x.size(), self.qkv_split_shapes))
            return

        self._normalize_single(x, is_out_proj, mode, radius_scale)

    def _normalize_single(self, x, is_out_proj: bool, mode: str, radius_scale: float = 1.0):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "flat":
            dim = None
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        else:
            raise ValueError(f"Unsupported hypersphere mode: {mode}")
        norm = torch.norm(x, dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)
        x.div_(norm)
        if mode == "flat":
            x.mul_(max(x.size(-2), x.size(-1)) ** 0.5)
        if radius_scale != 1.0:
            x.mul_(radius_scale)

    @torch.no_grad()
    def eval(self):
        for gidx, group in enumerate(self.param_groups):
            train_mode = group['train_mode']

            if train_mode:
                for p in group['params']:
                    state = self.state[p]

                    if 'x' in state:
                        # Switch p to x
                        p.detach().copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for gidx, group in enumerate(self.param_groups):
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.detach().copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = self.loss_val
        
        if loss is None:
            # Fallback to 0.0 or raise warning if no loss value is set
            loss = 0.0

        return self.step_func(function_value=loss)

    @torch.no_grad()
    def step_func(self, function_value=None) -> Optional[float]:
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        grad_l1_ema = self.param_groups[0]['grad_l1_ema']
        k = self.param_groups[0]['k']
        beta1, beta2 = self.param_groups[0]['betas']
        eps = self.param_groups[0]['eps']
        polyak_beta = self.param_groups[0]['polyak_beta']
        sf_beta1 = self.param_groups[0]['sf_beta1']
        sf_beta1_max = self.param_groups[0]['sf_beta1_max']
        sf_beta1_anneal_steps = self.param_groups[0]['sf_beta1_anneal_steps']

        if sf_beta1_anneal_steps > 0:
            progress = min(k / sf_beta1_anneal_steps, 1.0)
            sf_beta1_k = 1 - math.exp(math.log(1 - sf_beta1) * (1 - progress) + math.log(1 - sf_beta1_max) * progress)
        else:
            sf_beta1_k = sf_beta1

        grad_l1_list = []
        ip_term_list = []

        is_distributed = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                # Detect DTensor-sharded gradients: only DTensors require an
                # all_reduce to obtain the global L1 / inner-product values.
                # Regular tensors (e.g. under DDP) are already replicated, so
                # the local values are already the global ones.
                if hasattr(grad, 'to_local'):
                    is_distributed = True
                    local_grad = grad.to_local()
                else:
                    local_grad = grad

                # 1. Compute LOCALLY to avoid network syncs inside the loop
                grad_l1_p = torch.linalg.vector_norm(local_grad, ord=1)
                grad_l1_list.append(grad_l1_p)

                if 'z' in state:
                    z = state['z']
                    x = state['x']
                    if hasattr(z, 'to_local'):
                        local_z = z.to_local()
                    else:
                        local_z = z
                    if hasattr(x, 'to_local'):
                        local_x = x.to_local()
                    else:
                        local_x = x
                    ip_term_p = sf_beta1_k * (local_grad.mul(local_z - local_x)).sum()
                    ip_term_list.append(ip_term_p)

        # Find device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        for group in self.param_groups:
            for p in group['params']:
                device = p.device
                break

        # 2. Stack and sum the local scalars into single tensors on this device
        local_grad_l1 = torch.stack(grad_l1_list).sum() if grad_l1_list else torch.tensor(0.0, device=device)
        local_ip_term = torch.stack(ip_term_list).sum() if ip_term_list else torch.tensor(0.0, device=device)

        # 3. Perform global all_reduce over the network to get the final
        # values. We use self.process_group (set from Megatron wrappers) to
        # correctly reduce over sharded parameter groups.
        pg = getattr(self, 'process_group', None)
        # Fallback for FSDP2/DTensor-based single-controller distributed setups
        if pg is None and is_distributed and dist.is_available() and dist.is_initialized():
            pg = dist.group.WORLD

        if pg is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_grad_l1, op=dist.ReduceOp.SUM, group=pg)
            dist.all_reduce(local_ip_term, op=dist.ReduceOp.SUM, group=pg)

        clip_coeff = 1.0
        grad_norm_l2 = getattr(self, 'grad_norm', None)
        clip_grad_val = getattr(self, 'clip_grad', 0.0)
        if grad_norm_l2 is not None and clip_grad_val > 0.0:
            if grad_norm_l2 > clip_grad_val:
                clip_coeff = clip_grad_val / (grad_norm_l2 + 1e-6)

        grad_l1 = local_grad_l1.item() / clip_coeff
        ip_term = local_ip_term.item() / clip_coeff

        if pg is not None and dist.is_available() and dist.is_initialized():
            dist_tensor = torch.zeros(1).cuda()
            dist_tensor[0] = function_value
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.AVG, group=pg)
            global_function_value = dist_tensor[0].item()
        else:
            global_function_value = float(function_value)

        grad_l1_ema = polyak_beta * (grad_l1_ema) + (1 - polyak_beta) * grad_l1 * math.sqrt(math.pi / 2)
        grad_l1_ema_corr = grad_l1_ema / (1 - polyak_beta ** (k + 1))

        polyak_lr = max(0, global_function_value + ip_term) / grad_l1_ema_corr

        for group in self.param_groups:
            eps = group['eps']
            lr = max(group['lr'], eps)
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            r = group['r']
            weight_lr_power = group['weight_lr_power']
            c_warmup = group['c_warmup']

            bias_correction1 = 1 - beta1 ** (k + 1)
            bias_correction2 = 1 - beta2 ** (k + 1)

            # Apply any warmup that's part of the lr sequence
            group_lr = lr * polyak_lr

            # For plotting / tracking
            group['grad_l1_ema'] = grad_l1_ema
            group['grad_l1_ema_corr'] = grad_l1_ema_corr
            group['function_value_ema'] = global_function_value + ip_term
            group['ip_term'] = ip_term
            group['scheduled_lr'] = group_lr  # For logging purposes

            lr_max = group['lr_max'] = max(group_lr, group['lr_max'])

            if k < c_warmup:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                ckp1 = weight / weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                is_qkv = self.is_qkv_fn(p)
                is_out_proj = getattr(p, "is_out_proj", False)
                is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                is_router = getattr(p, "is_router", False)

                if 'z' not in state:
                    state['z'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p.detach(), memory_format=torch.preserve_format)

                    state['exp_avg_sq'] = torch.zeros_like(p.detach(), memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    if (not self.hypersphere_preserve_init
                            and p.ndim == 2
                            and self._resolve_mode(is_embedding, is_router) is not None):
                        self._normalize(p, state['z'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router)
                        self._normalize(p, state['x'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router)
                        self._normalize(p, state['y'], is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                z = state['z']
                x = state['x']
                y = state['y']

                # Tangent gradient projection
                if self.hypersphere_tangential_grad and p.ndim == 2:
                    self._project_tangent_inplace(y, grad, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                                   is_embedding=is_embedding, is_router=is_router)

                # Step updates
                if group.get('use_orthogonal_updates', False) and p.ndim == 2:
                    # Weight decay
                    if decay != 0:
                        z.add_(z, alpha=-decay * group_lr)

                    # Momentum
                    exp_avg.lerp_(grad, 1 - self.momentum_beta)
                    if self.use_nesterov:
                        update_grad = grad.lerp(exp_avg, self.momentum_beta)
                    else:
                        update_grad = exp_avg

                    # Orthogonal updates
                    with emerging_optimizers.utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        update = self._orthogonalize_param(p, update_grad, is_qkv=is_qkv)

                    if self.use_normuon:
                        update = self._normuon_rescale(update, state)

                    radius_scale = self._resolve_radius_scale(is_out_proj)
                    if radius_scale != 1.0:
                        update = update * radius_scale

                    z.add_(update, alpha=-group_lr)
                else:
                    # Weight decay (AdamC-style lr**2 scaling)
                    z.sub_(y, alpha=group_lr * group_lr * decay)

                    # Momentum
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_corr = exp_avg.div(bias_correction1)

                    # Decay the first and second moment running average coefficient
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    z.addcdiv_(exp_avg_corr, denom, value=-group_lr)

                # Normalize z
                if p.ndim == 2 and self._resolve_mode(is_embedding, is_router) is not None:
                    self._normalize(p, z, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)

                ### Take step
                x.mul_(1 - ckp1).add_(z, alpha=ckp1)

                # Normalize x
                if p.ndim == 2 and self._resolve_mode(is_embedding, is_router) is not None:
                    self._normalize(p, x, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)

                # Compute y
                y.copy_(x.mul(sf_beta1_k).add_(z, alpha=1 - sf_beta1_k))

                # Normalize y
                if p.ndim == 2 and self._resolve_mode(is_embedding, is_router) is not None:
                    self._normalize(p, y, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                    is_embedding=is_embedding, is_router=is_router)

                p.detach().copy_(y)

            group['k'] = k + 1
        self.grad_norm = None
        return function_value
