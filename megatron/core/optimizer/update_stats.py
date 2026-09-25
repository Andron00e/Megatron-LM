# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Optimizer-agnostic update statistics, exact under any TP/PP/EP/DP layout.

On a logging step the collector reads, around ``optimizer.step()``, the weights ``W_{t-1}``
and ``W_t``, the finalized gradient ``G_t`` the optimizer sees (``main_grad`` after the DP
reduction, before clipping) and the optimizer's momentum ``M_{t-1}``. Every metric is a ratio
of *additive* sums (``sum ||dW||^2``, ``sum ||W||^2``, ``sum <G, W>``, ...) accumulated per
(family, layer) bucket, i.e. it is the metric of the bucket's weights concatenated into one
vector. Each rank adds only what it owns:

* a parameter counts on the rank that steps it: under the layer-wise optimizer every DP rank
  holds its own share of the parameter list, otherwise data-parallel rank 0 of the parameter's
  DP group (``dp_cp`` for dense, expert-DP for experts) counts it;
* all TP shards of a matrix count (their sums add up to the full matrix); a parameter replicated
  over TP counts on TP rank 0 only; a pipeline-tied copy (``param.shared``) does not count.

The sums go into one fp64 buffer that is reduced with a single SUM all-reduce over the world
(plus a MAX all-reduce for the per-row maxima). Quantities that are not additive over shards
(row norms of row-parallel matrices, partial products for dY, Gram matrices) are first
all-reduced over the TP group, which needs every TP peer to own the same TP-sharded parameters;
this is checked once at setup.

``W`` is the optimizer's fp32 main parameter on the owner, so the realized step includes any
re-projection (MDDecoupling's hypersphere), weight decay and gain folding, and does not depend
on when the other DP ranks receive the updated weights (``--overlap-param-gather``).

Metrics (keys ``update/<metric>/<family>``, and ``update/layers/<L>/<metric>/<family>``):

``relative``                   ||dW|| / ||W_{t-1}||
``per-neuron-relative``        mean and max over output rows i of ||dW_i|| / ||W_{t-1,i}||
``angle-cos``, ``angle-deg``   angle between W_t and W_{t-1}; the sine comes from the Lagrange
                               identity ||W||^2 ||dW||^2 - <W, dW>^2, so small angles are exact
``grad-weight-cos``            cos(G, W_{t-1}); ``grad-radial`` = <G, W>/||W||,
                               ``grad-tangential`` = sqrt(||G||^2 - radial^2),
                               ``grad-tangential-fraction`` = tangential / ||G||
``delta-row-norm``, ``delta-col-norm``
                               mean/std of the row (column) norms of dW, pooled over the bucket.
                               Alex Hagele's ``update_steps`` logged the pre-lr update direction;
                               this is the realized, lr-scaled step.
``grad-momentum-cos``          cos(G_t, M_{t-1}); ``momentum-grad-norm-ratio`` = ||M||/||G||. M is
                               the first-moment buffer before the step: Neutrino and Muon
                               ``momentum_buffer`` (Neutrino's error-feedback buffer is not part
                               of M), Dion ``momentum``, MDDecoupling/NeutrinoMD and Adam
                               ``exp_avg``, all EMAs with weight 1 - beta on the new gradient.
``delta-y``                    ||X dW^T|| / ||X W_{t-1}^T|| over a fixed token subsample X of
                               the layer input of the step's last microbatch (after the fused
                               norm of LayerNorm-linear modules)
``stable-rank``                ||W||_F^2 / sigma_max^2 and ``momentum-r-eff`` = ||M||_*^2/||M||_F^2
                               (the r_eff of ``Neutrino._r_eff_for``), from exact eigenvalues of
                               the Gram matrix, averaged over matrices (every expert is one)
``grad-norm``, ``weight-norm`` Frobenius norms of the bucket
``phase-step``, ``phase-index``, ``phase-rl``
                               steps since the job start or the last :func:`notify_phase_change`,
                               number of phase changes in this job, and 1 while the phase is "rl"
``bf16-snapshot-floor``        ||W_{t-1} - bf16(W_{t-1})|| / ||W_{t-1}||, only with a bf16
                               snapshot: dW is biased by up to this amount

Families are those of :mod:`muon_logging`; parameters stepped by AdamW (the chained AdamW of the
Muon-type optimizers, or MDDecoupling's Adam branch) get the suffix ``-adam``. 1D parameters
other than norm scales are skipped. For routed experts, per-expert sums are also placed at the
global expert id ``ep_rank * num_local_experts + i`` of a ``[num_experts]`` vector per MoE layer
and summarized under ``update/experts/<family>/<metric>/{cv,min,max[,argmin,argmax]}``, with
histograms on the spectral cadence.

Not supported: the standard DistributedOptimizer (grads and main params are 1D shards of the
flat buffer; the collector disables itself with a warning), and gradient/momentum metrics of
parameters whose DP reduction is replaced by a sketch (Neutrino ``--neutrino-dp-projection``):
their ``main_grad`` is rank-local, so they contribute weight and update sums only. dY skips
routed experts (grouped GEMMs, no per-expert input) and the router.
"""

import importlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.utils import get_pg_rank, get_pg_size

from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .muon_logging import _GAIN_FAMILIES, _gain_log_family

logger = logging.getLogger(__name__)

_FAMILIES = _GAIN_FAMILIES + tuple(f"{family}-adam" for family in _GAIN_FAMILIES)
_EXPERT_FAMILIES = ("expert-in", "expert-out")
_DELTA_Y_FAMILIES = frozenset(
    ("attention-in", "attention-out", "dense-mlp-in", "dense-mlp-out", "moe-latent-in", "moe-latent-out")
)
_DELTA_Y_TOKENS = 128
_CHUNK_NUMEL = 1 << 24
_GRAM_BATCH_ELEMENTS = 1 << 27

# Columns of the (scope, family) buffer; every column is a plain sum.
(
    _A,  # ||W_{t-1}||^2
    _C,  # <W_{t-1}, dW>
    _D,  # ||dW||^2
    _GG,  # ||G||^2
    _GW,  # <G, W_{t-1}>
    _MM,  # ||M_{t-1}||^2
    _GM,  # <G, M_{t-1}>
    _GGM,  # ||G||^2 of the params that have a momentum buffer
    _SNAP_ERR,  # ||W_{t-1} - snapshot||^2
    _ROW_RATIO,  # sum over rows with W_i != 0 of ||dW_i|| / ||W_i||
    _ROW_RATIO_N,
    _ROW_D,  # sum over rows of ||dW_i||
    _ROW_D2,
    _ROW_N,
    _COL_D,
    _COL_D2,
    _COL_N,
    _DY_NUM,
    _DY_DEN,
    _SRANK,
    _SRANK_N,
    _REFF,
    _REFF_N,
    _PARAMS,
    _N_STATS,
) = range(25)

# Columns of the per-expert buffer [layer, in/out, expert, stat].
_E_A, _E_D, _E_GG, _E_GW, _E_MM, _E_GM, _N_EXPERT_STATS = range(7)
_EXPERT_COLUMNS = ((_A, _E_A), (_D, _E_D), (_GG, _E_GG), (_GW, _E_GW), (_MM, _E_MM), (_GM, _E_GM))


def _chunks(tensor: torch.Tensor):
    """Slices along dim 0 that bound the temporaries of a large parameter to ``_CHUNK_NUMEL``."""
    if tensor.ndim == 0 or tensor.numel() <= _CHUNK_NUMEL:
        yield slice(None)
        return
    step = max(1, _CHUNK_NUMEL // (tensor.numel() // tensor.shape[0]))
    for start in range(0, tensor.shape[0], step):
        yield slice(start, start + step)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-matrix inner products: shape [E] for a 3D expert tensor, [1] otherwise."""
    rows = a.shape[0] if a.ndim == 3 else 1
    return (a * b).reshape(rows, -1).sum(dim=1, dtype=torch.float64)


def _combine(values: List[torch.Tensor], per_expert: bool) -> torch.Tensor:
    """Chunks of a 3D tensor are different experts; chunks of a matrix add up."""
    return torch.cat(values) if per_expert else torch.stack(values).sum(0)


@dataclass(eq=False)
class _Entry:
    """One parameter, or the local experts of one grouped GEMM stacked into [E, out, in]."""

    name: str
    model_params: List[torch.Tensor]
    main_params: List[torch.Tensor]
    optimizer: torch.optim.Optimizer
    buckets: List[int]  # flat (scope, family) indices
    family: str
    tp_group: Optional[torch.distributed.ProcessGroup]  # set only for TP-sharded params
    tp_rank: int
    partition_dim: Optional[int]  # set only for TP-sharded params
    grad_ok: bool
    row_stats: bool
    expert_slots: Optional[List[int]] = None  # flat indices of stat 0 in the expert buffer
    # per-step state
    snapshot: Optional[torch.Tensor] = None
    row_w2: Optional[torch.Tensor] = None
    sums: Dict[int, torch.Tensor] = field(default_factory=dict)
    spectral: Dict[int, tuple] = field(default_factory=dict)

    def weight(self) -> torch.Tensor:
        if len(self.main_params) == 1:
            return self.main_params[0].detach()
        return torch.stack([param.detach() for param in self.main_params])

    def grad(self) -> Optional[torch.Tensor]:
        grads = [
            getattr(model, "main_grad", None) if getattr(model, "main_grad", None) is not None
            else main.grad
            for model, main in zip(self.model_params, self.main_params)
        ]
        if any(grad is None for grad in grads):
            return None
        return grads[0] if len(grads) == 1 else torch.stack(grads)

    def momentum(self) -> Optional[torch.Tensor]:
        buffers = [_momentum(self.optimizer, main) for main in self.main_params]
        if any(buffer is None for buffer in buffers):
            return None
        return buffers[0] if len(buffers) == 1 else torch.stack(buffers)

    @property
    def rows_split(self) -> bool:
        return self.row_stats and self.partition_dim == 1

    @property
    def cols_split(self) -> bool:
        return self.row_stats and self.partition_dim == 0

    def counts_rows(self) -> bool:
        return not self.rows_split or self.tp_rank == 0

    def counts_cols(self) -> bool:
        return not self.cols_split or self.tp_rank == 0


class _Accumulator:
    """(flat index, value) pairs added into a buffer with one ``index_add_``.

    Indices stay on the host until the flush so accumulating never copies host to device.
    """

    def __init__(self, device):
        self.device = device
        self.indices: List[int] = []
        self.values: List[torch.Tensor] = []
        self.constants: Dict[int, float] = {}

    def add(self, buckets: List[int], stat: int, value: torch.Tensor) -> None:
        value = value.reshape(1).to(self.device, torch.float64)
        for bucket in buckets:
            self.indices.append(bucket * _N_STATS + stat)
            self.values.append(value)

    def add_constant(self, buckets: List[int], stat: int, value: float) -> None:
        for bucket in buckets:
            index = bucket * _N_STATS + stat
            self.constants[index] = self.constants.get(index, 0.0) + value

    def add_vector(self, indices: List[int], values: torch.Tensor) -> None:
        self.indices.extend(indices)
        self.values.append(values.reshape(-1).to(self.device, torch.float64))

    def flush(self, buffer: torch.Tensor) -> None:
        indices = self.indices + list(self.constants)
        if not indices:
            return
        values = self.values + [
            torch.tensor(list(self.constants.values()), dtype=torch.float64)
        ]
        buffer.index_add_(
            0,
            torch.tensor(indices).to(buffer.device, non_blocking=True),
            torch.cat([v.to(buffer.device, non_blocking=True) for v in values]),
        )


class UpdateStatsCollector:
    """Collects the metrics of the module docstring around ``optimizer.step()``.

    Call :meth:`begin_step` before the forward pass of every step, and
    :meth:`before_optimizer_step` / :meth:`after_optimizer_step` around the optimizer step.
    Only logging steps do any work.
    """

    def __init__(
        self,
        model_chunks,
        optimizer,
        *,
        interval: int,
        per_layer: bool = False,
        dense_window: int = 20,
        spectral_interval: int = 0,
        delta_y: bool = False,
        snapshot_dtype: str = "fp32",
        num_layers: int = 0,
        num_experts: Optional[int] = None,
    ):
        assert interval > 0, interval
        assert snapshot_dtype in ("fp32", "bf16"), snapshot_dtype
        self.interval = interval
        self.per_layer = per_layer
        self.dense_window = dense_window
        self.spectral_interval = spectral_interval
        self.snapshot_dtype = torch.float32 if snapshot_dtype == "fp32" else torch.bfloat16
        self.num_layers = num_layers
        self.num_experts = num_experts or 0
        self.num_scopes = 1 + (num_layers if per_layer else 0)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._phase_start: Optional[int] = None
        self._phase_index = 0
        self._phase_name = "ntp"
        self._warned_offloaded = False
        self._spectral_pending = False
        self._step = 0
        self._active = False
        self._spectral = False
        self._capture = False
        self._inputs: Dict[torch.nn.Module, torch.Tensor] = {}
        self._delta_y_modules = []
        self._hooks = []

        self._distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.entries: List[_Entry] = []
        self.enabled = self._discover(model_chunks, optimizer)
        if self.enabled:
            self._check_tp_consistency()
            if delta_y:
                self._register_delta_y_hooks(model_chunks)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _tp_groups(self):
        if not self._distributed:
            return []
        return [
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_expert_tensor_parallel_group(),
        ]

    def _groups(self, expert: bool):
        if not self._distributed:
            return None, None
        if expert:
            return (
                parallel_state.get_expert_tensor_parallel_group(),
                parallel_state.get_expert_data_parallel_group(),
            )
        return (
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    def _discover(self, model_chunks, optimizer) -> bool:
        from .distrib_optimizer import DistributedOptimizer

        wrappers = getattr(optimizer, "chained_optimizers", [optimizer])
        if any(isinstance(wrapper, DistributedOptimizer) for wrapper in wrappers):
            logger.warning(
                "update stats: the standard DistributedOptimizer keeps 1D shards of grads and "
                "main params; update statistics are disabled."
            )
            return False
        owner = {}
        for wrapper in wrappers:
            inner = getattr(wrapper, "optimizer", None)
            for group in inner.param_groups if inner is not None else ():
                for param in group["params"]:
                    owner[param] = (inner, group)
        layer_wise = isinstance(optimizer, LayerWiseDistributedOptimizer)
        matrix_types = _matrix_optimizer_types()
        try:
            from megatron.core.distributed.param_and_grad_buffer import DP_PROJECTED_ATTR
        except ImportError:
            DP_PROJECTED_ATTR = None
        ep_rank, ep_size = 0, 1
        if self._distributed:
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            ep_size = parallel_state.get_expert_model_parallel_world_size()

        for chunk in model_chunks:
            modules = dict(chunk.named_modules())
            for name, param in chunk.named_parameters():
                if not param.requires_grad or not param_is_not_shared(param):
                    continue
                main = getattr(param, "main_param", param)
                if main not in owner:
                    continue
                inner, group = owner[main]
                family = _family(name, param)
                if param.ndim < 2 and family != "layernorm":
                    continue
                # DDP's rule: experts reduce over the expert-DP group only when EP > 1.
                expert = not getattr(param, "allreduce", True)
                tp_group, dp_group = self._groups(expert)
                if not layer_wise and dp_group is not None and get_pg_rank(dp_group) != 0:
                    continue
                tp_rank = get_pg_rank(tp_group) if tp_group is not None else 0
                tp_sharded = (
                    _is_tp_sharded(name, param)
                    and tp_group is not None
                    and get_pg_size(tp_group) > 1
                )
                if not tp_sharded and tp_rank != 0:
                    continue
                if not _is_matrix_step(inner, group, matrix_types):
                    family = f"{family}-adam"
                layer = _layer(name, param, modules)
                if layer is not None and not 0 <= layer < self.num_layers:
                    layer = None
                grad_ok = not (DP_PROJECTED_ATTR and getattr(param, DP_PROJECTED_ATTR, False))
                entry = _Entry(
                    name=name,
                    model_params=[param],
                    main_params=[main],
                    optimizer=inner,
                    buckets=self._buckets(family, layer),
                    family=family,
                    tp_group=tp_group if tp_sharded else None,
                    tp_rank=tp_rank,
                    partition_dim=getattr(param, "partition_dim", None) if tp_sharded else None,
                    grad_ok=grad_ok,
                    row_stats=param.ndim == 2 or (param.ndim == 3 and not tp_sharded),
                )
                if layer is not None:
                    entry.expert_slots = self._expert_slots(
                        name, param, modules, family, layer, ep_rank, ep_size
                    )
                self.entries.append(entry)
        self.entries = _stack_grouped_experts(self.entries)
        if not all(entry.grad_ok for entry in self.entries):
            logger.warning(
                "update stats: parameters with a sketched DP reduction (dp-projection) have a "
                "rank-local main_grad; their gradient and momentum metrics are skipped."
            )
        return True

    def _buckets(self, family: str, layer: Optional[int]) -> List[int]:
        family_index = _FAMILIES.index(family)
        buckets = [family_index]
        if self.per_layer and layer is not None:
            buckets.append((layer + 1) * len(_FAMILIES) + family_index)
        return buckets

    def _expert_slots(self, name, param, modules, family, layer, ep_rank, ep_size):
        base = family.removesuffix("-adam")
        if base not in _EXPERT_FAMILIES:
            return None
        grouped = re.search(r"weight(\d+)$", name)
        sequential = re.search(r"\.local_experts\.(\d+)\.", name)
        if grouped is not None:  # TE grouped GEMM: one [out, in] param per local expert
            num_local = getattr(modules.get(name.rpartition(".")[0]), "num_gemms", None)
            local = [int(grouped.group(1))]
        elif sequential is not None:  # SequentialMLP: one MLP module per local expert
            num_local = getattr(modules.get(name[: sequential.start()]), "num_local_experts", None)
            local = [int(sequential.group(1))]
        elif param.ndim == 3:  # merged [E, out, in]
            num_local = param.shape[0]
            local = list(range(num_local))
        else:
            return None
        if num_local is None or num_local * ep_size != self.num_experts:
            return None
        io = _EXPERT_FAMILIES.index(base)
        return [
            ((layer * 2 + io) * self.num_experts + ep_rank * num_local + i) * _N_EXPERT_STATS
            for i in local
        ]

    def _check_tp_consistency(self) -> None:
        """TP collectives need every TP peer to own the same TP-sharded parameters."""
        for group in self._tp_groups():
            if get_pg_size(group) == 1:
                continue
            mine = [e for e in self.entries if e.tp_group is group]
            signature = torch.tensor(
                [len(mine), sum(p.numel() for e in mine for p in e.main_params)],
                dtype=torch.int64,
                device=self.device,
            )
            low, high = signature.clone(), signature.clone()
            torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN, group=group)
            torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX, group=group)
            assert torch.equal(low, high), (
                "update stats: TP peers own different TP-sharded parameters "
                f"({low.tolist()} vs {high.tolist()})"
            )

    def _register_delta_y_hooks(self, model_chunks) -> None:
        entries = {id(e.model_params[0]): e for e in self.entries if len(e.model_params) == 1}
        for chunk in model_chunks:
            for module in chunk.modules():
                weight = getattr(module, "weight", None)
                entry = entries.get(id(weight)) if isinstance(weight, torch.Tensor) else None
                family = entry.family.removesuffix("-adam") if entry is not None else None
                if family not in _DELTA_Y_FAMILIES or weight.ndim != 2:
                    continue
                # Column-parallel inputs under sequence parallelism are sequence shards.
                gather = entry.partition_dim == 0 and getattr(module, "sequence_parallel", False)
                self._delta_y_modules.append((module, entry, gather))
                self._hooks.append(module.register_forward_pre_hook(self._capture_input))

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    @torch.no_grad()
    def _capture_input(self, module, args):
        if not self._capture or not args or not isinstance(args[0], torch.Tensor):
            return
        x = args[0].detach()
        x = x.reshape(-1, x.shape[-1])
        count = min(_DELTA_Y_TOKENS, x.shape[0])
        index = torch.linspace(0, x.shape[0] - 1, count, device=x.device).round().long()
        x = x.index_select(0, index).float()
        norm_weight = getattr(module, "layer_norm_weight", None)
        if norm_weight is not None:
            scale = norm_weight.float() + float(getattr(module, "zero_centered_gamma", False))
            eps = getattr(module, "eps", 1e-5)
            if getattr(module, "normalization", "LayerNorm") == "RMSNorm":
                x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * scale
            else:
                x = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps) * scale
                bias = getattr(module, "layer_norm_bias", None)
                if bias is not None:
                    x = x + bias.float()
        self._inputs[module] = x

    # ------------------------------------------------------------------
    # cadence
    # ------------------------------------------------------------------
    def notify_phase_change(self, name: str) -> None:
        """Restart the dense window at the next step and take spectra there (e.g. NTP <-> RL)."""
        logger.info(f"update stats: phase change to {name!r}")
        self._phase_start = None
        self._phase_index += 1
        self._phase_name = name
        self._spectral_pending = True

    def set_phase(self, name: str) -> None:
        """Name the current phase without starting a new window (logged as ``update/phase-rl``)."""
        self._phase_name = name

    def begin_step(self, iteration: int) -> bool:
        """``iteration`` is the number of completed steps; returns whether this step logs."""
        if not self.enabled:
            return False
        if self._phase_start is None:
            self._phase_start = iteration
        self._step = iteration + 1
        self._spectral = self._spectral_pending or (
            self.spectral_interval > 0 and self._step % self.spectral_interval == 0
        )
        self._active = (
            self._spectral
            or self._step % self.interval == 0
            or self._step - self._phase_start <= self.dense_window
        )
        if self._spectral:
            self._spectral_pending = False
        # dY follows the spectral cadence when there is one.
        self._capture = (
            bool(self._delta_y_modules)
            and self._active
            and (self._spectral or self.spectral_interval <= 0)
        )
        self._inputs.clear()
        return self._active

    # ------------------------------------------------------------------
    # pre-step: W_{t-1}, G_t, M_{t-1}
    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_optimizer_step(self) -> None:
        if not self._active:
            return
        self._capture = False
        if self._offloaded():
            self._active = False
            return
        for entry in self.entries:
            self._pre_step(entry)
        if self._spectral:
            self._spectra()

    def _offloaded(self) -> bool:
        """Main params or optimizer state still on the host (e.g. RL's
        --rl-offload-optimizer-during-inference between restore points): skip this step. The
        decision is all-reduced so that no rank enters the step's collectives alone."""
        local = self._offloaded_local()
        if self._distributed:
            flag = torch.tensor([float(local)], device=self.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())
        return local

    def _offloaded_local(self) -> bool:
        for entry in self.entries:
            momentum = entry.momentum() if entry.grad_ok else None
            if entry.main_params[0].device != self.device or (
                momentum is not None and momentum.device != self.device
            ):
                if not self._warned_offloaded:
                    logger.warning(
                        f"update stats: {entry.name} or its optimizer state is not on "
                        f"{self.device} (offloaded); skipping update statistics for this step."
                    )
                    self._warned_offloaded = True
                return True
        return False

    def _pre_step(self, entry: _Entry) -> None:
        weight = entry.weight()
        grad = entry.grad() if entry.grad_ok else None
        momentum = entry.momentum() if grad is not None else None
        entry.snapshot = torch.empty_like(weight, dtype=self.snapshot_dtype)
        sums = {stat: [] for stat in (_A, _GG, _GW, _MM, _GM, _SNAP_ERR)}
        rows = []
        for sl in _chunks(weight):
            w = weight[sl].float()
            entry.snapshot[sl].copy_(w)
            sums[_A].append(_dot(w, w))
            if self.snapshot_dtype != torch.float32:
                err = w - entry.snapshot[sl].float()
                sums[_SNAP_ERR].append(_dot(err, err))
            if entry.row_stats:
                rows.append(w.square().sum(dim=-1, dtype=torch.float64).reshape(-1))
            if grad is not None:
                g = grad[sl].float()
                sums[_GG].append(_dot(g, g))
                sums[_GW].append(_dot(g, w))
                if momentum is not None:
                    m = momentum[sl].float()
                    sums[_MM].append(_dot(m, m))
                    sums[_GM].append(_dot(g, m))
        entry.sums = {stat: _combine(v, weight.ndim == 3) for stat, v in sums.items() if v}
        entry.row_w2 = torch.cat(rows) if rows else None

    def _spectra(self) -> None:
        """Stable rank of W_{t-1} and r_eff of M_{t-1} from exact Gram eigenvalues.

        The Gram matrix over the dimension a TP shard holds completely (W^T W for output-sharded,
        W W^T for input-sharded matrices; the smaller one if unsharded) is additive over the TP
        shards, so one TP all-reduce of a [d, d] matrix gives the exact spectrum of the full
        matrix; d is the hidden size for the Megatron linears. Power iteration would need a
        collective per iteration and cannot give the nuclear norm that r_eff needs.
        Eigendecompositions are batched over matrices of the same size (every expert is one).
        """
        pending: Dict[int, list] = {}

        def flush(dim):
            items = pending.pop(dim, [])
            if not items:
                return
            eig = torch.linalg.eigvalsh(torch.cat([g for *_, g in items]).double()).clamp_min(0)
            offset = 0
            for entry, stat, gram in items:
                entry.spectral[stat] = _spectral_sums(eig[offset : offset + gram.shape[0]], stat)
                offset += gram.shape[0]

        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for entry in self.entries:
                entry.spectral = {}
                weight = entry.weight()
                if weight.ndim < 2 or (weight.ndim == 3 and entry.tp_group is not None):
                    continue
                sources = [(_SRANK, weight)]
                momentum = entry.momentum() if entry.grad_ok else None
                if momentum is not None:
                    sources.append((_REFF, momentum))
                for stat, tensor in sources:
                    gram = _gram(tensor.float(), entry.partition_dim)
                    if entry.tp_group is not None:
                        torch.distributed.all_reduce(gram, group=entry.tp_group)
                        if entry.tp_rank != 0:
                            continue
                    dim = gram.shape[-1]
                    pending.setdefault(dim, []).append((entry, stat, gram.reshape(-1, dim, dim)))
                    if sum(g.numel() for *_, g in pending[dim]) >= _GRAM_BATCH_ELEMENTS:
                        flush(dim)
            for dim in list(pending):
                flush(dim)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # ------------------------------------------------------------------
    # post-step: dW, reductions, formatting
    # ------------------------------------------------------------------
    @torch.no_grad()
    def after_optimizer_step(self, update_successful: bool = True) -> Dict:
        if not self._active:
            return {}
        self._active = False
        if not update_successful:
            for entry in self.entries:
                entry.snapshot = entry.row_w2 = None
                entry.sums, entry.spectral = {}, {}
            self._inputs.clear()
            return {}
        tp_vectors: Dict[int, list] = {}
        row_vectors = [self._post_step(entry, tp_vectors) for entry in self.entries]
        delta_y = self._delta_y_partials(tp_vectors) if self._inputs else []
        reduced = _tp_all_reduce(tp_vectors)

        buffer = torch.zeros(
            self.num_scopes * len(_FAMILIES) * _N_STATS, dtype=torch.float64, device=self.device
        )
        expert_numel = self.num_layers * 2 * self.num_experts * _N_EXPERT_STATS
        expert_buffer = torch.zeros(expert_numel, dtype=torch.float64, device=self.device)
        acc, expert_acc = _Accumulator(self.device), _Accumulator(self.device)
        row_max_indices, row_max_values = [], []
        for entry, (d2_rows, d2_cols) in zip(self.entries, row_vectors):
            self._accumulate(entry, acc, expert_acc)
            if d2_rows is not None:
                ratio_max = self._accumulate_rows(entry, d2_rows, d2_cols, reduced, acc)
                if ratio_max is not None:
                    row_max_indices.extend(entry.buckets)
                    row_max_values.extend([ratio_max] * len(entry.buckets))
            entry.snapshot = entry.row_w2 = None
            entry.sums, entry.spectral = {}, {}
        for entry, num, den in delta_y:
            if entry.rows_split:
                if entry.tp_rank != 0:
                    continue
                num = reduced[id(num)].square().sum(dtype=torch.float64)
                den = reduced[id(den)].square().sum(dtype=torch.float64)
            acc.add(entry.buckets, _DY_NUM, num)
            acc.add(entry.buckets, _DY_DEN, den)
        self._inputs.clear()

        acc.flush(buffer)
        expert_acc.flush(expert_buffer)
        row_max = torch.zeros(self.num_scopes * len(_FAMILIES), dtype=torch.float64, device=self.device)
        if row_max_values:
            row_max.scatter_reduce_(
                0,
                torch.tensor(row_max_indices).to(self.device, non_blocking=True),
                torch.stack(row_max_values).to(self.device, torch.float64),
                reduce="amax",
            )
        flat = torch.cat((buffer, expert_buffer))
        if self._distributed:
            torch.distributed.all_reduce(flat)
            torch.distributed.all_reduce(row_max, op=torch.distributed.ReduceOp.MAX)
        flat = flat.cpu()
        stats = self._format(
            flat[: buffer.numel()].view(self.num_scopes, len(_FAMILIES), _N_STATS).tolist(),
            row_max.view(self.num_scopes, len(_FAMILIES)).tolist(),
        )
        if expert_numel:
            self._format_experts(
                stats,
                flat[buffer.numel():].view(self.num_layers, 2, self.num_experts, _N_EXPERT_STATS),
            )
        stats["update/phase-step"] = self._step - self._phase_start
        stats["update/phase-index"] = self._phase_index
        stats["update/phase-rl"] = float(self._phase_name == "rl")
        return stats

    def _post_step(self, entry: _Entry, tp_vectors):
        weight = entry.weight()
        is_3d = weight.ndim == 3
        d_sums, c_sums, rows, cols = [], [], [], []
        for sl in _chunks(weight):
            prev = entry.snapshot[sl].float()
            delta = weight[sl].float() - prev
            d_sums.append(_dot(delta, delta))
            c_sums.append(_dot(prev, delta))
            if entry.row_stats:
                square = delta.square()
                rows.append(square.sum(dim=-1, dtype=torch.float64).reshape(-1))
                col = square.sum(dim=-2, dtype=torch.float64)
                cols.append(col.reshape(-1) if is_3d else col)
        entry.sums[_D] = _combine(d_sums, is_3d)
        entry.sums[_C] = _combine(c_sums, is_3d)
        if not entry.row_stats:
            return None, None
        d2_rows = torch.cat(rows)
        d2_cols = torch.cat(cols) if is_3d else torch.stack(cols).sum(0)
        if entry.rows_split:
            tp_vectors.setdefault(id(entry.tp_group), [entry.tp_group]).extend((entry.row_w2, d2_rows))
        if entry.cols_split:
            tp_vectors.setdefault(id(entry.tp_group), [entry.tp_group]).append(d2_cols)
        return d2_rows, d2_cols

    def _accumulate(self, entry: _Entry, acc: _Accumulator, expert_acc: _Accumulator) -> None:
        for stat, value in entry.sums.items():
            acc.add(entry.buckets, stat, value.sum())
        if _MM in entry.sums:
            acc.add(entry.buckets, _GGM, entry.sums[_GG].sum())
        acc.add_constant(entry.buckets, _PARAMS, 1.0)
        for stat, (total, count) in entry.spectral.items():
            acc.add(entry.buckets, stat, total)
            acc.add(entry.buckets, stat + 1, count)  # _SRANK_N / _REFF_N
        if entry.expert_slots is not None:
            for stat, expert_stat in _EXPERT_COLUMNS:
                if stat in entry.sums:
                    expert_acc.add_vector(
                        [slot + expert_stat for slot in entry.expert_slots], entry.sums[stat]
                    )

    def _accumulate_rows(self, entry, d2_rows, d2_cols, reduced, acc) -> Optional[torch.Tensor]:
        """Per-row and per-column statistics, counted once per full row/column."""
        ratio_max = None
        if entry.counts_rows():
            w2_rows = reduced.get(id(entry.row_w2), entry.row_w2)
            d2_rows = reduced.get(id(d2_rows), d2_rows)
            valid = w2_rows > 0
            ratio = torch.where(valid, d2_rows / w2_rows.where(valid, 1.0), 0.0).sqrt()
            acc.add(entry.buckets, _ROW_RATIO, ratio.sum())
            acc.add(entry.buckets, _ROW_RATIO_N, valid.sum(dtype=torch.float64))
            acc.add(entry.buckets, _ROW_D, d2_rows.sqrt().sum())
            acc.add(entry.buckets, _ROW_D2, d2_rows.sum())
            acc.add_constant(entry.buckets, _ROW_N, float(d2_rows.numel()))
            ratio_max = ratio.max()
        if entry.counts_cols():
            d2_cols = reduced.get(id(d2_cols), d2_cols)
            acc.add(entry.buckets, _COL_D, d2_cols.sqrt().sum())
            acc.add(entry.buckets, _COL_D2, d2_cols.sum())
            acc.add_constant(entry.buckets, _COL_N, float(d2_cols.numel()))
        return ratio_max

    def _delta_y_partials(self, tp_vectors) -> list:
        """(entry, num, den): squared norms of X dW^T and X W_{t-1}^T.

        Row-parallel products are partial sums over the input shards and are returned unsquared;
        they are completed by the TP all-reduce. Sequence-parallel column-parallel inputs are
        all-gathered over TP first, so every TP peer multiplies the same tokens by its own
        output rows and the squared norms add up over TP.
        """
        gathered = {}
        by_group = {}
        for module, entry, gather in self._delta_y_modules:
            if gather and module in self._inputs:
                by_group.setdefault(id(entry.tp_group), (entry.tp_group, []))[1].append(module)
        for group, modules in by_group.values():
            local = torch.cat([self._inputs[m].reshape(-1) for m in modules])
            out = torch.empty(get_pg_size(group) * local.numel(), dtype=local.dtype, device=local.device)
            torch.distributed.all_gather_into_tensor(out, local, group=group)
            out = out.view(get_pg_size(group), -1)
            offset = 0
            for module in modules:
                x = self._inputs[module]
                gathered[module] = out[:, offset : offset + x.numel()].reshape(-1, x.shape[-1])
                offset += x.numel()
        results = []
        for module, entry, _ in self._delta_y_modules:
            x = gathered.get(module, self._inputs.get(module))
            if x is None or x.shape[-1] != entry.snapshot.shape[-1]:
                continue
            prev = entry.snapshot.float()
            num = x @ (entry.weight().float() - prev).t()
            den = x @ prev.t()
            if entry.rows_split:
                tp_vectors.setdefault(id(entry.tp_group), [entry.tp_group]).extend((num, den))
            else:
                num = num.square().sum(dtype=torch.float64)
                den = den.square().sum(dtype=torch.float64)
            results.append((entry, num, den))
        return results

    def _format(self, buffer, row_max) -> Dict[str, float]:
        stats = {}
        for scope in range(self.num_scopes):
            prefix = "update" if scope == 0 else f"update/layers/{scope - 1}"
            for family_index, family in enumerate(_FAMILIES):
                s = buffer[scope][family_index]
                if s[_PARAMS] == 0:
                    continue
                a, c, d = s[_A], s[_C], s[_D]
                out = {}
                if a > 0:
                    out["relative"] = math.sqrt(d / a)
                    out["weight-norm"] = math.sqrt(a)
                    norms = math.sqrt(max(a + 2 * c + d, 0.0) * a)
                    if norms > 0:
                        cos = (a + c) / norms
                        sin = math.sqrt(max(a * d - c * c, 0.0)) / norms
                        out["angle-cos"] = cos
                        out["angle-deg"] = math.degrees(math.atan2(sin, cos))
                    if s[_SNAP_ERR] > 0:
                        out["bf16-snapshot-floor"] = math.sqrt(s[_SNAP_ERR] / a)
                if s[_GG] > 0:
                    out["grad-norm"] = math.sqrt(s[_GG])
                    if a > 0:
                        radial = s[_GW] / math.sqrt(a)
                        tangential = math.sqrt(max(s[_GG] - radial * radial, 0.0))
                        out["grad-weight-cos"] = s[_GW] / math.sqrt(s[_GG] * a)
                        out["grad-radial"] = radial
                        out["grad-tangential"] = tangential
                        out["grad-tangential-fraction"] = tangential / math.sqrt(s[_GG])
                if s[_MM] > 0 and s[_GGM] > 0:
                    out["grad-momentum-cos"] = s[_GM] / math.sqrt(s[_GGM] * s[_MM])
                    out["momentum-grad-norm-ratio"] = math.sqrt(s[_MM] / s[_GGM])
                if s[_ROW_RATIO_N] > 0:
                    out["per-neuron-relative/mean"] = s[_ROW_RATIO] / s[_ROW_RATIO_N]
                    out["per-neuron-relative/max"] = row_max[scope][family_index]
                for name, total, square, count in (
                    ("delta-row-norm", s[_ROW_D], s[_ROW_D2], s[_ROW_N]),
                    ("delta-col-norm", s[_COL_D], s[_COL_D2], s[_COL_N]),
                ):
                    if count > 0:
                        mean = total / count
                        out[f"{name}/mean"] = mean
                        out[f"{name}/std"] = math.sqrt(max(square / count - mean * mean, 0.0))
                if s[_DY_DEN] > 0:
                    out["delta-y"] = math.sqrt(s[_DY_NUM] / s[_DY_DEN])
                if s[_SRANK_N] > 0:
                    out["stable-rank"] = s[_SRANK] / s[_SRANK_N]
                if s[_REFF_N] > 0:
                    out["momentum-r-eff"] = s[_REFF] / s[_REFF_N]
                for metric, value in out.items():
                    head, _, tail = metric.partition("/")
                    stats[f"{prefix}/{head}/{family}" + (f"/{tail}" if tail else "")] = value
        return stats

    def _format_experts(self, stats, buffer: torch.Tensor) -> None:
        """Summaries over the experts of each MoE layer; histograms (tensors) on spectral steps.

        Family-level ``cv`` is the mean over MoE layers of the per-layer CV; ``min``/``max`` are
        over all (layer, expert) pairs.
        """
        for io, family in enumerate(_EXPERT_FAMILIES):
            pooled, cvs = {}, {}
            for layer in range(self.num_layers):
                s = buffer[layer, io]
                a = s[:, _E_A]
                if not bool((a > 0).all()):
                    continue
                metrics = {"relative": (s[:, _E_D] / a).sqrt(), "weight-norm": a.sqrt()}
                if bool((s[:, _E_GG] > 0).all()):
                    metrics["grad-norm"] = s[:, _E_GG].sqrt()
                    metrics["grad-weight-cos"] = s[:, _E_GW] / (s[:, _E_GG] * a).sqrt()
                    if bool((s[:, _E_MM] > 0).all()):
                        metrics["grad-momentum-cos"] = s[:, _E_GM] / (s[:, _E_GG] * s[:, _E_MM]).sqrt()
                for metric, values in metrics.items():
                    values = values.float()
                    cv = (values.std(unbiased=False) / values.mean().abs()).item()
                    pooled.setdefault(metric, []).append(values)
                    cvs.setdefault(metric, []).append(cv)
                    if self.per_layer:
                        prefix = f"update/layers/{layer}/experts/{family}/{metric}"
                        stats[f"{prefix}/cv"] = cv
                        stats[f"{prefix}/min"] = values.min().item()
                        stats[f"{prefix}/max"] = values.max().item()
                        stats[f"{prefix}/argmin"] = int(values.argmin())
                        stats[f"{prefix}/argmax"] = int(values.argmax())
                        if self._spectral:
                            stats[f"{prefix}/hist"] = values
            for metric, values in pooled.items():
                values = torch.cat(values)
                prefix = f"update/experts/{family}/{metric}"
                stats[f"{prefix}/cv"] = sum(cvs[metric]) / len(cvs[metric])
                stats[f"{prefix}/min"] = values.min().item()
                stats[f"{prefix}/max"] = values.max().item()
                if self._spectral:
                    stats[f"{prefix}/hist"] = values


def _tp_all_reduce(tp_vectors) -> Dict[int, torch.Tensor]:
    """One all-reduce per TP group; returns {id(original tensor): reduced tensor}."""
    reduced = {}
    for group, *tensors in tp_vectors.values():
        flat = torch.cat([t.reshape(-1).double() for t in tensors])
        torch.distributed.all_reduce(flat, group=group)
        offset = 0
        for tensor in tensors:
            reduced[id(tensor)] = flat[offset : offset + tensor.numel()].view(tensor.shape)
            offset += tensor.numel()
    return reduced


def _gram(tensor: torch.Tensor, partition_dim: Optional[int]) -> torch.Tensor:
    if partition_dim is None:
        use_rows = tensor.shape[-2] <= tensor.shape[-1]
    else:
        use_rows = partition_dim == 1
    return tensor @ tensor.transpose(-2, -1) if use_rows else tensor.transpose(-2, -1) @ tensor


def _spectral_sums(eig: torch.Tensor, stat: int) -> tuple:
    """(sum over matrices, number of matrices) of stable rank or r_eff from Gram eigenvalues."""
    trace = eig.sum(-1)
    valid = trace > 0
    if stat == _SRANK:
        value = trace / eig[:, -1].where(valid, 1.0)
    else:
        value = eig.sqrt().sum(-1).square() / trace.where(valid, 1.0)
    return value.where(valid, 0.0).sum(), valid.sum(dtype=torch.float64)


def _stack_grouped_experts(entries: List[_Entry]) -> List[_Entry]:
    """Merge the locally owned ``weight{i}`` of one grouped GEMM into a single [E, out, in] entry.

    Saves one set of kernels per expert (hundreds per rank at 256 experts). TP-sharded experts
    stay separate, since row statistics of TP-sharded 3D tensors are not supported.
    """
    groups: Dict[tuple, List[_Entry]] = {}
    merged = []
    for entry in entries:
        param = entry.main_params[0]
        if entry.expert_slots is None or entry.tp_group is not None or param.ndim != 2:
            merged.append(entry)
            continue
        key = (
            entry.name.rpartition(".")[0], entry.family, param.shape, param.dtype,
            param.device, entry.grad_ok, id(entry.optimizer),
        )
        if key not in groups:
            groups[key] = []
            merged.append(groups[key])
        groups[key].append(entry)
    result = []
    for item in merged:
        if isinstance(item, _Entry):
            result.append(item)
            continue
        first = item[0]
        first.model_params = [p for e in item for p in e.model_params]
        first.main_params = [p for e in item for p in e.main_params]
        first.expert_slots = [slot for e in item for slot in e.expert_slots]
        first.row_stats = True
        result.append(first)
    return result


def _is_tp_sharded(name: str, param: torch.Tensor) -> bool:
    """TE >= 2.13 no longer marks grouped-GEMM expert weights ``tensor_model_parallel``; Megatron
    then stamps only ``partition_dim`` on them (TEGroupedLinear, explicit expert comm), so a
    grouped expert weight with a partition dim is TP-sharded too."""
    if getattr(param, "tensor_model_parallel", False):
        return True
    return re.search(r"\.weight\d+$", name) is not None and getattr(param, "partition_dim", -1) in (0, 1)


def _momentum(optimizer, param) -> Optional[torch.Tensor]:
    """First-moment buffer before the step (keys in the module docstring)."""
    state = optimizer.state.get(param)
    if not state:
        return None
    for key in ("momentum_buffer", "momentum", "exp_avg"):
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.shape == param.shape:
            return value
    return None


def _matrix_optimizer_types() -> tuple:
    """Muon-type optimizers that step whole matrices (the others are Adam-like). Neutrino and Dion
    exist only in the Neutrino fork, so every import is optional."""
    types = []
    for module, name in (("neutrino", "Neutrino"), ("dion", "Dion"), ("muon", "TensorParallelMuon")):
        try:
            types.append(getattr(importlib.import_module(f"{__package__}.{module}"), name))
        except (ImportError, AttributeError):
            pass
    return tuple(types)


def _is_matrix_step(optimizer, group, matrix_types) -> bool:
    if "use_orthogonal_updates" in group:  # MDDecoupling and NeutrinoMD choose per group
        return bool(group["use_orthogonal_updates"])
    return isinstance(optimizer, matrix_types)


class _NameTags:
    """The attributes :func:`muon_logging._gain_log_family` reads, derived from the name for
    parameters the Muon/MD factories did not tag (Neutrino, Dion, Adam)."""

    def __init__(self, name: str, param: torch.Tensor):
        tied = getattr(param, "shared_embedding", False)
        self.ndim = param.ndim
        self.is_router = getattr(param, "is_router", name.endswith("router.weight"))
        self.is_md_embedding_parameter = getattr(
            param,
            "is_md_embedding_parameter",
            name.endswith("word_embeddings.weight") or (name.endswith("output_layer.weight") and tied),
        )
        self.is_md_output_parameter = getattr(
            param, "is_md_output_parameter", name.endswith("output_layer.weight") and not tied
        )
        self.is_out_proj = getattr(
            param,
            "is_out_proj",
            "linear_fc2" in name
            or "linear_proj" in name
            or name.endswith("out_proj.weight")
            or "experts.weight2" in name,
        )


def _family(name: str, param: torch.Tensor) -> str:
    family = getattr(param, "md_gain_log_family", None)
    return family if family is not None else _gain_log_family(name, _NameTags(name, param))


def _layer(name: str, param: torch.Tensor, modules) -> Optional[int]:
    layer = getattr(param, "md_gain_log_layer", None)
    if layer is not None:
        return layer
    module_name = name.rpartition(".")[0]
    while module_name:
        layer_number = getattr(modules.get(module_name), "layer_number", None)
        if layer_number is not None:
            return int(layer_number) - 1
        module_name = module_name.rpartition(".")[0]
    return None


# ----------------------------------------------------------------------
# process-wide collector used by megatron.training
# ----------------------------------------------------------------------
_COLLECTOR: Optional[UpdateStatsCollector] = None
_LAST_STATS: Dict = {}


def setup(model_chunks, optimizer, **kwargs) -> UpdateStatsCollector:
    """Create the process-wide collector (kwargs of :class:`UpdateStatsCollector`)."""
    global _COLLECTOR
    if _COLLECTOR is not None:
        _COLLECTOR.remove_hooks()
    _COLLECTOR = UpdateStatsCollector(model_chunks, optimizer, **kwargs)
    return _COLLECTOR


def notify_phase_change(name: str) -> None:
    """Log every step again for the next ``dense_window`` steps, e.g. when switching NTP <-> RL."""
    if _COLLECTOR is not None:
        _COLLECTOR.notify_phase_change(name)


def set_phase(name: str) -> None:
    """Name the current phase ("ntp" or "rl") without restarting the dense window."""
    if _COLLECTOR is not None:
        _COLLECTOR.set_phase(name)


def begin_step(iteration: int) -> None:
    if _COLLECTOR is not None:
        _COLLECTOR.begin_step(iteration)


def before_optimizer_step() -> None:
    if _COLLECTOR is not None:
        _COLLECTOR.before_optimizer_step()


def after_optimizer_step(update_successful: bool) -> None:
    global _LAST_STATS
    if _COLLECTOR is not None:
        _LAST_STATS = _COLLECTOR.after_optimizer_step(update_successful) or _LAST_STATS


def pop_stats() -> Dict:
    """Stats of the last logging step, returned once."""
    global _LAST_STATS
    stats, _LAST_STATS = _LAST_STATS, {}
    return stats
