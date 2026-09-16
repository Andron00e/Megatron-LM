"""
Neutrino projects the momentum matrix onto a thin k-dimensional random basis
regenerated locally from a shared seed, and preserves the unprojected
residual as error feedback.

Key design points (see _research notes / hat-muon report):

* Shape-bucketing. All eligible 2D matrix params of identical shape are stacked
  into a single batch dimension and the projection / Gram / Cholesky-QR / solve
  / reconstruction run as batched (bmm) ops with ONE basis-generation kernel and
  ONE TP collective per bucket. This collapses the per-parameter Python loop
  (hundreds of tiny kernels + blocking collectives) that dominated wallclock.
  3D expert params ([E, M, N]) are already batched over E and are processed one
  tensor at a time, reusing the same batched primitives.

* Metrics gating. The diagnostic block (eigvalsh condition numbers, per-param
  norms, dozens of .item() device syncs) only runs every ``metrics_interval``
  steps and aggregates on-device, with a single host sync at the end. This was
  the single biggest throughput sink at --log-interval 1.

* Rank-correction scale mode. A low-rank polar update U Vᵀ has Frobenius norm
  ~sqrt(k) whereas a full Muon update has norm sqrt(min(M,N)); ``scale_mode``
  values ``rank_correct`` / ``shape_up_rank`` multiply by sqrt(min(M,N)/k_eff)
  so the update magnitude matches full Muon at the same LR.

* Effective-k policy. ``k_mode='fixed'`` uses a constant ``k``; ``k_mode='ratio'``
  sets k per shape as round(k_ratio * min(M,N)). Both are exposed for ablation.

* DP projection hook (flag-gated, default off). When enabled and a per-DDP
  communication hook is registered, the data-parallel reduction of the dense
  gradient is replaced by an all-reduce of the thin Y. Because projection is
  linear and V is identical across DP ranks, this is mathematically identical to
  reducing G first; it only moves the collective from O(MN) to O(Mk) bytes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

# NOTE: these package-level imports are safe because this module is only imported lazily from
# megatron.training.training (never from this package's __init__), so megatron.core.optimizer is
# fully initialized by the time this module loads — same pattern as muon.py / md_decoupling.py.
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


try:
    from emerging_optimizers.orthogonalized_optimizers import (
        get_muon_scale_factor as _emerging_get_muon_scale_factor,
    )
except ImportError:
    _emerging_get_muon_scale_factor = None


def get_muon_scale_factor(
    size_out: int,
    size_in: int,
    mode: str = "spectral",
    k_eff: Optional[int] = None,
    r_eff: Optional[float] = None,
) -> float:
    """Muon orthogonalization scale factor.

    ``rank_correct`` and ``shape_up_rank`` are Neutrino-specific and require
    ``k_eff`` (the effective projection rank actually used for this param). They
    rescale a low-rank polar update (Frobenius norm ~sqrt(k_eff)) up to the norm
    of a full-rank Muon update (sqrt(min(d_out, d_in))).

    These extra modes are handled locally here so adding them never changes the
    behavior of master/muon (which use their own get_muon_scale_factor).
    """
    if mode == "rank_correct_saturating":
        # sqrt(min_dim / max(k_eff, r_eff)) -- the rank lift SATURATES once k exceeds
        # the momentum's nuclear effective rank, because past that point the extra
        # directions carry no energy and lifting by sqrt(min_dim/k) overshoots.
        #
        # Only binds where r_eff > k_eff. Measured on a finished 350m run (20B tokens,
        # exact svdvals over all 32 managed matrices): r_eff is 360-514 for mlp.fc1,
        # 178-458 for mlp.fc2, 219-301 for attn.qkv and 87-165 for attn.proj, all of
        # min_dim 1024. So at k=256 this binds for fc1 and part of fc2/qkv and is inert
        # for proj; at k=512 it is inert almost everywhere. Deliberately NOT a default:
        # on synthetic problems this was the largest win on two convex objectives and
        # the largest loss on a nonconvex MLP, and that sign flip is still unexplained.
        min_dim = min(size_out, size_in)
        denom = float(max(1, k_eff if k_eff is not None else 1))
        if r_eff is not None:
            denom = max(denom, r_eff)
        return (min_dim / denom) ** 0.5
    if mode in ("rank_correct", "shape_up_rank", "rank_correct_calibrated"):
        min_dim = min(size_out, size_in)
        # sqrt(min(M,N) / k_eff): lift low-rank update norm to full-rank Muon norm.
        rank_corr = (min_dim / max(1, k_eff)) ** 0.5 if k_eff is not None else 1.0
        if mode == "shape_up_rank":
            # also apply the shape_up aspect correction on top of rank correction.
            return rank_corr * (max(size_out / size_in, size_in / size_out) ** 0.5)
        if mode == "rank_correct_calibrated":
            # Empirical finding (researchtools/scratchpad/neutrino-code/variants/
            # rank_correction_calibration.py, brainstorm doc §2 item 5): shape_up_rank's
            # aspect-ratio multiplier sqrt(max(M/N,N/M)) massively OVER-corrects for the
            # low-rank case (measured 38-57% too large at aspect ratios 2-4, e.g. typical
            # SwiGLU FFN aspect ~2.7) — it was derived for full-rank Muon and doesn't
            # transfer. Measuring the true ||full-rank Muon update|| / ||raw rank-k polar
            # update|| ratio directly (white-noise and power-law synthetic spectra,
            # aspect 1-4, k/min_dim 0.06-0.5): plain rank_correct (NO aspect term) already
            # gets within ~10-15%, with a striking, low-noise, aspect-dependent but
            # k-INDEPENDENT residual ratio: ~0.90 at aspect=1 (square), settling to
            # ~0.85-0.88 for aspect>=2 and staying flat there. A single constant (0.87)
            # captures this to within ~3% across the whole tested grid — far simpler than
            # fitting a full aspect-dependent curve, and the improvement over shape_up_rank
            # is large enough (2-4x smaller error) that the residual imprecision from using
            # one flat constant is a minor concern next to what it replaces.
            return rank_corr * 0.87
        return rank_corr
    if mode == "shape_up":
        return max(size_out / size_in, size_in / size_out) ** 0.5
    if mode == "none":
        return 1.0
    if _emerging_get_muon_scale_factor is not None:
        return _emerging_get_muon_scale_factor(size_out, size_in, mode=mode)
    # kimi's muon fallback
    return 0.2 * (max(size_out, size_in) ** 0.5)


def mark_dp_projected(params, enabled: bool = True) -> int:
    """Tag the params whose data-parallel reduction this optimizer takes over.

    The DDP grad buffer skips its dense all-reduce for exactly these params and the
    optimizer all-reduces the thin projection instead. Shared entry point so Neutrino,
    NeutrinoMD and any future variant opt in the same way, and so the attribute name has
    one definition (``param_and_grad_buffer.DP_PROJECTED_ATTR``).

    Pass params from the MODEL, not from a sharded optimizer's param_groups: every rank
    must tag the same set or ranks issue different collectives and NCCL hangs. The buffer
    verifies this once at the first sync, but tagging correctly is cheaper than the error.

    Returns the number of params tagged.
    """
    # Imported lazily: this module is itself imported lazily from megatron.training, so
    # megatron.core.distributed is loaded by now, and a top-level import here would risk
    # an optimizer -> distributed import cycle.
    from megatron.core.distributed.param_and_grad_buffer import DP_PROJECTED_ATTR

    count = 0
    for p in params:
        # 2D matrices and 3D stacked expert weights are what the compressed branch
        # handles; everything else (embeddings, biases, norms) stays on the dense path.
        if p.ndim not in (2, 3):
            continue
        if enabled:
            setattr(p, DP_PROJECTED_ATTR, True)
            count += 1
        else:
            # Clear rather than skip. A param object can outlive the optimizer that tagged
            # it -- rebuilding the optimizer with projection off (a resume with the flag
            # dropped, say) would otherwise leave a live tag, so the buffer would keep
            # skipping reductions that nothing replaces: wrong gradients, no error.
            if hasattr(p, DP_PROJECTED_ATTR):
                delattr(p, DP_PROJECTED_ATTR)
    return count


def dp_projection_group(pg_collection, expert: bool):
    """Data-parallel process group the thin exchange runs over.

    Must be the same group the DDP grad buffer would have reduced the dense gradient
    over, or the skip and the replacement cover different rank sets.
    """
    if pg_collection is not None:
        g = getattr(pg_collection, 'expt_dp' if expert else 'dp_cp', None)
        if g is not None:
            return g
    import megatron.core.parallel_state as ps

    if expert:
        return ps.get_expert_data_parallel_group()
    return ps.get_data_parallel_group(with_context_parallel=True)


@torch.no_grad()
def drain_dp_projection_stats(dp_thin_bytes: int) -> dict:
    """Consume the grad buffer's skip counters and turn them into realized DP metrics.

    Always drains, even when there is nothing to report, so the counters cannot run on
    unbounded across a job whose metrics happen to be disabled and then produce a
    nonsensical ratio the first time they are read.

    Returns {} when no skip happened, or when the DP group has a single rank -- there is
    no data-parallel traffic to save at DP=1, and reporting a multiplier there would
    describe an all-reduce that never costs anything.
    """
    from megatron.core.distributed.param_and_grad_buffer import DP_PROJECTION_STATS

    dp = DP_PROJECTION_STATS.snapshot()
    DP_PROJECTION_STATS.reset()
    if dp['syncs'] <= 0 or dp['skipped_bytes'] <= 0:
        return {}
    if not torch.distributed.is_initialized():
        return {}
    # Size of the DATA-parallel group, not the world. A TP-only job has world_size > 1 and
    # dp_size == 1: there is no DP traffic to save there, and the global size would report
    # a multiplier for an all-reduce that never had a cost.
    try:
        dp_size = torch.distributed.get_world_size(group=dp_projection_group(None, False))
    except Exception:
        return {}
    if dp_size <= 1:
        return {}
    baseline = dp['skipped_bytes'] + dp['reduced_bytes']
    actual = dp['reduced_bytes'] + dp_thin_bytes
    syncs = float(dp['syncs'])
    return {
        # Per grad-sync, i.e. per bucket group per step -- NOT per training step; a model
        # has several bucket groups. The ratio is unaffected, the absolutes are not.
        "neutrino/dp_baseline_bytes_per_sync": baseline / syncs,
        "neutrino/dp_actual_bytes_per_sync": actual / syncs,
        "neutrino/dp_dense_bytes_skipped_per_sync": dp['skipped_bytes'] / syncs,
        "neutrino/dp_thin_bytes_sent_per_sync": dp_thin_bytes / syncs,
        "neutrino/dp_comm_savings_multiplier": baseline / actual if actual > 0 else 1.0,
    }


def dp_average_(tensor: Tensor, group, wire_dtype: Optional[torch.dtype] = None) -> int:
    """In-place data-parallel AVG all-reduce. Returns bytes actually sent.

    AVG, not SUM, to match Megatron's average_in_collective reduction — which the DDP
    skip requires — so projected and unprojected params come out on the same scale.
    Correctness rests on linearity: V is DP-shared, so avg_DP(G V) == avg_DP(G) V.

    ``wire_dtype`` decouples the *transport* precision from the *compute* precision. Y is
    built in fp32 because Cholesky-QR on the k x k Gram is ill-conditioned in bf16 — that is
    a property of the factorization, not of the exchange. The dense gradient this collective
    replaces is itself bf16 in Megatron's grad buffer, so sending Y in fp32 moves twice the
    bytes of the thing it stands in for, and is *less* faithful to the baseline rather than
    more. Casting to bf16 for the collective and back afterwards halves DP traffic and leaves
    the factorization untouched.

    Returns 0 when no collective ran, so callers accumulate only realized traffic.
    """
    if group is None or not torch.distributed.is_initialized():
        return 0
    if torch.distributed.get_world_size(group=group) <= 1:
        return 0
    if wire_dtype is None or wire_dtype == tensor.dtype:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG, group=group)
        return tensor.numel() * tensor.element_size()
    payload = tensor.to(wire_dtype)
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.AVG, group=group)
    tensor.copy_(payload)
    return payload.numel() * payload.element_size()


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


# Newton-Schulz steps for the k*k inverse square root used by --neutrino-orth-mode polar.
# 30 chosen from a measured accuracy sweep against eigh (see test_polar2.py); the analytic
# bound in _inv_sqrt_newton_schulz says ~20 suffices at k=256, so this carries slack.
_POLAR_NS_STEPS = 30


def _resolve_neutrino_variant(name: str):
    """Look up a Neutrino subclass by name from _research/variants/, lazily.

    Late import (inside the function, not at module scope) so this file has no
    load-time dependency on _research/ being on PYTHONPATH — mirrors the
    try/except ImportError pattern already used for _research.logging_patch in
    pretrain_gpt.py. 'base' (the default) always returns the vanilla Neutrino
    class without even attempting the import.
    """
    if name in (None, 'base'):
        return Neutrino
    try:
        from _research.variants.neutrino_variants import VARIANT_REGISTRY
    except ImportError:
        log_single_rank(
            logger, logging.WARNING,
            f"neutrino_variant={name!r} requested but _research.variants is not importable "
            "(is _research/ on PYTHONPATH?) -- falling back to the base Neutrino optimizer.",
        )
        return Neutrino
    if name not in VARIANT_REGISTRY:
        log_single_rank(
            logger, logging.WARNING,
            f"neutrino_variant={name!r} not found in VARIANT_REGISTRY "
            f"(known: {list(VARIANT_REGISTRY)}) -- falling back to base Neutrino.",
        )
        return Neutrino
    return VARIANT_REGISTRY[name]


_VARIANT_FIELD_NAMES = {
    'kschedule': [
        'neutrino_k_schedule_start_k', 'neutrino_k_schedule_end_k',
        'neutrino_k_schedule_total_steps', 'neutrino_k_schedule_style',
    ],
    'perlayerk': ['neutrino_layer_k_overrides'],
    'ef21': ['neutrino_ef_variant'],
    # srht, powersgd, base: no extra config-driven kwargs today (SRHT/PowerSGD's
    # only knob is the existing basis_init/k machinery); listed for completeness.
    'srht': [],
    'powersgd': [],
    'base': [],
}


def _variant_kwargs_from_config(config: OptimizerConfig, variant_name: str) -> dict:
    """Collect only the hyperparameter fields relevant to ``variant_name``.

    Scoped per-variant (rather than passing every variant field to whichever
    subclass got picked) so an unrelated variant's __init__ never sees a kwarg
    it doesn't declare.
    """
    out = {}
    for f in _VARIANT_FIELD_NAMES.get(variant_name, []):
        v = getattr(config, f, None)
        if v is not None:
            out[f] = v
    return out


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
        k: int = 512,  # random basis vectors for proj (used when k_mode == "fixed")
        no_momentum: bool = False,  # not to allocate buffer
        scale_mode: str = "spectral",  # spectral | shape_up | shape_scaling | unit_rms_norm | none | rank_correct | shape_up_rank
        basis_init: str = "gaussian",  # gaussian (default), rademacher, uniform, orthonormal
        # --- orthonormalization of the thin projection Y ---
        orth_mode: str = "cholesky_qr",  # cholesky_qr (default, = current behavior) | polar
        # --- effective-k policy (ablation knob) ---
        k_mode: str = "fixed",  # "fixed" -> use k; "ratio" -> round(k_ratio * min(M,N))
        k_ratio: float = 0.25,  # used when k_mode == "ratio"
        # --- which side the sketch compresses ---
        sketch_side: str = "short",  # short (default, = current behavior) | long
        k_long: Optional[Union[int, str]] = None,  # int or "auto"; transposed path only
        # --- basis refresh period (ablation knob) ---
        basis_refresh: int = 1,  # 1 = fresh basis every step; T > 1 reuses it T steps; 0 = fixed
        # --- error feedback ---
        error_feedback: bool = True,  # fold the off-subspace residual back in next step
        # --- metrics gating ---
        metrics_interval: int = 50,  # compute/log diagnostics every N steps; 0 disables
        overlap_lags: Optional[List[int]] = None,  # log overlap of V_t vs V_{t-h} per lag h
        # --- DP projection hook (flag-gated, default off) ---
        dp_projection: bool = False,
        dp_wire_bf16: bool = False,
        # --- subspace-drift diagnostic (flag-gated, default off; no effect on the update) ---
        log_subspace_drift: bool = False,
        subspace_drift_interval: int = 50,
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
            k_mode=k_mode,
            k_ratio=k_ratio,
        )
        super().__init__(params, defaults)
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.orth_mode = orth_mode
        # Y = G V sketches the input side (N -> k), so the wire tensor is M x k and a tall
        # matrix (fc1: M = 2.5 N) ships 2.5x the bytes of its transpose at the same k.
        # 'long' transposes such matrices before sketching so the sketched dimension is
        # always max(M, N) and every wire tensor is min(M, N) x k; k_long, if set, is the
        # rank used on exactly those transposed matrices: an int is a fixed override, "auto"
        # is the equal-bytes arm, k' = round(k M/N) per shape so N x k' has the bytes of M x k
        # (at 350m every GQA qkv is tall too, so a single int cannot be equal-bytes for
        # both qkv and fc1).
        assert sketch_side in ("short", "long"), sketch_side
        assert k_long is None or sketch_side == "long", "k_long needs sketch_side='long'"
        assert k_long is None or k_long == "auto" or (isinstance(k_long, int) and k_long > 0), k_long
        cls = type(self)
        if sketch_side != "short" or k_long is not None:
            assert (
                cls._process_bucket is Neutrino._process_bucket
                and cls._process_expert is Neutrino._process_expert
            ), f"{cls.__name__} overrides _process_bucket/_process_expert and ignores sketch_side/k_long"
        self.sketch_side = sketch_side
        self.k_long = k_long
        # basis_refresh T: the basis seed uses step // T, so one sketch V serves T
        # consecutive steps (T == 1: fresh every step; T == 0: one fixed subspace).
        assert basis_refresh >= 0, basis_refresh
        self.basis_refresh = basis_refresh
        self.overlap_lags = tuple(overlap_lags or ())
        assert all(h > 0 for h in self.overlap_lags), self.overlap_lags
        self.error_feedback = error_feedback
        self.metrics_interval = metrics_interval
        self.dp_projection = dp_projection
        self.log_subspace_drift = log_subspace_drift
        self.subspace_drift_interval = max(1, subspace_drift_interval)
        self._subspace_log_buffer: List[tuple] = []
        self._global_step = 0
        # bucket_id -> r_eff. Deliberately NOT in self.state: the checkpoint saver
        # asserts .shape on every self.state value, so a bare float there breaks saves.
        self._r_eff_cache: dict = {}
        # Bytes this optimizer actually handed to NCCL for the thin DP exchange, so the
        # reported saving is measured on both sides rather than derived from shapes.
        self._dp_thin_bytes = 0
        # Transport dtype for the thin exchange, independent of the fp32 used for
        # Cholesky-QR. None keeps the historical fp32-on-the-wire behaviour.
        self.dp_wire_dtype = torch.bfloat16 if dp_wire_bf16 else None

        # assign a deterministic unique param_id to each parameter in the optimizer
        # this is used to form the random basis generation seed and bucket ids.
        # Kept in self.state (a defaultdict) so it migrates with the optimizer
        # state across Megatron's param reloads; see state_dict() below for why
        # the checkpoint path must filter to owned params.
        # When dp_projection is on, also mark the 2D/3D matrix params this
        # optimizer actually compresses: the DDP grad-sync hook keys off this
        # attribute to skip the dense DP reduction for exactly these params (and
        # nothing else — so no other optimizer's params are ever affected).
        param_id = 0
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['param_id'] = param_id
                param_id += 1
                all_params.append(p)
        mark_dp_projected(all_params, enabled=self.dp_projection)

    # ------------------------------------------------------------------
    # data-parallel groups (only used when dp_projection is enabled)
    # ------------------------------------------------------------------
    def _dp_group_for(self, expert: bool):
        return dp_projection_group(self.pg_collection, expert)

    @torch.no_grad()
    def _dp_average_(self, tensor: Tensor, expert: bool) -> None:
        """In-place data-parallel average of `tensor` (the thin Y, or the dense
        g_adj in the full-rank fallback)."""
        if not self.dp_projection:
            return
        self._dp_thin_bytes += dp_average_(
            tensor, self._dp_group_for(expert), wire_dtype=self.dp_wire_dtype
        )

    # ------------------------------------------------------------------
    # effective rank
    # ------------------------------------------------------------------
    def _effective_k(self, M_global: int, N_global: int, group: dict, items: Optional[list] = None) -> int:
        """Effective projection rank for a param of global shape (M_global, N_global).

        ``items`` (the bucket's [(param, grad), ...] list, when the caller has it)
        is accepted but unused here — a hook for variants (e.g. per-layer k) that
        need to key off which parameter(s) this is, not just their shape.
        """
        min_dim = min(M_global, N_global)
        if group['k_mode'] == "ratio":
            k = int(round(group['k_ratio'] * min_dim))
        else:
            k = group['k']
        # keep at least 1, never exceed the smaller dimension.
        return max(1, min(k, min_dim))

    def _sketch_transposed(self, M_global: int, N_global: int) -> bool:
        """True when this matrix is sketched as its transpose (see ``sketch_side``)."""
        return self.sketch_side == "long" and M_global > N_global

    def _k_for_side(self, k_eff: int, transposed: bool, M_global: int, N_global: int) -> int:
        if transposed and self.k_long is not None:
            k_long = self.k_long
            if k_long == "auto":  # equal bytes: N_global * k' == M_global * k_eff
                k_long = round(k_eff * M_global / N_global)
            return max(1, min(k_long, min(M_global, N_global)))
        return k_eff

    # ------------------------------------------------------------------
    # basis generation
    # ------------------------------------------------------------------
    def _basis_seed(self, bucket_id: int, step: int, tp_rank: int) -> int:
        # combine step, bucket_id, and tp_rank deterministically for a 32-bit uint seed.
        if self.basis_refresh != 1:
            step = step // self.basis_refresh if self.basis_refresh else 0
        return (step * 997 + bucket_id * 1000003 + tp_rank * 17) & 0xFFFFFFFF

    def _generate_basis(
        self,
        P: int,
        N: int,
        k: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        N_global: int,
        basis_init: str = "gaussian",
    ) -> Tensor:
        """Deterministic basis tensor of shape [P, N, k] (batched over the bucket).

        Identical across DP ranks (seed has no DP term); for column-sharded params
        the caller folds tp_rank into ``seed`` so each TP slice differs.
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        if basis_init == "rademacher":
            rand_u = torch.rand((P, N, k), generator=generator, device=device, dtype=dtype)
            V = torch.where(rand_u > 0.5, torch.ones_like(rand_u), -torch.ones_like(rand_u))
            return V / (N_global**0.5)
        elif basis_init == "uniform":
            rand_u = torch.rand((P, N, k), generator=generator, device=device, dtype=dtype)
            V = (rand_u * 2.0 - 1.0) * (3.0**0.5)
            return V / (N_global**0.5)
        elif basis_init == "orthonormal":
            G = torch.randn((P, N, k), generator=generator, device=device, dtype=torch.float32)
            Q, R = torch.linalg.qr(G)
            # deterministic sign convention so Q is identical across ranks
            d = torch.diagonal(R, dim1=-2, dim2=-1)
            ph = d.sign()
            Q = Q * ph.unsqueeze(-2)
            # orthonormal columns (Qᵀ Q = I): no 1/sqrt(N) scaling needed.
            return Q.to(dtype)
        else:
            # gaussian (default)
            V = torch.randn((P, N, k), generator=generator, device=device, dtype=dtype)
            return V / (N_global**0.5)

    # ------------------------------------------------------------------
    # Cholesky-QR
    # ------------------------------------------------------------------
    def _cholesky_qr(self, Gram: Tensor) -> Tensor:
        """Batched Cholesky factorization with adaptive jitter and eigh fallback.

        Gram is always [B, k, k] (B = bucket size or expert count).
        """
        k = Gram.shape[-1]
        device = Gram.device
        dtype = Gram.dtype

        diag = torch.diagonal(Gram, dim1=-2, dim2=-1)  # [B, k]
        mean_diag = diag.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        eye = torch.eye(k, device=device, dtype=dtype).unsqueeze(0)

        Gram_jittered = Gram + 1e-4 * mean_diag * eye
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
        """Eigenvalue decomposition fallback for singular Gram matrices."""
        eigenvalues, eigenvectors = torch.linalg.eigh(Gram)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        return eigenvectors @ torch.diag_embed(eigenvalues.sqrt())

    def _r_eff_for(self, bucket_id: int, g_adj: Tensor) -> float:
        """Nuclear effective rank ‖M‖_*²/‖M‖_F² of the adjusted gradient, cached.

        Exact svdvals rather than newton_schulz5: NS5 does not converge to the polar
        factor, it drives singular values into a band around 1, so it underestimates
        ‖M‖_* — and r_eff squares that error. Far too sloppy to decide whether r_eff
        sits above or below k, which is the only thing this value is used for.

        Refreshed on the metrics cadence and reused in between: the momentum spectrum
        drifts over thousands of steps, so a per-step SVD would buy nothing. One value
        per bucket (mean over the stacked params), since one scale applies to all of them.

        Cached OFF `self.state`, in a plain dict keyed by bucket_id. Anything living in
        `self.state[p]` is walked by `make_sharded_optimizer_tensor` at checkpoint save,
        which asserts `.shape` on every value — a bare float there raises AttributeError
        and kills the save while training itself looks healthy. Keying by the int
        bucket_id rather than the param object also keeps the cache from going stale
        when Megatron migrates params across a reload; and since this is only a cache,
        a miss costs one recomputation rather than correctness.
        """
        every = self.metrics_interval if self.metrics_interval > 0 else 50
        cached = self._r_eff_cache.get(bucket_id)
        if cached is not None and (self._global_step % every) != 0:
            return cached
        M = g_adj.float()
        if M.ndim == 2:
            M = M.unsqueeze(0)
        sv = torch.linalg.svdvals(M)
        nuc = sv.sum(dim=-1)
        fro_sq = (sv * sv).sum(dim=-1).clamp_min(1e-12)
        r_eff = float((nuc * nuc / fro_sq).mean().item())
        self._r_eff_cache[bucket_id] = r_eff
        return r_eff

    def _solve_triangular(self, L: Tensor, Y: Tensor) -> Tensor:
        """Solve L @ Xᵀ = Yᵀ to obtain U = X (so that U Lᵀ = Y, Uᵀ U = I)."""
        Y_T = Y.transpose(-2, -1)
        X = torch.linalg.solve_triangular(L, Y_T, upper=False, left=True)
        return X.transpose(-2, -1)

    def _polar_orthogonalize(self, Gram: Tensor, Y: Tensor) -> Tensor:
        """polar(Y) = Y (YᵀY)^(-1/2), the exact subspace-restricted LMO solution.

        Cholesky-QR returns Q = Y L⁻ᵀ, the QR factor. Q spans the right column space but
        is NOT the polar factor, and the achieved update alignments differ:

            ⟨M, Q V̄ᵀ⟩ = tr(R),        ⟨M, polar(Y) V̄ᵀ⟩ = ‖R‖_*      (R = Lᵀ)

        with tr(R) ≤ ‖R‖_* and equality iff R is diagonal, i.e. iff the restricted
        spectrum is flat. Only polar(Y) attains the LMO value that Neutrino is
        implicitly maximizing. The shortfall in guaranteed per-step decrease is
        (tr(R)/‖R‖_*)² and is scale-invariant, so no learning rate, scale mode or
        calibration constant can recover it.

        Measured on this project's real momentum (350m, iteration 9537): tr(R)/‖R‖_* has
        median 0.814 at k=256 and 0.741 at k=512 — the loss is large and grows with k,
        which is why raising k stops paying for itself.

        Implementation notes, all three from measurement rather than theory:
          * Computed as Y Gram^(-1/2) via eigh of the symmetric k×k Gram we already
            formed, NOT as Q @ polar(Lᵀ). The latter is algebraically identical (since
            RᵀR = L Lᵀ = Gram) but numerically forms and then cancels L⁻ᵀ against an
            ill-conditioned L, which left the result with singular values in
            [0.817, 1.012] instead of 1.
          * Never newton_schulz5: its quintic coefficients settle into a band around 1
            rather than onto it (10/20/40 steps measured WORSE than 5), it returned
            max|W − I| = 0.31 on a flat spectrum where polar(R) is provably I, and it
            computes in bfloat16.
          * Gram^(-1/2) comes from a matmul-only Newton-Schulz iteration, NOT eigh.
            Measured per bucket of 8 at M=2560: this path 3.6 ms at k=256 and 5.3 ms at
            k=512, against a batched k*k eigh at 35.7 / 103.2 ms (svd worse still, 47 ms)
            and the cholesky_qr path it replaces at 0.87 / 1.88 ms. cuSOLVER only batches
            syevj for n <= 32, so an eigh here runs as P sequential factorizations and
            would have cost 3-8% of a 350m step; this costs ~0.1-0.2%. It is also the
            MORE accurate of the two, since it iterates in fp64 while an fp32 eigh carries
            ~cond*eps ~ 1e-2 relative error at the conditioning we actually see.
            See _inv_sqrt_newton_schulz.

        The jitter mirrors _cholesky_qr's: adding 1e-4·mean(diag)·I to Gram shifts every
        eigenvalue by exactly that amount, so both paths regularize identically.
        """
        G = Gram.float()
        k = G.shape[-1]
        eye = torch.eye(k, device=G.device, dtype=G.dtype).expand_as(G)
        # Same relative jitter as _cholesky_qr, so both paths regularize identically.
        mean_diag = torch.diagonal(G, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        A = G + 1e-4 * mean_diag * eye
        inv_sqrt = self._inv_sqrt_newton_schulz(A, eye)
        return torch.bmm(Y.float(), inv_sqrt).to(Y.dtype)

    def _inv_sqrt_newton_schulz(
        self, A: Tensor, eye: Tensor, steps: int = _POLAR_NS_STEPS
    ) -> Tensor:
        """A^(-1/2) for symmetric positive-definite A, by the coupled Newton iteration.

        Matmul-only, which is the whole point: torch.linalg.eigh on a [P, k, k] batch cost
        36 ms at k=256 and 103 ms at k=512 here, because cuSOLVER only batches syevj for
        n <= 32 and otherwise runs P sequential factorizations. This runs in a few ms.

            Y_0 = A/alpha,  Z_0 = I;    T = (3I - Z Y)/2;    Y <- Y T,   Z <- T Z

        Then Y -> I and Z -> (A/alpha)^(-1/2), so A^(-1/2) = Z / sqrt(alpha). Convergence
        needs alpha >= lambda_max; we use ‖A‖_F, which bounds ‖A‖_2 for any matrix, so the
        iteration is unconditionally stable rather than relying on an eigenvalue estimate.

        Unlike newton_schulz5's quintic (whose coefficients settle into a band around 1
        rather than onto it, so extra steps do not help), this iteration genuinely
        converges, and the step count is a real accuracy knob.

        Why a fixed step count is safe, with no host sync in the hot path: the jitter above
        floors every eigenvalue of A at 1e-4·mean(diag), while ‖A‖_F bounds the top, so
        cond(A) <= 1e4·k regardless of the data. In the commuting scalar reduction
        p = ZY obeys p <- p(3-p)²/4, i.e. p grows ~2.25x per step while small and then
        converges quadratically near 1, so ~log_2.25(1e4·k) + a few steps suffices:
        18 + slack at k=256. Measured: 20, 30 and 40 steps agree to all printed digits on
        the worst spectrum tested, so 30 carries real slack.
        """
        # float64 is REQUIRED, not defensive. In float32 the iterates sit flat for 17
        # steps and then explode -- measured |Y| 5.1e-2 -> 1.5e8 at step 18, NaN by 20,
        # on a Gram with cond 1.8e5. Symmetrizing Z@Y each step does not help. This is
        # the documented instability of the Newton-Schulz square-root iteration, which is
        # only stable for ||I - A|| < 1 and amplifies rounding error with the condition
        # number. In float64 the same iteration is flat through 40 steps and converged by
        # 20. These are k*k matrices, so the cost is negligible against the [M, N] update.
        A64 = A.double()
        eye64 = eye.double()
        alpha = torch.linalg.matrix_norm(A64, ord="fro", dim=(-2, -1), keepdim=True)
        Yk = A64 / alpha
        Zk = eye64.clone()
        for _ in range(steps):
            T = 0.5 * (3.0 * eye64 - torch.bmm(Zk, Yk))
            Yk = torch.bmm(Yk, T)
            Zk = torch.bmm(T, Zk)
        return (Zk * alpha.rsqrt()).to(A.dtype)

    # ------------------------------------------------------------------
    # per-param helpers
    # ------------------------------------------------------------------
    def _tp_group_for(self, p: Tensor):
        if not self.pg_collection:
            return None
        return self.pg_collection.expt_tp if getattr(p, 'expert_tp', False) else self.pg_collection.tp

    def _g_adj(self, p: Tensor, grad: Tensor, group: dict) -> Tensor:
        """Momentum + Nesterov + error-feedback adjusted gradient."""
        state = self.state[p]
        no_momentum = group['no_momentum']
        momentum = group['momentum']
        nesterov = group['nesterov']
        if no_momentum:
            g_hat = grad
        else:
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
            if nesterov:
                g_hat = grad.mul(1.0 - momentum).add_(buf, alpha=momentum)
            else:
                g_hat = buf
        e_prev = state.get('error_buffer')
        return g_hat if e_prev is None else g_hat + e_prev

    def _ef_writeback(self, p: Tensor, g_adj: Tensor, YV: Tensor) -> None:
        """Cache what the low-rank reconstruction (YV) didn't capture of g_adj.

        Extracted from the two inline call sites (_process_bucket/_process_expert)
        so variants can override the error-feedback bookkeeping (e.g. EF21-style
        state accumulation) without duplicating the surrounding batched flow.
        Behavior-preserving extraction — identical to the prior inline code.
        """
        if self.error_feedback:
            self.state[p]['error_buffer'].copy_(g_adj - YV)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1
        do_metrics = (
            self.metrics_interval > 0 and (self._global_step % self.metrics_interval == 0)
        )
        metrics_accum = _MetricsAccumulator() if do_metrics else None
        # Unlike metrics_accum (gated to every `metrics_interval` steps), subspace-drift
        # values are collected EVERY step (cheap: one extra matmul+trace on top of the
        # Cholesky-QR/solve already computed) — the host sync is what's batched, at
        # `subspace_drift_interval`, so we still get true step-to-step drift rather than
        # drift-over-an-interval.
        subspace_values: Optional[list] = [] if self.log_subspace_drift else None

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            scale_mode = group['scale_mode']
            basis_init = group.get('basis_init', 'gaussian')

            # ---- partition params: 1D (inline SGD), 3D experts, 2D matrices ----
            buckets: dict[tuple, list] = {}
            experts_3d: list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    if not group['no_momentum']:
                        state['momentum_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)
                    if self.error_feedback:
                        state['error_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)
                state['step'] += 1

                grad = p.grad.data.float()

                if p.ndim not in (2, 3):
                    p.data.add_(grad.to(p.dtype), alpha=-lr)
                    continue

                if p.ndim == 3:
                    experts_3d.append((p, grad))
                else:
                    # bucket 2D matrices by (shape, partition_dim, expert_tp)
                    partition_dim = getattr(p, 'partition_dim', None)
                    if partition_dim == -1:
                        partition_dim = None
                    key = (tuple(p.shape), partition_dim, bool(getattr(p, 'expert_tp', False)))
                    buckets.setdefault(key, []).append((p, grad))

            # ---- process 2D shape-buckets (batched) ----
            for key, items in buckets.items():
                self._process_bucket(
                    items, key, group, lr, wd, scale_mode, basis_init, metrics_accum,
                    subspace_values,
                )

            # ---- process 3D expert tensors (already batched over E) ----
            for p, grad in experts_3d:
                self._process_expert(
                    p, grad, group, lr, wd, scale_mode, basis_init, metrics_accum,
                    subspace_values,
                )

        if metrics_accum is not None:
            # Drain unconditionally, outside log(): log() returns early when wandb is
            # absent or disabled, and leaving the counters un-drained there would let them
            # accumulate across the whole run and report a meaningless ratio the first
            # time wandb did appear.
            dp_metrics = drain_dp_projection_stats(self._dp_thin_bytes)
            self._dp_thin_bytes = 0
            metrics_accum.log(self._global_step, dp_metrics=dp_metrics)

        if subspace_values:
            all_overlaps = torch.cat([ov for ov, _ in subspace_values])
            total_n = sum(ov.numel() for ov, _ in subspace_values)
            weighted_chance = sum(ov.numel() * ch for ov, ch in subspace_values) / total_n
            self._subspace_log_buffer.append(
                (self._global_step, all_overlaps.mean(), weighted_chance)
            )
        if self.log_subspace_drift and self._global_step % self.subspace_drift_interval == 0:
            self._flush_subspace_drift()

        return loss

    @torch.no_grad()
    def _track_subspace_drift(self, items, U, k_eff, M_global, subspace_values):
        """Diagnostic only — no effect on the update.

        Compares each item's newly-orthonormalized U (this step's realized thin
        projection) against the same param's U from the previous step, via the
        standard subspace-overlap metric ||U_prev^T U_cur||_F^2 / k (in [0, 1]; 1 =
        identical subspace). For two INDEPENDENT random k-dim subspaces of R^M this
        has expectation ~k/M ("chance level") — logged alongside so an above-chance
        value is legible without an offline reference computation. See
        researchtools/scratchpad/neutrino-scratch/brainstorm-neutrino-improvements.md
        section 4: if Neutrino's realized subspace drifts slower than chance despite V
        being fresh-random every step, that's evidence a warm-started (PowerSGD-style)
        basis would track real gradient structure instead of noise.
        """
        overlaps = []
        for i, (p, _grad) in enumerate(items):
            state = self.state[p]
            cur_U = U[i].detach()
            prev_U = state.get('prev_subspace_U')
            if prev_U is not None and prev_U.shape == cur_U.shape:
                overlaps.append(torch.sum(torch.matmul(prev_U.transpose(-2, -1), cur_U) ** 2) / k_eff)
            state['prev_subspace_U'] = cur_U.clone()
        if overlaps:
            subspace_values.append((torch.stack(overlaps), k_eff / max(1, M_global)))

    @torch.no_grad()
    def _flush_subspace_drift(self):
        if not self._subspace_log_buffer:
            return
        try:
            import wandb
        except ImportError:
            self._subspace_log_buffer = []
            return
        if wandb.run is None:
            self._subspace_log_buffer = []
            return
        steps = [s for s, _, _ in self._subspace_log_buffer]
        chances = [c for _, _, c in self._subspace_log_buffer]
        # single host sync for the whole buffered window.
        overlaps = torch.stack([o for _, o, _ in self._subspace_log_buffer]).tolist()
        for step, ov, ch in zip(steps, overlaps, chances):
            try:
                wandb.log(
                    {
                        "neutrino/subspace_overlap": ov,
                        "neutrino/subspace_overlap_chance_level": ch,
                    },
                    step=step,
                )
            except Exception:
                pass
        self._subspace_log_buffer = []

    @torch.no_grad()
    def _track_basis_overlap(self, V, k_eff, bucket_id, seed_tp_rank, gen, metrics_accum):
        """Diagnostic only — no effect on the update; runs on the metrics cadence.

        Overlap ||Q_{t-h}^T Q_t||_F^2 / k (same definition as _track_subspace_drift)
        between this step's sketch basis V_t and the basis V_{t-h} used h steps ago,
        both orthonormalized by QR; V [P, N, k]. V_{t-h} is regenerated from its seed via
        ``gen`` (one draw + QR per lag, no stored history), so under basis_refresh T the
        value is exactly 1 whenever t and t-h fall in the same block of T steps and
        ~k/N (chance) otherwise. This is the temporal-coverage measure for the (k, mu)
        coupling: how much of the last h steps' subspace the current basis still spans.
        """
        Q_cur = torch.linalg.qr(V.float())[0]
        for h in self.overlap_lags:
            if self._global_step - h < 1:
                continue
            seed = self._basis_seed(bucket_id, self._global_step - h, seed_tp_rank)
            Q_prev = torch.linalg.qr(gen(seed).float())[0]
            ov = torch.sum(torch.bmm(Q_prev.transpose(-2, -1), Q_cur) ** 2, dim=(-2, -1)) / k_eff
            metrics_accum.add_overlap(h, ov)

    def build_sharded_optimizer_state(self, model_param, value, state_key, prefix):
        """Keep non-model-shaped optimizer state out of the sharded checkpoint.

        Hook called by dist_checkpointing.optimizer.optim_state_to_sharding_state.
        A non-None return is used as-is (no shape assertion); None falls back to the
        standard model-shaped path. See fix_state_dict.py for the full rationale.
        """
        # local import keeps this module free of load-order constraints
        from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject

        # Claim OUR OWN state explicitly, by key name -- never by shape introspection.
        # model_param may be a ShardedTensorFactory (no .local_shape), in which case a
        # shape-based check silently falls through to the asserting path and blows up in
        # validate_metadata_integrity. These entries are all reconstructible: param_id is
        # reassigned in __init__, the drift cache is diagnostic, and EF restarts from zero.
        if (
            state_key == "param_id"
            or state_key == "prev_subspace_U"
            or state_key.startswith("neutrino_error_buffer")
        ):
            return LocalNonpersistentObject(value)

        # Anything else: let the standard model-shaped path handle it.
        return None

    def state_dict(self):
        """Serialize only the params in the current param_groups.

        ``param_id`` is written into ``self.state`` for every parameter at
        ``__init__``, but on the layer-wise distributed path ``param_groups`` are
        sharded per rank, so ``self.state`` also holds entries for params this
        rank does not own. torch's ``Optimizer.state_dict()`` maps every
        ``self.state`` key through ``param_groups`` and raises ``KeyError`` on
        those non-owned params at checkpoint save. The owned params are exactly
        ``param_groups``, so temporarily filter ``self.state`` to them for the
        duration of serialization (restored in ``finally``; non-destructive).
        """
        live = {id(p) for group in self.param_groups for p in group['params']}
        full_state = self.state
        self.state = defaultdict(
            dict, {p: s for p, s in full_state.items() if id(p) in live}
        )
        try:
            return super().state_dict()
        finally:
            self.state = full_state

    # ------------------------------------------------------------------
    # 2D matrix bucket (batched over P params of identical shape)
    # ------------------------------------------------------------------
    def _process_bucket(
        self, items, key, group, lr, wd, scale_mode, basis_init, metrics_accum, subspace_values=None
    ):
        shape, partition_dim, _expert_tp = key
        M, N = shape
        P = len(items)
        p0 = items[0][0]
        device = items[0][1].device
        fdtype = torch.float32

        tp_group = self._tp_group_for(p0)
        tp_size = 1
        tp_rank = 0
        if tp_group is not None and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(group=tp_group)
            tp_rank = torch.distributed.get_rank(group=tp_group)

        M_global, N_global = M, N
        if tp_size > 1:
            if partition_dim == 0:
                M_global = M * tp_size
            elif partition_dim == 1:
                N_global = N * tp_size

        transposed = self._sketch_transposed(M_global, N_global)
        k_eff = self._k_for_side(
            self._effective_k(M_global, N_global, group, items=items), transposed, M_global, N_global
        )
        # Scale modes describe the update of the M x N parameter (spectral is asymmetric in
        # out/in), so they always see the parameter's own global shape, never the sketched one.
        scale_out, scale_in = M_global, N_global

        expert = bool(_expert_tp)

        # --- full-rank fallback (projection rank meets/exceeds the matrix) ---
        if min(M_global, N_global) <= k_eff:
            for p, grad in items:
                g_adj = self._g_adj(p, grad, group)
                # In dp_projection mode the DDP skipped this param's DP reduction,
                # so average the dense g_adj here (no compression for fallback params).
                self._dp_average_(g_adj, expert)
                U = newton_schulz5(g_adj, steps=5)
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                scale = get_muon_scale_factor(M_global, N_global, mode=scale_mode, k_eff=k_eff)
                p.data.add_(U.to(p.dtype), alpha=-lr * scale)
            return

        # --- stack adjusted gradients: [P, M, N] ---
        g_adj = torch.stack([self._g_adj(p, grad, group) for p, grad in items], dim=0)

        # --- long-side sketch: work on G^T, so from here M, N, partition_dim and the
        # TP path are those of the transposed matrix; update/YV are transposed back below.
        if transposed:
            g_adj = g_adj.transpose(-2, -1)
            M, N, M_global, N_global = N, M, N_global, M_global
            partition_dim = None if partition_dim is None else 1 - partition_dim

        # --- basis [P, N, k_eff], shared across DP, tp_rank folded in for col-shard ---
        bucket_id = self.state[p0]['param_id']
        seed_tp_rank = tp_rank if partition_dim == 1 else 0
        seed = self._basis_seed(bucket_id, self._global_step, seed_tp_rank)
        V = self._generate_basis(P, N, k_eff, seed, device, fdtype, N_global, basis_init)
        if metrics_accum is not None and self.overlap_lags:
            gen = lambda s: self._generate_basis(P, N, k_eff, s, device, fdtype, N_global, basis_init)
            self._track_basis_overlap(V, k_eff, bucket_id, seed_tp_rank, gen, metrics_accum)

        # --- project: Y_local = g_adj @ V -> [P, M, k] ---
        Y_local = torch.bmm(g_adj, V)

        # --- data-parallel exchange of the THIN Y (the communication saving) ---
        # Done before the TP collective; DP-avg and TP-reduce commute (distinct groups).
        self._dp_average_(Y_local, expert)

        # --- TP collective + Cholesky-QR ---
        if tp_size > 1 and partition_dim == 0:
            # row (output) sharded: each rank holds M/TP rows, basis identical.
            Gram = torch.bmm(Y_local.transpose(-2, -1), Y_local)
            torch.distributed.all_reduce(Gram, group=tp_group)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y_local)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y_local)
            Y_for_resid = Y_local
        elif tp_size > 1 and partition_dim == 1:
            # column (input) sharded: reconstruct global Y via all-reduce of partials.
            torch.distributed.all_reduce(Y_local, group=tp_group)
            Y = Y_local
            Gram = torch.bmm(Y.transpose(-2, -1), Y)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y)
            Y_for_resid = Y
        else:
            Gram = torch.bmm(Y_local.transpose(-2, -1), Y_local)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y_local)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y_local)
            Y_for_resid = Y_local

        update = torch.bmm(U, V.transpose(-2, -1))  # [P, M, N]
        YV = torch.bmm(Y_for_resid, V.transpose(-2, -1))  # [P, M, N]
        if transposed:
            g_adj, update, YV = (t.transpose(-2, -1) for t in (g_adj, update, YV))

        if self.log_subspace_drift and subspace_values is not None:
            self._track_subspace_drift(items, U, k_eff, M_global, subspace_values)

        r_eff = (
            self._r_eff_for(bucket_id, g_adj)
            if scale_mode == "rank_correct_saturating"
            else None
        )
        scale = get_muon_scale_factor(
            scale_out, scale_in, mode=scale_mode, k_eff=k_eff, r_eff=r_eff
        )

        # --- apply update + write error feedback, per param ---
        for i, (p, _grad) in enumerate(items):
            if wd != 0:
                p.data.mul_(1.0 - lr * wd)
            p.data.add_(update[i].to(p.dtype), alpha=-lr * scale)
            self._ef_writeback(p, g_adj[i], YV[i])

        if metrics_accum is not None:
            # Comm savings apply on whichever axis actually ran a collective this
            # step: the TP Gram/Y sync above, and/or the DP thin-Y exchange in
            # _dp_average_ (the real "Neutrino is communication-efficient" path —
            # active whenever dp_projection is on, independent of tp_size).
            track_comm = tp_size > 1 or self.dp_projection
            metrics_accum.add_batch(
                g_adj, YV, U, update, items, lr, scale, k_eff, M, N, track_comm
            )

    # ------------------------------------------------------------------
    # 3D expert tensor [E, M, N]
    # ------------------------------------------------------------------
    def _process_expert(
        self, p, grad, group, lr, wd, scale_mode, basis_init, metrics_accum, subspace_values=None
    ):
        E, M, N = p.shape
        device = grad.device
        fdtype = torch.float32

        tp_group = self._tp_group_for(p)
        tp_size = 1
        tp_rank = 0
        if tp_group is not None and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(group=tp_group)
            tp_rank = torch.distributed.get_rank(group=tp_group)

        partition_dim = getattr(p, 'partition_dim', None)
        if partition_dim == -1:
            partition_dim = None

        M_global, N_global = M, N
        if tp_size > 1:
            if partition_dim == 0:
                M_global = M * tp_size
            elif partition_dim == 1:
                N_global = N * tp_size

        transposed = self._sketch_transposed(M_global, N_global)
        k_eff = self._k_for_side(
            self._effective_k(M_global, N_global, group, items=[(p, grad)]),
            transposed, M_global, N_global,
        )
        scale_out, scale_in = M_global, N_global

        expert = bool(getattr(p, 'expert_tp', False))

        if min(M_global, N_global) <= k_eff:
            g_adj = self._g_adj(p, grad, group)
            self._dp_average_(g_adj, expert)
            U = newton_schulz5(g_adj, steps=5)
            if wd != 0:
                p.data.mul_(1.0 - lr * wd)
            scale = get_muon_scale_factor(M_global, N_global, mode=scale_mode, k_eff=k_eff)
            p.data.add_(U.to(p.dtype), alpha=-lr * scale)
            return

        g_adj = self._g_adj(p, grad, group)  # [E, M, N]
        if transposed:
            g_adj = g_adj.transpose(-2, -1)
            M, N, M_global, N_global = N, M, N_global, M_global
            partition_dim = None if partition_dim is None else 1 - partition_dim

        # shared basis across experts of identical shape: a single [N, k] matrix
        # broadcast over the E batch dim via matmul (no stride-0 bmm).
        bucket_id = self.state[p]['param_id']
        seed_tp_rank = tp_rank if partition_dim == 1 else 0
        seed = self._basis_seed(bucket_id, self._global_step, seed_tp_rank)
        V = self._generate_basis(1, N, k_eff, seed, device, fdtype, N_global, basis_init)[0]  # [N, k]
        Vt = V.transpose(-2, -1)  # [k, N]
        if metrics_accum is not None and self.overlap_lags:
            gen = lambda s: self._generate_basis(1, N, k_eff, s, device, fdtype, N_global, basis_init)
            self._track_basis_overlap(V.unsqueeze(0), k_eff, bucket_id, seed_tp_rank, gen, metrics_accum)

        Y_local = torch.matmul(g_adj, V)  # [E, M, k]
        self._dp_average_(Y_local, expert)

        if tp_size > 1 and partition_dim == 0:
            Gram = torch.bmm(Y_local.transpose(-2, -1), Y_local)
            torch.distributed.all_reduce(Gram, group=tp_group)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y_local)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y_local)
            Y_for_resid = Y_local
        elif tp_size > 1 and partition_dim == 1:
            torch.distributed.all_reduce(Y_local, group=tp_group)
            Y = Y_local
            Gram = torch.bmm(Y.transpose(-2, -1), Y)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y)
            Y_for_resid = Y
        else:
            Gram = torch.bmm(Y_local.transpose(-2, -1), Y_local)
            if self.orth_mode == "polar":
                U = self._polar_orthogonalize(Gram, Y_local)
            else:
                L = self._cholesky_qr(Gram)
                U = self._solve_triangular(L, Y_local)
            Y_for_resid = Y_local

        update = torch.matmul(U, Vt)  # [E, M, N]
        YV = torch.matmul(Y_for_resid, Vt)  # [E, M, N]
        if transposed:
            g_adj, update, YV = (t.transpose(-2, -1) for t in (g_adj, update, YV))

        if self.log_subspace_drift and subspace_values is not None:
            # E-batched analog of _track_subspace_drift: one param p, but U stacks E
            # experts, so cache/compare the whole [E, M, k] tensor per expert-slice.
            state = self.state[p]
            cur_U = U.detach()
            prev_U = state.get('prev_subspace_U')
            if prev_U is not None and prev_U.shape == cur_U.shape:
                overlap_per_expert = (
                    torch.sum(torch.matmul(prev_U.transpose(-2, -1), cur_U) ** 2, dim=(-2, -1))
                    / k_eff
                )
                subspace_values.append((overlap_per_expert, k_eff / max(1, M_global)))
            state['prev_subspace_U'] = cur_U.clone()

        if wd != 0:
            p.data.mul_(1.0 - lr * wd)
        r_eff = (
            self._r_eff_for(bucket_id, g_adj)
            if scale_mode == "rank_correct_saturating"
            else None
        )
        scale = get_muon_scale_factor(
            scale_out, scale_in, mode=scale_mode, k_eff=k_eff, r_eff=r_eff
        )
        p.data.add_(update.to(p.dtype), alpha=-lr * scale)
        self._ef_writeback(p, g_adj, YV)

        if metrics_accum is not None:
            track_comm = tp_size > 1 or self.dp_projection
            metrics_accum.add_batch(
                g_adj, YV, U, update, [(p, grad)], lr, scale, k_eff, M, N, track_comm
            )


class _MetricsAccumulator:
    """On-device metric accumulation with a single host sync at log time.

    Avoids the per-parameter .item()/eigvalsh storm that serialized rank 0.
    """

    def __init__(self):
        self._err = []
        self._upd = []
        self._orth = []
        self._n = 0
        self._neutrino_bytes = 0
        self._muon_bytes = 0
        self._overlap: dict = {}  # lag h -> [per-param overlap tensors]

    def add_overlap(self, lag, ov):
        self._overlap.setdefault(lag, []).append(ov)

    @torch.no_grad()
    def add_batch(self, g_adj, YV, U, update, items, lr, scale, k_eff, M, N, track_comm):
        # g_adj/YV/update: [B, M, N]; U: [B, M, k]
        resid = g_adj - YV
        err = torch.linalg.vector_norm(resid, dim=(-2, -1))
        gnorm = torch.linalg.vector_norm(g_adj, dim=(-2, -1))
        self._err.append((err / (gnorm + 1e-8)))

        k = U.shape[-1]
        eye = torch.eye(k, device=U.device, dtype=U.dtype)
        UTU = torch.bmm(U.transpose(-2, -1), U)
        self._orth.append(torch.linalg.vector_norm(UTU - eye, dim=(-2, -1)))

        unorm = torch.linalg.vector_norm(update, dim=(-2, -1))
        wnorm = torch.stack([torch.linalg.vector_norm(p.data.float()) for p, _ in items])
        self._upd.append((lr * scale * unorm / (wnorm.to(unorm.device) + 1e-8)))

        B = g_adj.shape[0]
        self._n += B
        # Nominal payload comparison for the TENSOR-parallel axis only: the Gram/Y sync
        # that runs when tp_size > 1. These are the sizes the code would send, derived
        # from shapes rather than observed, so they answer "how much smaller is the thin
        # exchange" and NOT "how much did we save". The data-parallel saving is measured
        # for real in param_and_grad_buffer.DP_PROJECTION_STATS and reported separately;
        # do not conflate the two.
        if track_comm:
            self._neutrino_bytes += B * M * k_eff * 4  # thin Y, fp32
            self._muon_bytes += B * M * N * 2  # dense G, bf16

    @torch.no_grad()
    def log(self, step, dp_metrics: Optional[dict] = None):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None or self._n == 0:
            return
        err = torch.cat(self._err)
        upd = torch.cat(self._upd)
        orth = torch.cat(self._orth)
        # single host sync for the whole step's diagnostics.
        has_comm = self._neutrino_bytes > 0
        metrics = {
            "neutrino/global_mean_error_ratio": err.mean().item(),
            "neutrino/global_mean_update_ratio": upd.mean().item(),
            "neutrino/global_mean_orth_deviation": orth.mean().item(),
            # Nominal, shape-derived, TENSOR-parallel axis only. Answers "how much
            # smaller is the thin exchange", not "what did we save".
            "neutrino/tp_nominal_thin_bytes": float(self._neutrino_bytes),
            "neutrino/tp_nominal_ratio": (
                self._muon_bytes / self._neutrino_bytes if has_comm else 1.0
            ),
            "neutrino/tp_comm_axis_active": float(has_comm),
        }
        for h, vals in self._overlap.items():
            metrics[f"neutrino/overlap_lag{h}"] = torch.cat(vals).mean().item()

        # Realized data-parallel saving, measured on both sides: skipped/reduced from the
        # slices the grad buffer did or did not hand to NCCL, thin bytes from the tensors
        # this optimizer actually all-reduced. Drained by the caller so the counters clear
        # even on the paths that return before reaching here.
        if dp_metrics:
            metrics.update(dp_metrics)

        try:
            wandb.log(metrics, step=step)
        except Exception:
            pass


def get_megatron_neutrino_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build the Neutrino optimizer for the given model chunks.

    Mirrors :func:`megatron.core.optimizer.muon.get_megatron_muon_optimizer`: 2D matrix params
    AND 3D expert-batched params (``[E, M, N]``, e.g. grouped-GEMM MoE experts) are optimized by
    :class:`Neutrino`; everything else (1D biases/norms, embedding/output params) is delegated to
    a chained external AdamW via :func:`get_megatron_optimizer`. Unlike ``muon.py`` in this tree,
    3D expert tensors are included in the Neutrino-managed group directly — Neutrino's
    ``_process_expert`` already batches the projection/orthogonalization over the expert axis, so
    no separate MoE-specific optimizer path is needed. When ``layer_wise_distributed_optimizer`` is
    True the whole chain is wrapped in :class:`LayerWiseDistributedOptimizer` to shard optimizer
    state over DP (required: the standard distributed optimizer flattens each shard to 1-D, which
    breaks the 2-D/3-D projection math).
    """
    if config.use_distributed_optimizer:
        raise Exception(
            'neutrino with the standard distributed optimizer is not supported; '
            'use --use-layer-wise-distributed-optimizer to shard optimizer state.'
        )
    if config.fp16:
        raise Exception('neutrino with fp16 is not supported (use bf16 or fp32).')
    if config.neutrino_dp_projection and layer_wise_distributed_optimizer:
        # The layer-wise optimizer shards param_groups per rank, so each rank would tag a
        # different subset and the ranks would issue different numbers of collectives. The
        # grad buffer catches this at the first sync, but failing here names the cause.
        raise Exception(
            '--neutrino-dp-projection is incompatible with the layer-wise distributed '
            'optimizer: it shards param_groups per rank, so ranks tag different params. '
            'Drop --use-distributed-optimizer so the replicated all-reduce path is used.'
        )

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up Neutrino optimizer with config {config}')

    base_overrides = (
        config_overrides if config_overrides is not None else get_standard_config_overrides(config)
    )

    def neutrino_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if 'momentum_buffer' not in opt.state[p] and not group['no_momentum']:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)
                if 'error_buffer' not in opt.state[p] and opt.error_feedback:
                    opt.state[p]['error_buffer'] = torch.zeros_like(p.data, dtype=torch.float32)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # Tag expert-parallel params (needed for both the TP/EP-aware process-group lookup in
    # _tp_group_for and the is_expert_parallel split below); mirrors muon.py's own tagging.
    # Also tag a coarse layer-type ('attention' | 'mlp' | None) by name substring — used
    # only by the per-layer-k variant (NeutrinoPerLayerK._effective_k); a no-op attribute
    # for every other variant/the base optimizer.
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if 'self_attention' in name or 'attention' in name:
                param.neutrino_layer_tag = 'attention'
            elif 'mlp' in name or 'linear_fc' in name:
                param.neutrino_layer_tag = 'mlp'

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

    neutrino_kwargs = dict(
        lr=(config.matrix_lr if config.matrix_lr is not None
            else config.muon_lr_factor * (config.lr or 0.0)),
        momentum=config.muon_momentum,
        nesterov=config.muon_use_nesterov,
        weight_decay=config.weight_decay,
        k=config.neutrino_k,
        no_momentum=config.neutrino_no_momentum,
        scale_mode=config.muon_scale_mode,
        basis_init=config.neutrino_basis_init,
        orth_mode=config.neutrino_orth_mode,
        k_mode=config.neutrino_k_mode,
        k_ratio=config.neutrino_k_ratio,
        sketch_side=config.neutrino_sketch_side,
        k_long=config.neutrino_k_long,
        basis_refresh=config.neutrino_basis_refresh,
        error_feedback=not config.neutrino_no_error_feedback,
        metrics_interval=config.neutrino_metrics_interval,
        overlap_lags=config.neutrino_overlap_lags,
        dp_projection=config.neutrino_dp_projection,
        dp_wire_bf16=getattr(config, 'neutrino_dp_wire_bf16', False),
        log_subspace_drift=config.neutrino_log_subspace_drift,
        subspace_drift_interval=config.neutrino_subspace_drift_interval,
        pg_collection=pg_collection,
        tp_mode='duplicated',  # forced — see Neutrino's own docstring / __init__ default.
    )

    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, base_overrides)
    if config.matrix_lr is not None or config.muon_lr_factor != 1.0:
        floor = config.min_lr if config.min_lr is not None else 0.0
        matrix_lr = neutrino_kwargs['lr']
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

    neutrino_variant_name = getattr(config, 'neutrino_variant', None) or 'base'
    NeutrinoClass = _resolve_neutrino_variant(neutrino_variant_name)
    variant_kwargs = _variant_kwargs_from_config(config, neutrino_variant_name)
    optimizer = NeutrinoClass(linear_param_groups, **neutrino_kwargs, **variant_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            config.bf16 = False
            reset_config_bf16 = True
        else:
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, neutrino_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, neutrino_init_state_fn)

    optimizers = [optimizer]

    if len(expert_param_groups) > 0:
        expert_optimizer = NeutrinoClass(expert_param_groups, **neutrino_kwargs, **variant_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, neutrino_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, neutrino_init_state_fn)
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

    init_fns = [neutrino_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Neutrino')
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
