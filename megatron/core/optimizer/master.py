# adapted from megatron/core/optimizer/muon.py and https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b8365dbdce94a979090af735698fabc6be497f06/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py.
import math
import logging
from typing import Callable,Optional, Literal, override

import torch

from . import _get_param_groups, get_megatron_optimizer
from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey
from megatron.core.transformer.module import MegatronModule
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank
from megatron.core.optimizer.muon import get_muon_scale_factor


logger = logging.getLogger(__name__)


try:
    import emerging_optimizers
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
except ImportError:
    emerging_optimizers = None


class MasterOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        # Common settings.
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        weight_decay_method: Literal["decoupled", "independent"]  = "decoupled",

        # adam & ademamix settings.
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 2.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
        eps: float = 1e-8,

        # Hypersphere optimization.
        hypersphere_mode: Optional[Literal["row", "col", "rowcol", "flat"]] = None,
        hypersphere_kind: Optional[Literal["l2", "standard", "spectral", "orthogonal"]] = None,
        hypersphere_radius: Literal["learnable"] | float = 1.0,
        hypersphere_eps: float = 1e-8,
        hypersphere_update: bool = True,
        hypersphere_project: bool = False,
        hypersphere_soft: bool = False,

        # Muon.
        use_orthogonal_updates: bool = False,  # Enable or disable muon entirely.
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        split_qkv: bool = True,  # Also applies to hypersphere optimization.
        split_qkv_heads: bool = True,  # Also applies to hypersphere optimization.
        qkv_split_shapes: Optional[tuple[int, int, int]] = None,
        qkv_dim: Optional[int] = None,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ):

        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov
        self.weight_decay_method = weight_decay_method

        self.hypersphere_mode = hypersphere_mode
        self.hypersphere_kind = hypersphere_kind
        self.hypersphere_radius = hypersphere_radius
        self.hypersphere_eps = hypersphere_eps
        self.hypersphere_update = hypersphere_update
        self.hypersphere_project = hypersphere_project
        self.hypersphere_soft = hypersphere_soft

        self.split_qkv = split_qkv
        self.split_qkv_heads  = split_qkv_heads
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_dim = qkv_dim

        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        
        default_args_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,

            beta1=betas[0],
            beta2=betas[1],
            beta3=betas[2],
            momentum_beta=momentum_beta,
            alpha=alpha,
            step=0,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,

            use_orthogonal_updates=use_orthogonal_updates,
        )
        super().__init__(params, default_args_dict)

        # Normalize parameters at initialization so the first forward pass
        # uses weights that are already on the hypersphere.
        if self.hypersphere_mode is not None:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        is_qkv = self.is_qkv_fn(p)
                        self._normalize(p, p, is_qkv=is_qkv)

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1

            if "momentum_beta" not in group:  # To be able to use old checkpoints.
                group["momentum_beta"] = group["beta1"]
            for p in group["params"]:
                if p.grad is not None:
                    self._param_step(p, group)

        return loss


    def _param_step(self, p, group):
        grad = p.grad
        state = self.state[p]

        # Initialization.
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(grad)
            # TODO: Make it such that we can use ademamix-like updates with muon.
            if not group["use_orthogonal_updates"]:  # Enables g^2 EMA as in adam & ademamix.
                state["exp_avg_sq"] = torch.zeros_like(grad)
                if group["alpha"] != 0:  # Enables slow momentum as in ademamix.
                    state["exp_avg_slow"] = torch.zeros_like(grad)

        exp_avg = state["exp_avg"]
        beta1 = group["beta1"]
        momentum_beta = group["momentum_beta"]
        is_qkv = self.is_qkv_fn(p)

        # TODO: potentially project gradient to tangent space here.
        if self.hypersphere_project:
            grad = self._project(p, grad, is_qkv=is_qkv)

        # Get update direction.
        if group["use_orthogonal_updates"]:  # Muon branch.
            assert emerging_optimizers is not None

            # Weight deacy.
            self._apply_weight_decay_inplace(p, p.grad, group)

            # Update momentum buffer with EMA of gradient
            exp_avg.lerp_(grad, 1 - momentum_beta)

            # Include nesterov momentum
            if self.use_nesterov:
                grad = grad.lerp(exp_avg, momentum_beta)
            else:
                grad = exp_avg

            # Get update.
            with emerging_optimizers.utils.fp32_matmul_precision(self.fp32_matmul_prec):
                group_kwargs = {k: v for k, v in group.items() if k != "params"}
                update = self.orthogonalize(p, grad, **group_kwargs, is_qkv=is_qkv)

        else: # AdamW & Ademamix branch.
            beta2 = group["beta2"]
            exp_avg_sq = state["exp_avg_sq"]

            bias_correction1 = 1.0 - (beta1 ** group["step"])
            bias_correction2 = 1.0 - (beta2 ** group["step"])

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

            if group["alpha"] == 0:  # adam logic.
                update = exp_avg.div(bias_correction1) / denom  # TODO original equation.
                #update = exp_avg.div(bias_correction1 * denom)  # TODO is this equivalent?
            else:  # ademamix logic.
                if group["alpha_warmup"] is None:
                    alpha = group["alpha"]
                else:
                    alpha = linear_warmup_scheduler(group["step"], group["alpha"], alpha_start=0, warmup=group["alpha_warmup"])

                if group["beta3_warmup"] is None:
                    beta3 = group["beta3"]
                else:
                    beta3 = linear_hl_warmup_scheduler(group["step"], group["beta3"], beta_start=beta1, warmup=group["beta3_warmup"])


                exp_avg_slow = state["exp_avg_slow"]
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                update = (exp_avg.div(bias_correction1) + alpha * exp_avg_slow) / denom  # TODO Original equation.
                #update = exp_avg.div(bias_correction1).add_(exp_avg_slow, alpha=alpha).div_(denom)  # TODO is this equivalent?

            self._apply_weight_decay_inplace(p, update, group)

        # Optionally, normalize update.
        if self.hypersphere_mode is not None and self.hypersphere_update:
            self._normalize(p, update, is_qkv=is_qkv)

        # Update parameter.
        lr = group["lr"]
        p.add_(update, alpha=-lr)

        # Optionally, normalize parameter.
        if self.hypersphere_mode is not None:
            self._normalize(p, p, is_qkv=is_qkv)

    def _apply_weight_decay_inplace(self, p, update, group):
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        if weight_decay != 0:
            weight_decay_method = group["weight_decay_method"]
            if weight_decay_method == "decoupled":
                p.add_(p, alpha=-weight_decay*lr)
            elif weight_decay_method == "independent":
                p.add_(p, alpha=-weight_decay)
            else:
                raise ValueError(f"Unknown weight decode method {weight_decay_method}")


    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, ignore_scale: bool = False,
                      is_qkv: bool = False, **kwargs) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and is_qkv:  # type: ignore[misc]
            qs, ks, vs = split_qkv(grad, self.qkv_split_shapes)
            if self.split_qkv_heads:
                qs = merge_heads([self._orthogonalize(q, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for q in split_heads(qs, self.qkv_dim)])
                ks = merge_heads([self._orthogonalize(k, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for k in split_heads(ks, self.qkv_dim)])
                vs = merge_heads([self._orthogonalize(v, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for v in split_heads(vs, self.qkv_dim)])
            else:
                qs = self._orthogonalize(qs, tp_group, partition_dim, ignore_scale=ignore_scale)
                ks = self._orthogonalize(ks, tp_group, partition_dim, ignore_scale=ignore_scale)
                vs = self._orthogonalize(vs, tp_group, partition_dim, ignore_scale=ignore_scale)
            grad = merge_qkv([qs, ks, vs], grad.shape, self.qkv_split_shapes)
        else:
            grad = self._orthogonalize(grad, tp_group, partition_dim, ignore_scale=ignore_scale)
        return grad


    def _orthogonalize(
        self,
        grad: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
        partition_dim: int | None = None,
        ignore_scale: bool = False,
    ) -> torch.Tensor:
        assert grad.ndim == 2
        log_single_rank(
            logger,
            logging.DEBUG,
            f'Orthogonalizing grad with {self.num_ns_steps} steps, {self.coefficient_type} coefficient, '
            f'{self.scale_mode} scale mode, extra_scale_factor={self.extra_scale_factor}',
        )
        size = [grad.size(-2), grad.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth_grad = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode="duplicated" if self.mode == "blockwise" else self.mode,
        )
        scale_factor = get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        if ignore_scale:
            return orth_grad
        return orth_grad * scale_factor * self.extra_scale_factor


    def _normalize(self, p: torch.Tensor, x: torch.Tensor, is_qkv: bool = False):
        if self.hypersphere_mode is None:
            return
        if is_qkv and self.split_qkv:
            qs, ks, vs = split_qkv(x, self.qkv_split_shapes)
            if self.split_qkv_heads and self.hypersphere_mode in {"col", "rowcol", "flat"}:
                # When splitting heads using torch.split, we only get views of the
                # original tensor, meaning the qs tensor gets modified in-place,
                # no need to copy the updated q to qs after.
                for q in split_heads(qs, self.qkv_dim):
                    self._normalize(p, q)
                for k in split_heads(ks, self.qkv_dim):
                    self._normalize(p, k)
                for v in split_heads(vs, self.qkv_dim):
                    self._normalize(p, v)
            else:
                # If hypersphere_mode is row, we don't need to split heads manually as before
                # because each head are just contiguous *rows* in qs, splitting is unnecessary.
                self._normalize(p, qs)
                self._normalize(p, ks)
                self._normalize(p, vs)
            x.copy_(merge_qkv((qs, ks, vs), x.size(), self.qkv_split_shapes))
            return


        if self.hypersphere_radius == "learnable":
            raise NotImplementedError(f"Learnable hypersphere NYI")

        if self.hypersphere_mode == "col":
            dim = 0
        elif self.hypersphere_mode == "row":
            dim = 1
        elif self.hypersphere_mode in {"flat", "rowcol"}:
            dim = None
        else:
            raise ValueError(f"Unknown normalization {self.hypersphere_mode}")

        eps = self.hypersphere_radius if self.hypersphere_soft else self.hypersphere_eps

        if self.hypersphere_mode == "rowcol":
            assert self.hypersphere_kind == "l2"
            assert self.hypersphere_radius == 1.0
            sinkhorn(x, eps=eps)
        elif self.hypersphere_kind == "l2":
            norm = torch.norm(x, dim=dim, keepdim=True).clamp_min(eps)
            #if torch.any(norm < eps):
            #    print("Letsgoo")
            #else:
            #    print("avg norm:", norm.mean())
            x.mul_(self.hypersphere_radius / norm)
        elif self.hypersphere_kind == "spectral":
            assert self.hypersphere_mode == "flat"
            norm = spectral_norm(x).clamp_min(eps)
            x.mul_(self.hypersphere_radius / norm)
        elif self.hypersphere_kind == "orthogonal":  # TODO verify lol.
            assert self.hypersphere_mode == "flat"
            x_normalized = self.hypersphere_radius * self.orthogonalize(p, x, ignore_scale=True, is_qkv=is_qkv)
            x.copy_(x_normalized)
        elif self.hypersphere_kind == "standard":
            mu = x.mean(dim=dim, keepdim=True)
            std = x.std(dim=dim, keepdim=True).clamp_min(eps)
            x.add_(mu).mul_(self.hypersphere_radius / std)
        else:
            raise ValueError(f"Unknown hypersphere_kind {self.hypersphere_kind}")

    def _project(self, p, g, is_qkv: bool = False):
        if self.hypersphere_mode is None or not self.hypersphere_project:
            return
        if is_qkv and self.split_qkv and self.hypersphere_mode != "row":
            p_qs, p_ks, p_vs = split_qkv(p, self.qkv_split_shapes)
            g_qs, g_ks, g_vs = split_qkv(g.copy(), self.qkv_split_shapes)
            if self.split_qkv_heads:
                for p_q, g_q in zip(split_heads(p_qs, self.qkv_dim), split_heads(g_qs, self.qkv_dim)):
                    g_q.copy_(self._project(p_q, g_q))
                for p_k, g_k in zip(split_heads(p_ks, self.qkv_dim), split_heads(g_ks, self.qkv_dim)):
                    g_k.copy_(self._project(p_k, g_k))
                for p_v, g_v in zip(split_heads(p_vs, self.qkv_dim), split_heads(g_vs, self.qkv_dim)):
                    g_v.copy_(self._project(p_v, g_v))
            else:
                # If hypersphere_mode is row, we don't need to split heads manually as before
                # because each head are just contiguous *rows* in qs, splitting is unnecessary.
                g_qs = self._project(p_qs, g_qs)
                g_ks = self._project(p_ks, g_ks)
                g_vs = self._project(p_vs, g_vs)
            return merge_qkv((g_qs, g_ks, g_vs), p.size(), self.qkv_split_shapes)

        if self.hypersphere_mode == "col":
            dim = 0
        elif self.hypersphere_mode == "row":
            dim = 1
        elif self.hypersphere_mode == "flat":
            dim = None
        elif self.hypersphere_mode == "rowcol":
            raise ValueError(f"Project rowcol nyi")
        else:
            raise ValueError(f"Unknown normalization {self.hypersphere_mode}")

        if self.hypersphere_kind != "l2":
            raise ValueError(f"Project {self.hypersphere_kind} nyi")

        dots = torch.sum(p * g, dim=dim, keepdim=True)
        return g - (dots / self.hypersphere_radius**2) * p


def split_qkv(x, shapes: tuple[int, int, int]) -> list[torch.Tensor]:
    # split grouped attention parameters (e.g., QKV, GQA, etc.)
    shape = x.shape
    num_query_groups = shape[0] // sum(shapes)
    qkv = torch.split(
        x.view(num_query_groups, sum(shapes), -1),
        shapes,
        dim=1,
    )
    qkv = [g.reshape(-1, shape[-1]) for g in qkv]
    return qkv


def split_heads(x, head_dim: int) -> tuple[torch.Tensor]:
    return torch.split(x, head_dim, dim=0)


def merge_qkv(qkv, xshape: tuple[int, int], shapes: tuple[int, int, int]) -> torch.Tensor:
    num_query_groups = xshape[0] // sum(shapes)
    qkv = [g.view(num_query_groups, -1, xshape[-1]) for g in qkv]
    return torch.cat(qkv, dim=1).view(xshape)


def merge_heads(xs) -> torch.Tensor:
    return torch.cat(xs, dim=1)


def spectral_norm(x, n_iters: int = 10):
    if x.size(-2) < x.size(-1):
        x = x @ x.T
    else:
        x = x.T @ x

    u = torch.randn(x.size(-1), device=x.device, dtype=x.dtype)
    for _ in range(n_iters):
        u = u / torch.norm(u)
        u = x @ u
    return torch.norm(u)**0.5


def sinkhorn(x, n_iters: int = 10, eps: float = 1e-8):
    for _ in range(n_iters):
        norm_col = torch.norm(x, dim=0, keepdim=True).clamp_min(eps)
        norm_row = torch.norm(x, dim=1, keepdim=True).clamp_min(eps)
        x.div_(norm_col).div_(norm_row)



def get_megatron_master_optimizer(
    config: OptimizerConfig,
    model_chunks: list[MegatronModule],
    config_overrides: Optional[dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # Muon currently use adam config. setting str here to call regular get for adam creation
    # side effect is muon optimizer will have wrong name, i.e. config.optimizer == 'adam'
    config.optimizer = 'adam'

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('master with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('master with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging master with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    # TODO: do we need this anymore?
    def master_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    if not group["use_orthogonal_updates"]:  # Enables g^2 EMA as in adam & ademamix.
                        opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
                        if group["alpha"] != 0:  # Enables slow momentum as in ademamix.
                            opt.state[p]["exp_avg_slow"] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # TODO(deyuf): currently only allow 2D non-embedding weight to avoid breaking
            if (
                (config.hypersphere_embeddings or not getattr(param, 'is_embedding_or_output_parameter', False))
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    master_kwargs = {
        # Common.
        "lr": config.muon_lr_factor * config.lr,
        "weight_decay": config.weight_decay,
        "weight_decay_method": config.weight_decay_method,

        # Adam & Ademamix settings.
        "betas": (config.adam_beta1, config.adam_beta2, config.ademamix_beta3),
        "alpha": config.ademamix_alpha,
        "beta3_warmup": config.ademamix_beta3_warmup,
        "alpha_warmup": config.ademamix_alpha_warmup,
        "eps": config.adam_eps,

        # Hypersphere optimization.
        "hypersphere_mode": config.hypersphere_mode,
        "hypersphere_kind": config.hypersphere_kind,
        "hypersphere_radius": config.hypersphere_radius,
        "hypersphere_update": config.hypersphere_update,
        "hypersphere_project": config.hypersphere_project,

        # Muon.
        "use_orthogonal_updates": config.use_orthogonal_updates,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "split_qkv": config.muon_split_qkv,
        "split_qkv_heads": config.hypersphere_split_heads,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,

        "qkv_split_shapes": qkv_split_shapes,
        "qkv_dim": kv_channels,  # head dim for split_heads when split_qkv_heads=True
    }

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    config_overrides_master = {**config_overrides}
    config_overrides_master[ParamKey(name="*")] = ParamGroupOverride(max_lr=config.muon_lr_factor * config.lr)
    if config.use_orthogonal_updates and not config.use_orthogonal_embeddings:
        config_overrides_master[ParamKey(attr="is_embedding_or_output_parameter")] = ParamGroupOverride(use_orthogonal_updates=False)

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides_master)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = MasterOptimizer(linear_param_groups, **master_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, master_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, master_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = MasterOptimizer(expert_param_groups, **master_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, master_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, master_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [master_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers, config, pg_collection, init_state_fn_list=init_fns
        )
    return ChainedOptimizer(optimizers)
