"""
Here is an original implementation of MARS.
Source: https://github.com/AGI-Arena/MARS

Ported from Andron00e/Megatron-LM origin/muon (f3dfe32f1). Only the approximate variant is
implemented: the exact one needs a second forward/backward on the same batch at the previous
iterate, which train_step cannot provide. All three inner optimizers of the framework are
here (mars_type: mars-adamw, mars-lion, mars-shampoo), written from the reference
implementation's own update_fn, plus MARS-M (mars_type: mars-muon, arXiv:2510.21800), written
from Algorithm 2 of that paper and the reference MARS_M/optimizers/mars_m.py.
"""

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import math

import torch


def exists(val):
    return val is not None


def is_matrix_param(p):
    """Params MARS treats as matrices, mirroring the Muon/MDDecoupling split in muon.py.

    A 3-D param is a grouped expert stack [local_experts, out, in] (OffloadingExpertsMLP under
    --moe-use-inplace-fp8-param; TEGroupedMLP keeps one 2-D `weight{i}` per expert instead), and
    every slice along dim 0 is one expert's own matrix -- the same reading as the `ndim == 3`
    branches in dion.py and neutrino.py.
    """
    assert p.ndim <= 3, (
        f"MARS routes on param.ndim and knows 1-D, 2-D and 3-D expert stacks only, "
        f"got shape {tuple(p.shape)}"
    )
    if p.ndim == 3:
        return True
    return p.ndim == 2 and not getattr(p, 'is_embedding_or_output_parameter', False)


def mars_correction(grad, last_grad, beta1, gamma, clip):
    """c_t = g_t + gamma * beta1 / (1 - beta1) * (g_t - g_{t-1}), clipped to L2 norm `clip`.

    An expert stack is clipped one expert at a time, so each expert gets the clip MARS would
    give it as a standalone matrix and the update does not depend on how many experts the
    stack holds (i.e. on the expert-parallel degree).
    """
    c_t = (grad - last_grad).mul(gamma * (beta1 / (1.0 - beta1))).add(grad)
    if c_t.ndim == 3:
        c_t_norm = c_t.flatten(1).norm(dim=1).view(-1, 1, 1)
        return c_t.mul(torch.clamp(clip / c_t_norm, max=1.0))
    c_t_norm = torch.norm(c_t)
    if c_t_norm > clip:
        c_t = c_t.mul(clip / c_t_norm)
    return c_t


def adamw_denom(exp_avg_sq, beta1, beta2, step, eps):
    """MARS folds the first-moment bias correction into the denominator."""
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    return exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)


def newton_schulz(m, steps=5, eps=1e-7):
    """Quintic Newton-Schulz orthogonalization, the inner step of mars-shampoo.

    An expert stack is orthogonalized one expert at a time -- the batched matmuls do that for
    free, and the pre-scaling is per expert so a quiet expert is not left under-iterated.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    if m.ndim == 3:
        scale = m.flatten(1).norm(dim=1).view(-1, 1, 1)
    else:
        scale = m.norm()
    x = m.div(scale + eps)
    transposed = m.shape[-2] > m.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        xxt = x @ x.transpose(-2, -1)
        bx = xxt @ x
        x = a * x + b * bx + c * xxt @ bx
    if transposed:
        x = x.transpose(-2, -1)
    return x


def shampoo_shape_factor(p):
    """max(1, d_out/d_in)**0.5, the aspect correction the MARS reference applies after
    orthogonalizing (Muon's `shape_scaling` mode, not the `spectral` one)."""
    return max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5


def marsm_scale_factor(p):
    """0.2 * sqrt(max(d_out, d_in)), the Moonlight factor MARS-M applies after orthogonalizing
    (arXiv:2510.21800, Algorithm 2 line 7; `adjust_lr_for_muon` in the reference mars_m.py).

    It is this tree's Muon recipe, `--muon-scale-mode spectral` times `--muon-extra-scale-factor
    0.2`: a semi-orthogonal d_out x d_in matrix has RMS 1/sqrt(max(d_out, d_in)), so the update
    leaves at RMS ~0.2 whatever the shape, like the Muon arm's and unlike an AdamW-type update.
    """
    return 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))


def muon_rms_scale(update, target):
    """Rescale each matrix's update to root-mean-square `target`, per expert for a stack.

    This is what Muon's scale factor does to an orthogonalized update: `spectral` is
    sqrt(max(d_out, d_in)), which cancels the 1/sqrt(max(d_out, d_in)) RMS of a semi-orthogonal
    matrix, so the update leaves Muon at RMS = --muon-extra-scale-factor whatever its shape.
    An AdamW-type update instead carries whatever RMS the second moment gives it, which is why
    multiplying it by Muon's own factor would be sqrt(max(d_out, d_in)) too large.
    """
    if update.ndim == 3:
        rms = update.flatten(1).norm(dim=1).view(-1, 1, 1) / math.sqrt(update[0].numel())
    else:
        rms = update.norm() / math.sqrt(update.numel())
    return update.mul(target / rms.clamp(min=1e-12))


NORM_STATE_KEYS = ("weight_norm", "update_norm")


def matrix_norms(x):
    """Frobenius norm of each matrix: a 0-dim tensor, or [local_experts] for a stack."""
    if x.ndim == 3:
        return x.flatten(1).norm(dim=1)
    return x.norm()


def norm_state_like(p):
    """One fp32 scalar per matrix of `p`, the shape of its recorded norms."""
    return torch.zeros(p.shape[:-2], dtype=torch.float32, device=p.device)


def scale_update_to_weight_norm(update, weight_norm, update_norm):
    """Rescale each matrix's update to the Frobenius norm recorded for its weight (per expert for
    a stack), a zero update staying zero, and record the update's own norm in `update_norm`.

    md_decoupling's --md-normalize-update-to-weight-norm (`_scale_update_to_fixed_norm`,
    `_normalize_muon_update_blocks`): the update's MEASURED norm is matched to the weight norm
    cached at the first step, which supersedes every scalar factor -- there Muon's shape factor,
    here the RMS an AdamW-type update happens to have. lr is then the relative step
    ||dW||_F / ||W_0||_F of every matrix, whatever its shape or init.
    """
    update_norm.copy_(matrix_norms(update))
    scale = torch.where(update_norm > 0, weight_norm / update_norm, torch.zeros_like(update_norm))
    if update.ndim == 3:
        scale = scale.view(-1, 1, 1)
    return update.mul(scale)


def sharded_norm_state(model_param, value, state_key, prefix):
    """dist-checkpointing hook (Float16OptimizerWithFloat16Params.sharded_state_dict): the recorded
    norms are one scalar per matrix, not param-shaped, so they go into the checkpoint as objects
    under the param's key. None sends every other key down the param-shaped path."""
    if state_key not in NORM_STATE_KEYS:
        return None
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    return ShardedObject(
        f"{prefix}.{model_param.key}", value, (1,), (0,), replica_id=model_param.replica_id
    )


@torch.no_grad()
def collect_norm_stats(optimizer):
    """Recorded weight RMS, measured update RMS and their ratio (the multiplier the match applies
    to the update), as equal-weighted per-matrix means, globally and per matrix shape.

    muon_logging's conventions and sink: one value per matrix (per expert for a stack), read from
    replicated state so nothing is reduced, returned as the md_gain_stats dict that training_log
    writes to TensorBoard and W&B.
    """
    wrapped = getattr(optimizer, "chained_optimizers", (optimizer,))
    per_shape = {}
    for opt in (getattr(w, "optimizer", w) for w in wrapped):
        if not getattr(opt, "normalize_update_to_weight_norm", False):
            continue
        for p, state in opt.state.items():
            if "weight_norm" not in state:
                continue
            w, u = state["weight_norm"].flatten(), state["update_norm"].flatten()
            sqrt_numel = math.sqrt(p.shape[-2] * p.shape[-1])
            rows = per_shape.setdefault(f"{p.shape[-2]}x{p.shape[-1]}", [])
            rows.append(
                torch.stack(
                    (w / sqrt_numel, u / sqrt_numel, torch.where(u > 0, w / u, torch.zeros_like(u)))
                )
            )
    if not per_shape:
        return {}
    per_shape = {shape: torch.cat(rows, dim=1) for shape, rows in per_shape.items()}
    stats = {}
    for i, name in enumerate(("weight-rms", "update-rms", "update-scale")):
        stats[f"mars/{name}"] = torch.cat([v[i] for v in per_shape.values()]).mean().item()
        for shape, v in per_shape.items():
            stats[f"mars/{name}/{shape}"] = v[i].mean().item()
    return stats


def update_fn(
    p,
    grad,
    exp_avg,
    exp_avg_sq,
    lr,
    wd,
    beta1,
    beta2,
    last_grad,
    eps,
    step,
    gamma,
    clip,
    mars_type,
    muon_rms_target,
    weight_norm,
    update_norm,
    is_matrix,
    optimize_1d,
    lr_1d_factor,
    betas_1d,
):
    # optimize_1d: use MARS for 1d para, not: use AdamW for 1d para
    if optimize_1d or is_matrix:
        c_t = mars_correction(grad, last_grad, beta1, gamma, clip)
        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
        if mars_type == "mars-lion":
            update = exp_avg.sign()
        elif mars_type == "mars-shampoo" and is_matrix:
            update = newton_schulz(exp_avg.div(1.0 - beta1**step), eps=eps).mul_(
                shampoo_shape_factor(p)
            )
        elif mars_type == "mars-muon" and is_matrix:
            # MARS-M orthogonalizes the raw momentum: no bias correction (Newton-Schulz is scale
            # invariant up to its 1e-7 pre-normalization, which the reference keeps too).
            update = newton_schulz(exp_avg).mul_(marsm_scale_factor(p))
        else:
            # mars-adamw, and the 1-D params of mars-shampoo/mars-muon, which the reference keeps
            # on AdamW.
            exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)
            denom = adamw_denom(exp_avg_sq, beta1, beta2, step, eps)
            update = exp_avg.div(denom)
        if muon_rms_target is not None and is_matrix:
            update = muon_rms_scale(update, muon_rms_target)
        if weight_norm is not None and is_matrix:
            update = scale_update_to_weight_norm(update, weight_norm, update_norm)
        p.data.add_(-lr * torch.mul(p.data, wd).add(update))
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1 - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1 - beta2_1d)
        denom = adamw_denom(exp_avg_sq, beta1_1d, beta2_1d, step, eps)
        p.data.add_(-lr * lr_1d_factor * torch.mul(p.data, wd).add(exp_avg.div(denom)))
    return exp_avg, exp_avg_sq


class MARS(torch.optim.Optimizer):
    """MARS (arXiv:2411.10438), approximate variant, with AdamW on non-matrix params.

    ``mars_type`` picks the inner optimizer applied to the corrected gradient, exactly as in
    the reference implementation: ``mars-adamw``, ``mars-lion`` (the sign of the corrected
    momentum, no second moment), ``mars-shampoo`` (Newton-Schulz orthogonalization times
    max(1, d_out/d_in)**0.5, with 1-D params still on AdamW) or ``mars-muon`` (MARS-M,
    arXiv:2510.21800: Newton-Schulz orthogonalization of the corrected momentum times
    0.2 * sqrt(max(d_out, d_in)), the Moonlight/Muon scale, with 1-D params still on AdamW).

    ``normalize_update_to_weight_norm`` is MuonMD's --md-normalize-update-to-weight-norm on the
    matrix group: each matrix's ||W||_F is recorded at its first step (state ``weight_norm``) and
    every later update direction is rescaled to that norm (its measured norm kept in
    ``update_norm``), superseding whatever scale the inner optimizer gave it. The decoupled
    weight-decay term and the 1-D/embedding/output params are never touched.
    """

    def __init__(
        self,
        params,
        lr=3e-3,
        betas=(0.95, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        gamma=0.025,
        clip=1.0,
        mars_type="mars-adamw",
        muon_rms_target=None,
        normalize_update_to_weight_norm=False,
        optimize_1d=False,
        lr_1d=None,
        betas_1d=(0.9, 0.95),
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if normalize_update_to_weight_norm and muon_rms_target is not None:
            raise ValueError(
                "normalize_update_to_weight_norm and muon_rms_target are two targets for the "
                "same matrix update RMS; set one."
            )
        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo", "mars-muon"], (
            "MARS type not supported"
        )
        if mars_type in ("mars-shampoo", "mars-muon") and muon_rms_target is not None:
            raise ValueError(
                f"{mars_type} already fixes the matrix update scale with its own "
                "orthogonalization and shape factor; muon_rms_target would apply a second one."
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mars_type=mars_type,
            gamma=gamma,
            clip=clip,
            optimize_1d=optimize_1d,
        )
        super(MARS, self).__init__(params, defaults)
        self.eps = eps
        self.update_fn = update_fn
        self.gamma = gamma
        self.clip = clip
        self.mars_type = mars_type
        self.muon_rms_target = muon_rms_target
        self.normalize_update_to_weight_norm = normalize_update_to_weight_norm
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = 1.0 if lr_1d is None else lr_1d / lr
        self.betas_1d = betas_1d

    build_sharded_optimizer_state = staticmethod(sharded_norm_state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "MARS does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["last_grad"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                last_grad = state["last_grad"]
                lr, wd, (beta1, beta2) = group["lr"], group["weight_decay"], group["betas"]
                is_matrix = is_matrix_param(p)
                if self.normalize_update_to_weight_norm and is_matrix and state["step"] == 0:
                    # Measured once, before the first decay/update, like MuonMD's
                    # _cache_fixed_weight_norms; a checkpoint carries it from then on.
                    state["weight_norm"] = matrix_norms(p.data)
                    state["update_norm"] = norm_state_like(p)

                state["step"] += 1
                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    last_grad,
                    self.eps,
                    state["step"],
                    self.gamma,
                    self.clip,
                    self.mars_type,
                    self.muon_rms_target,
                    state.get("weight_norm"),
                    state.get("update_norm"),
                    is_matrix=is_matrix,
                    optimize_1d=self.optimize_1d,
                    lr_1d_factor=self.lr_1d_factor,
                    betas_1d=self.betas_1d,
                )
                # Never alias p.grad here: Megatron reuses one fp32 grad buffer across steps
                # (and rescales it in place when clipping), which would make g_t - g_{t-1} == 0.
                last_grad.copy_(grad)

        return loss
