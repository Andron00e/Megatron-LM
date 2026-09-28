# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer policy for RL phases that resume a pretraining (NTP) checkpoint.

An RL phase (train_rl.py) resumes the full pretraining state: weights, optimizer state and the
OptimizerParamScheduler. These helpers let the RL phase run its own LR (linear warmup, then
constant) instead of the pretraining schedule, start from fresh optimizer moments, use RL-specific
Adam hyperparameters, and switch md_decoupling to a plain AdamW. The pretraining scheduler is not
stepped while RL runs its own LR, so the scheduler state is saved unchanged.

Param-group values written here (lr, weight_decay, betas/beta1/beta2, eps) are saved with the
checkpoint; md_decoupling's gain settings and plain-AdamW switch are object attributes and are not.
"""

from typing import List, Optional

import torch

MOMENT_KEYS = frozenset(
    {
        "exp_avg",
        "exp_avg_sq",
        "row_gain_m",
        "row_gain_v",
        "col_gain_m",
        "col_gain_v",
        "flat_gain_m",
        "flat_gain_v",
        "plain_adamw_exp_avg",
        "plain_adamw_exp_avg_sq",
    }
)
_STEP_KEYS = ("step", "plain_adamw_step")


def base_optimizers(optimizer) -> List[torch.optim.Optimizer]:
    """The torch optimizers inside a (chained / layer-wise / mixed-precision) Megatron optimizer."""
    chained = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    bases = [getattr(opt, "optimizer", opt) for opt in chained]
    return [opt for opt in bases if opt is not None]


def _is_md(opt) -> bool:
    from .md_decoupling import _MDDecouplingBase

    return isinstance(opt, _MDDecouplingBase)


@torch.no_grad()
def reset_moments(optimizer) -> None:
    """Zero every first/second moment (Muon momentum, Adam moments, gain moments) and step count.

    Under the layer-wise distributed optimizer each rank resets the state it owns. Weights, fp32
    master weights and the gains themselves are untouched.
    """
    for opt in base_optimizers(optimizer):
        for state in opt.state.values():
            for key, value in state.items():
                if key in MOMENT_KEYS or key in _STEP_KEYS:
                    if torch.is_tensor(value):
                        value.zero_()
                    else:
                        state[key] = 0
        for group in opt.param_groups:
            for key in _STEP_KEYS:
                if key in group:
                    group[key] = 0


def set_adam_hparams(
    optimizer,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    eps: Optional[float] = None,
) -> None:
    """Override Adam betas / eps on every Adam-style param group and on the md_decoupling gains.

    md_decoupling Muon groups read these values only in plain-AdamW mode; otherwise they are left
    as they are.
    """
    for opt in base_optimizers(optimizer):
        muon_groups = _is_md(opt) and not opt.plain_adamw
        for group in opt.param_groups:
            if muon_groups and group.get("use_orthogonal_updates", False):
                continue
            if "betas" in group:
                old1, old2 = group["betas"]
                group["betas"] = (
                    old1 if beta1 is None else beta1,
                    old2 if beta2 is None else beta2,
                )
            if beta1 is not None and "beta1" in group:
                group["beta1"] = beta1
            if beta2 is not None and "beta2" in group:
                group["beta2"] = beta2
            if eps is not None and "eps" in group:
                group["eps"] = eps
        if hasattr(opt, "gains_betas"):
            old1, old2 = opt.gains_betas
            opt.gains_betas = (old1 if beta1 is None else beta1, old2 if beta2 is None else beta2)
            if eps is not None:
                opt.gains_eps = eps


def set_plain_adamw(optimizer, enabled: bool) -> None:
    """Switch every md_decoupling optimizer to plain AdamW (fresh step count) or back."""
    for opt in base_optimizers(optimizer):
        if _is_md(opt):
            opt.plain_adamw = enabled
            for group in opt.param_groups:
                group["plain_adamw_step"] = 0


def warmup_scale(steps_done: int, warmup_steps: int) -> float:
    """Linear warmup factor for the step after `steps_done` completed RL steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (steps_done + 1) / warmup_steps)


def set_learning_rates(
    optimizer,
    lr: float,
    matrix_lr: float,
    gains_lr: float,
    weight_decay: float,
    scale: float = 1.0,
) -> None:
    """Set absolute RL learning rates, multiplied by `scale`.

    md_decoupling Muon-branch (direction) groups get `matrix_lr`, md_decoupling gains get
    `gains_lr`, and everything else (md_decoupling Adam branch, chained Adam, and every md group
    in plain-AdamW mode) gets `lr`.
    """
    for opt in base_optimizers(optimizer):
        md = _is_md(opt)
        muon_groups = md and not opt.plain_adamw
        for group in opt.param_groups:
            group_lr = scale * (
                matrix_lr if muon_groups and group.get("use_orthogonal_updates", False) else lr
            )
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(group_lr)
            else:
                group["lr"] = group_lr
            group["weight_decay"] = weight_decay * group.get("wd_mult", 1.0)
        if hasattr(opt, "gains_lr_override"):
            opt.gains_lr_override = scale * gains_lr
            opt.gains_weight_decay = weight_decay
