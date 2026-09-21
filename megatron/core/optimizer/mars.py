"""
Here is an original implementation of MARS.
Source: https://github.com/AGI-Arena/MARS

Ported from Andron00e/Megatron-LM origin/muon (f3dfe32f1). Only the approximate variant is
implemented: the exact one needs a second forward/backward on the same batch at the previous
iterate, which train_step cannot provide.
"""

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import math

import torch


def exists(val):
    return val is not None


def is_matrix_param(p):
    """Params MARS treats as matrices, mirroring the Muon/MDDecoupling split in muon.py."""
    return p.ndim == 2 and not getattr(p, 'is_embedding_or_output_parameter', False)


def mars_correction(grad, last_grad, beta1, gamma, clip):
    """c_t = g_t + gamma * beta1 / (1 - beta1) * (g_t - g_{t-1}), clipped to L2 norm `clip`."""
    c_t = (grad - last_grad).mul(gamma * (beta1 / (1.0 - beta1))).add(grad)
    c_t_norm = torch.norm(c_t)
    if c_t_norm > clip:
        c_t = c_t.mul(clip / c_t_norm)
    return c_t


def adamw_denom(exp_avg_sq, beta1, beta2, step, eps):
    """MARS folds the first-moment bias correction into the denominator."""
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    return exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)


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
    is_matrix,
    optimize_1d,
    lr_1d_factor,
    betas_1d,
):
    # optimize_1d: use MARS for 1d para, not: use AdamW for 1d para
    if optimize_1d or is_matrix:
        c_t = mars_correction(grad, last_grad, beta1, gamma, clip)
        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)
        denom = adamw_denom(exp_avg_sq, beta1, beta2, step, eps)
        p.data.add_(-lr * torch.mul(p.data, wd).add(exp_avg.div(denom)))
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1 - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1 - beta2_1d)
        denom = adamw_denom(exp_avg_sq, beta1_1d, beta2_1d, step, eps)
        p.data.add_(-lr * lr_1d_factor * torch.mul(p.data, wd).add(exp_avg.div(denom)))
    return exp_avg, exp_avg_sq


class MARS(torch.optim.Optimizer):
    """MARS (arXiv:2411.10438), approximate variant, with AdamW on non-matrix params."""

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
        assert mars_type in ["mars-adamw"], "MARS type not supported"
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
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = 1.0 if lr_1d is None else lr_1d / lr
        self.betas_1d = betas_1d

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
                    is_matrix=is_matrix_param(p),
                    optimize_1d=self.optimize_1d,
                    lr_1d_factor=self.lr_1d_factor,
                    betas_1d=self.betas_1d,
                )
                # Never alias p.grad here: Megatron reuses one fp32 grad buffer across steps
                # (and rescales it in place when clipping), which would make g_t - g_{t-1} == 0.
                last_grad.copy_(grad)

        return loss
