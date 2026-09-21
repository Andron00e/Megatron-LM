"""
Mu2MARS: a second (outer) EMA on top of the MARS corrected momentum.

Recovered from Andron00e/learning-at-scale (`src/optim/mu2mars.py`, a971981) -- whiteboard
option (2) -- with the bias corrections re-derived (the original divided the outer EMA by the
inner EMA's bias correction) and the gradient-difference clip made configurable.

Per matrix parameter, with g_t the current gradient and g_{t-1} the previous iteration's
gradient (MARS "approx"; the exact same-batch variant needs a second forward/backward at the
previous iterate and is not implementable inside train_step):

    c_t = g_t + gamma * beta1 / (1 - beta1) * (g_t - g_{t-1}),  clipped to ||c_t||_2 <= clip
    m_t = beta1 * m_{t-1} + (1 - beta1) * c_t,      m_hat = m_t / (1 - beta1^t)
    mu_t = beta3 * mu_{t-1} + (1 - beta3) * m_hat,  mu_hat = mu_t / (1 - beta3^t)
    v_t = beta2 * v_{t-1} + (1 - beta2) * c_t^2,    v_hat = v_t / (1 - beta2^t)
    p <- p - lr * mu_hat / (sqrt(v_hat) + eps) - lr * wd * p

With beta3 = 0 this is exactly MARS. Note the role swap behind the recorded configs:
MARS(beta1=0.95, gamma=0.025) has gradient-difference coefficient gamma*beta1 = 0.02375 and
Mu2MARS(beta1=0.025, gamma=1.0) has gamma*beta1 = 0.025, so the two match in absolute
correction strength while in Mu2MARS the inner momentum is essentially off (beta1 = 0.025 is
the STORM correction weight, not a momentum) and the heavy momentum moves to beta3.
"""

import math

import torch

from .mars import adamw_denom, exists, is_matrix_param, mars_correction


def update_fn(
    p,
    grad,
    exp_avg,
    exp_avg_sq,
    mu_avg,
    lr,
    wd,
    beta1,
    beta2,
    beta3,
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
    # optimize_1d: use Mu2MARS for 1d para, not: use AdamW for 1d para. 1d params carry no
    # variance-reduction term (gamma = 0) -- g_t - g_{t-1} is a matrix-level correction.
    if optimize_1d or is_matrix:
        c_t = mars_correction(grad, last_grad, beta1, gamma if is_matrix else 0.0, clip)
        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
        mu_avg.mul_(beta3).add_(exp_avg.div(1.0 - beta1**step), alpha=1.0 - beta3)
        exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt().mul(1 / math.sqrt(1.0 - beta2**step)).add(eps)
        mu_hat = mu_avg.div(1.0 - beta3**step)
        p.data.add_(-lr * torch.mul(p.data, wd).add(mu_hat.div(denom)))
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1 - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1 - beta2_1d)
        denom = adamw_denom(exp_avg_sq, beta1_1d, beta2_1d, step, eps)
        p.data.add_(-lr * lr_1d_factor * torch.mul(p.data, wd).add(exp_avg.div(denom)))
    return exp_avg, exp_avg_sq, mu_avg


class Mu2MARS(torch.optim.Optimizer):
    """MARS with double momentum: an outer EMA (beta3) over the bias-corrected inner EMA."""

    def __init__(
        self,
        params,
        lr=3e-3,
        betas=(0.025, 0.99, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        gamma=1.0,
        clip=1.0,
        optimize_1d=False,
        lr_1d=None,
        betas_1d=(0.9, 0.95),
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError("Invalid beta parameter at index {}: {}".format(i, beta))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            clip=clip,
            optimize_1d=optimize_1d,
        )
        super(Mu2MARS, self).__init__(params, defaults)
        self.eps = eps
        self.update_fn = update_fn
        self.gamma = gamma
        self.clip = clip
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
                        "Mu2MARS does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["mu_avg"] = torch.zeros_like(p.data)
                    state["last_grad"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                mu_avg, last_grad = state["mu_avg"], state["last_grad"]
                lr, wd, (beta1, beta2, beta3) = (
                    group["lr"],
                    group["weight_decay"],
                    group["betas"],
                )

                state["step"] += 1
                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    mu_avg,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    beta3,
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
                # See mars.py: never alias p.grad, Megatron reuses the fp32 grad buffer.
                last_grad.copy_(grad)

        return loss
