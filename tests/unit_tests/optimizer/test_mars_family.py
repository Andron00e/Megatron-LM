# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU reference tests for the MARS / Mu2MARS / AdEMAMix optimizers.

The reference updates below are written from the published algorithms and loop over single
elements on purpose: they must not share code with megatron.core.optimizer.{mars,mu2mars,
ademamix}, otherwise a bug in the tensorized implementation would cancel out.
"""

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.ademamix import AdEMAMix
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.mars import (
    MARS,
    NORM_STATE_KEYS,
    collect_norm_stats,
    is_matrix_param,
    marsm_scale_factor,
    norm_state_like,
    shampoo_shape_factor,
)
from megatron.core.optimizer.mu2mars import Mu2MARS

TOL = dict(atol=1e-6, rtol=1e-6)


def make_model(seed=1234):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 4))


def make_grads(model, seed):
    return grads_like(list(model.parameters()), seed)


def grads_like(params, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(p.shape, generator=g) for p in params]


def set_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = g.clone()


def as_state(params):
    return [
        {
            'p': p.detach().flatten().double().tolist(),
            'm': [0.0] * p.numel(),
            'v': [0.0] * p.numel(),
            'mu': [0.0] * p.numel(),
            'w': p.detach().flatten().double().tolist(),
            'lg': [0.0] * p.numel(),
            'ndim': p.ndim,
        }
        for p in params
    ]


def corrected(st, g, beta1, gamma, clip):
    c = [g[i] + gamma * (beta1 / (1.0 - beta1)) * (g[i] - st['lg'][i]) for i in range(len(g))]
    norm = math.sqrt(sum(x * x for x in c))
    if norm > clip:
        c = [x * (clip / norm) for x in c]
    return c


def ref_adamw_element(st, i, g_i, lr, wd, beta1, beta2, eps, step):
    st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * g_i
    st['v'][i] = beta2 * st['v'][i] + (1.0 - beta2) * g_i * g_i
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    denom = math.sqrt(st['v'][i] / bc2) + eps
    st['p'][i] -= lr * (wd * st['p'][i] + (st['m'][i] / bc1) / denom)


def ref_mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, optimize_1d, lr_1d_factor,
                  betas_1d):
    beta1, beta2 = betas
    for st, grad in zip(states, grads):
        g = grad.flatten().double().tolist()
        if optimize_1d or st['ndim'] == 2:
            c = corrected(st, g, beta1, gamma, clip)
            for i in range(len(g)):
                ref_adamw_element(st, i, c[i], lr, wd, beta1, beta2, eps, step)
        else:
            for i in range(len(g)):
                ref_adamw_element(st, i, g[i], lr * lr_1d_factor, wd, *betas_1d, eps, step)
        st['lg'] = g


def ref_mu2mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, optimize_1d,
                     lr_1d_factor, betas_1d):
    beta1, beta2, beta3 = betas
    for st, grad in zip(states, grads):
        g = grad.flatten().double().tolist()
        if optimize_1d or st['ndim'] == 2:
            c = corrected(st, g, beta1, gamma if st['ndim'] == 2 else 0.0, clip)
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            bc3 = 1.0 - beta3**step
            for i in range(len(g)):
                st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * c[i]
                st['mu'][i] = beta3 * st['mu'][i] + (1.0 - beta3) * (st['m'][i] / bc1)
                st['v'][i] = beta2 * st['v'][i] + (1.0 - beta2) * c[i] * c[i]
                denom = math.sqrt(st['v'][i] / bc2) + eps
                st['p'][i] -= lr * (wd * st['p'][i] + (st['mu'][i] / bc3) / denom)
        else:
            for i in range(len(g)):
                ref_adamw_element(st, i, g[i], lr * lr_1d_factor, wd, *betas_1d, eps, step)
        st['lg'] = g


def ref_mu2mars_anytime_step(states, grads, step, lr, wd, betas, eps, gamma, clip,
                             anytime_gamma, optimize_1d, lr_1d_factor, betas_1d):
    beta1, beta2 = betas[0], betas[1]
    for st, grad in zip(states, grads):
        g = grad.flatten().double().tolist()
        if optimize_1d or st['ndim'] == 2:
            c = corrected(st, g, beta1, gamma if st['ndim'] == 2 else 0.0, clip)
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            for i in range(len(g)):
                st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * c[i]
                st['v'][i] = beta2 * st['v'][i] + (1.0 - beta2) * c[i] * c[i]
                denom = math.sqrt(st['v'][i] / bc2) + eps
                st['w'][i] -= lr * (wd * st['w'][i] + (st['m'][i] / bc1) / denom)
                st['p'][i] = anytime_gamma * st['w'][i] + (1.0 - anytime_gamma) * st['p'][i]
        else:
            for i in range(len(g)):
                ref_adamw_element(st, i, g[i], lr * lr_1d_factor, wd, *betas_1d, eps, step)
        st['lg'] = g


def ref_ademamix_step(states, grads, step, lr, wd, betas, alpha, eps):
    beta1, beta2, beta3 = betas
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    for st, grad in zip(states, grads):
        g = grad.flatten().double().tolist()
        for i in range(len(g)):
            st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * g[i]
            st['mu'][i] = beta3 * st['mu'][i] + (1.0 - beta3) * g[i]
            st['v'][i] = beta2 * st['v'][i] + (1.0 - beta2) * g[i] * g[i]
            denom = math.sqrt(st['v'][i] / bc2) + eps
            update = (st['m'][i] / bc1 + alpha * st['mu'][i]) / denom + wd * st['p'][i]
            st['p'][i] -= lr * update


def assert_matches(params, states):
    for p, st in zip(params, states):
        torch.testing.assert_close(
            p.detach().double().flatten(), torch.tensor(st['p'], dtype=torch.float64), **TOL
        )


def test_mars_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.1, 1e-8, 0.025, 1.0
    betas, betas_1d = (0.95, 0.99), (0.9, 0.95)
    opt = MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
               betas_1d=betas_1d)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        opt.step()
        ref_mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, False, 1.0, betas_1d)
    assert_matches(params, states)


def test_mars_optimize_1d_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.0, 1e-8, 0.05, 0.5
    betas = (0.9, 0.95)
    opt = MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
               optimize_1d=True)
    for step in range(1, 11):
        grads = make_grads(model, seed=100 + step)
        set_grads(params, grads)
        opt.step()
        ref_mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, True, 1.0, (0.9, 0.95))
    assert_matches(params, states)


def test_mu2mars_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.1, 1e-8, 1.0, 1.0
    betas, betas_1d = (0.025, 0.99, 0.95), (0.9, 0.95)
    opt = Mu2MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
                  betas_1d=betas_1d)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        opt.step()
        ref_mu2mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, False, 1.0, betas_1d)
    assert_matches(params, states)


def test_mu2mars_optimize_1d_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.05, 1e-8, 1.0, 2.0
    betas = (0.1, 0.99, 0.9)
    opt = Mu2MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
                  optimize_1d=True)
    for step in range(1, 11):
        grads = make_grads(model, seed=200 + step)
        set_grads(params, grads)
        opt.step()
        ref_mu2mars_step(states, grads, step, lr, wd, betas, eps, gamma, clip, True, 1.0,
                         (0.9, 0.95))
    assert_matches(params, states)


def test_mu2mars_anytime_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.1, 1e-8, 1.0, 1.0
    betas, betas_1d, anytime_gamma = (0.025, 0.99, 0.95), (0.9, 0.95), 0.1
    opt = Mu2MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
                  variant='anytime', anytime_gamma=anytime_gamma, betas_1d=betas_1d)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        opt.step()
        ref_mu2mars_anytime_step(states, grads, step, lr, wd, betas, eps, gamma, clip,
                                 anytime_gamma, False, 1.0, betas_1d)
    assert_matches(params, states)


def test_mu2mars_anytime_optimize_1d_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-2, 0.05, 1e-8, 1.0, 2.0
    betas, anytime_gamma = (0.1, 0.99, 0.9), 0.25
    opt = Mu2MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
                  variant='anytime', anytime_gamma=anytime_gamma, optimize_1d=True)
    for step in range(1, 11):
        grads = make_grads(model, seed=300 + step)
        set_grads(params, grads)
        opt.step()
        ref_mu2mars_anytime_step(states, grads, step, lr, wd, betas, eps, gamma, clip,
                                 anytime_gamma, True, 1.0, (0.9, 0.95))
    assert_matches(params, states)


def test_mu2mars_anytime_gamma_one_equals_mars():
    """x_{t+1} = w_{t+1} kills the averaging, so the 2D rule is exactly MARS."""
    model = make_model()
    params = list(model.parameters())
    reference = [p.detach().clone().requires_grad_(True) for p in params]
    lr, wd, eps, gamma, clip = 1e-2, 0.1, 1e-8, 0.025, 1.0
    opt = Mu2MARS(params, lr=lr, betas=(0.95, 0.99, 0.5), eps=eps, weight_decay=wd, gamma=gamma,
                  clip=clip, variant='anytime', anytime_gamma=1.0)
    ref = MARS(reference, lr=lr, betas=(0.95, 0.99), eps=eps, weight_decay=wd, gamma=gamma,
               clip=clip)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        set_grads(reference, grads)
        opt.step()
        ref.step()
    for p, q in zip(params, reference):
        torch.testing.assert_close(p.detach(), q.detach(), **TOL)


def test_mu2mars_anytime_w_is_not_aliased():
    """p is the query sequence x_t and state['w'] the descent sequence; they must diverge."""
    model = make_model()
    params = list(model.parameters())
    opt = Mu2MARS(params, lr=1e-2, weight_decay=0.1, variant='anytime', anytime_gamma=0.1)
    for step in range(1, 4):
        set_grads(params, make_grads(model, seed=step))
        opt.step()
    matrices = [p for p in params if p.ndim == 2]
    assert matrices
    for p in matrices:
        w = opt.state[p]['w']
        assert w.data_ptr() != p.data_ptr()
        assert not torch.allclose(w, p.detach())
    for p in params:
        if p.ndim != 2:
            assert 'w' not in opt.state[p]


def test_ademamix_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, alpha = 1e-2, 0.1, 1e-8, 8.0
    betas = (0.9, 0.999, 0.9999)
    opt = AdEMAMix(params, lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=wd)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        opt.step()
        ref_ademamix_step(states, grads, step, lr, wd, betas, alpha, eps)
    assert_matches(params, states)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_last_grad_is_not_aliased(cls):
    """Megatron hands the optimizer the same fp32 grad buffer every step and rescales it in
    place when clipping; if last_grad aliases it, g_t - g_{t-1} is identically zero."""
    model = make_model()
    params = list(model.parameters())
    opt = cls(params, lr=1e-2)
    grads = make_grads(model, seed=7)
    set_grads(params, grads)
    buffers = [p.grad for p in params]
    opt.step()
    saved = [opt.state[p]['last_grad'].clone() for p in params]
    for buf in buffers:
        buf.mul_(0.0).add_(3.14)
    for p, ref in zip(params, saved):
        assert opt.state[p]['last_grad'].data_ptr() != p.grad.data_ptr()
        torch.testing.assert_close(opt.state[p]['last_grad'], ref)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_1d_and_embedding_params_take_adamw_path(cls):
    """1D params, and 2D params tagged as embedding/output, must follow plain AdamW."""
    torch.manual_seed(5)
    linear = torch.nn.Parameter(torch.randn(4, 3))
    embedding = torch.nn.Parameter(torch.randn(4, 3))
    embedding.is_embedding_or_output_parameter = True
    bias = torch.nn.Parameter(torch.randn(4))
    params = [linear, embedding, bias]
    lr, wd, eps, betas_1d = 1e-2, 0.1, 1e-8, (0.9, 0.95)
    opt = cls(params, lr=lr, weight_decay=wd, eps=eps, betas_1d=betas_1d)

    adamw_params = [embedding.detach().clone(), bias.detach().clone()]
    for q in adamw_params:
        q.requires_grad_(True)
    adamw = torch.optim.AdamW(adamw_params, lr=lr, betas=betas_1d, eps=eps, weight_decay=wd)

    for step in range(1, 11):
        g = torch.Generator().manual_seed(step)
        grads = [torch.randn(p.shape, generator=g) for p in params]
        set_grads(params, grads)
        opt.step()
        set_grads(adamw_params, [grads[1], grads[2]])
        adamw.step()

    torch.testing.assert_close(embedding.detach(), adamw_params[0].detach(), **TOL)
    torch.testing.assert_close(bias.detach(), adamw_params[1].detach(), **TOL)
    assert not torch.allclose(linear.detach(), embedding.detach())


def test_mars_gamma_zero_equals_adamw():
    """gamma = 0 kills the variance-reduction term, so MARS collapses to AdamW everywhere.

    The c_t clip has to be disabled too: it is applied to the corrected gradient whatever
    gamma is, and a raw gradient of norm > clip would be rescaled.
    """
    model = make_model()
    params = list(model.parameters())
    reference = [p.detach().clone().requires_grad_(True) for p in params]
    lr, wd, eps, betas = 1e-2, 0.1, 1e-8, (0.9, 0.95)
    opt = MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=0.0,
               clip=float('inf'), optimize_1d=True)
    adamw = torch.optim.AdamW(reference, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        set_grads(reference, grads)
        opt.step()
        adamw.step()
    for p, q in zip(params, reference):
        torch.testing.assert_close(p.detach(), q.detach(), **TOL)


def test_mu2mars_beta3_zero_equals_mars():
    """With no outer EMA, mu_hat == m_hat and Mu2MARS is exactly MARS."""
    model = make_model()
    params = list(model.parameters())
    reference = [p.detach().clone().requires_grad_(True) for p in params]
    lr, wd, eps, gamma, clip = 1e-2, 0.1, 1e-8, 0.025, 1.0
    opt = Mu2MARS(params, lr=lr, betas=(0.95, 0.99, 0.0), eps=eps, weight_decay=wd, gamma=gamma,
                  clip=clip)
    ref = MARS(reference, lr=lr, betas=(0.95, 0.99), eps=eps, weight_decay=wd, gamma=gamma,
               clip=clip)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        set_grads(reference, grads)
        opt.step()
        ref.step()
    for p, q in zip(params, reference):
        torch.testing.assert_close(p.detach(), q.detach(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS, AdEMAMix])
def test_state_dict_round_trip(cls):
    check_state_dict_round_trip(lambda ps: cls(ps, lr=1e-2, weight_decay=0.1))


def test_mu2mars_anytime_state_dict_round_trip():
    check_state_dict_round_trip(
        lambda ps: Mu2MARS(ps, lr=1e-2, weight_decay=0.1, variant='anytime', anytime_gamma=0.1)
    )


def check_state_dict_round_trip(make_opt, params=None):
    if params is None:
        params = list(make_model().parameters())
    opt = make_opt(params)
    for step in range(1, 4):
        set_grads(params, grads_like(params, seed=step))
        opt.step()

    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    opt2 = make_opt(resumed)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))

    grads = grads_like(params, seed=99)
    set_grads(params, grads)
    set_grads(resumed, grads)
    opt.step()
    opt2.step()
    for p, q in zip(params, resumed):
        torch.testing.assert_close(p.detach(), q.detach())
    for p, q in zip(params, resumed):
        assert opt.state[p]['step'] == opt2.state[q]['step'] == 4


# ---- mars_type: the lion and shampoo inner optimizers ----


def ref_mars_lion_step(states, grads, step, lr, wd, betas, eps, gamma, clip, betas_1d):
    """MARS-Lion: the sign of the corrected momentum, no second moment, no bias correction."""
    beta1, beta2 = betas
    for st, grad in zip(states, grads):
        g = grad.flatten().double().tolist()
        if st['ndim'] == 2:
            c = corrected(st, g, beta1, gamma, clip)
            for i in range(len(g)):
                st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * c[i]
                sign = 0.0 if st['m'][i] == 0.0 else math.copysign(1.0, st['m'][i])
                st['p'][i] -= lr * (wd * st['p'][i] + sign)
        else:
            for i in range(len(g)):
                ref_adamw_element(st, i, g[i], lr, wd, *betas_1d, eps, step)
        st['lg'] = g


def test_mars_lion_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-3, 0.1, 1e-8, 0.025, 1.0
    betas, betas_1d = (0.95, 0.99), (0.9, 0.95)
    opt = MARS(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, gamma=gamma, clip=clip,
               mars_type='mars-lion', betas_1d=betas_1d)
    for step in range(1, 11):
        grads = make_grads(model, seed=step)
        set_grads(params, grads)
        opt.step()
        ref_mars_lion_step(states, grads, step, lr, wd, betas, eps, gamma, clip, betas_1d)
    assert_matches(params, states)


def test_mars_lion_moves_every_matrix_weight_by_exactly_lr():
    """With wd = 0 the 2-D step is a pure sign vector, so every element moves by lr."""
    model = make_model()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    lr = 1e-3
    opt = MARS(params, lr=lr, weight_decay=0.0, mars_type='mars-lion')
    set_grads(params, make_grads(model, seed=7))
    opt.step()
    for p, b in zip(params, before):
        if p.ndim != 2:
            continue
        step = (b - p.detach()).abs()
        assert torch.isfinite(step).all()
        torch.testing.assert_close(step, torch.full_like(step, lr), **TOL)


def test_mars_shampoo_orthogonalizes_the_matrix_update():
    """The 2-D update is Newton-Schulz's U S' V^T times max(1, d_out/d_in)**0.5, so its singular
    values sit around that factor; 1-D params stay on the reference's AdamW branch."""
    g = torch.Generator().manual_seed(31)
    matrix = torch.nn.Parameter(torch.randn(8, 4, generator=g))
    vector = torch.nn.Parameter(torch.randn(6, generator=g))
    params = [matrix, vector]
    plain = [q.detach().clone().requires_grad_(True) for q in params]
    lr = 1e-2
    opt = MARS(params, lr=lr, weight_decay=0.0, mars_type='mars-shampoo')
    ref = MARS(plain, lr=lr, weight_decay=0.0)
    before = matrix.detach().clone()
    grads = grads_like(params, seed=32)
    set_grads(params, grads)
    set_grads(plain, grads)
    opt.step()
    ref.step()
    update = (before - matrix.detach()) / lr
    assert torch.isfinite(update).all()
    factor = shampoo_shape_factor(matrix)
    svals = torch.linalg.svdvals(update.double())
    assert svals.min() > 0.4 * factor and svals.max() < 1.6 * factor
    torch.testing.assert_close(vector.detach(), plain[1].detach(), **TOL)
    assert not torch.allclose(matrix.detach(), plain[0].detach())


def test_mars_shampoo_expert_stack_matches_independent_experts():
    """Newton-Schulz and its pre-scaling are per expert, so a stack moves as separate matrices."""
    stacked = expert_stack(size_out=6, size_in=4)
    num_experts = stacked.shape[0]
    singles = [torch.nn.Parameter(stacked.detach()[i].clone()) for i in range(num_experts)]
    kwargs = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, clip=0.5, mars_type='mars-shampoo')
    opt = MARS([stacked], **kwargs)
    refs = [MARS([s], **kwargs) for s in singles]
    for step in range(1, 6):
        grad = grads_like([stacked], seed=600 + step)[0]
        stacked.grad = grad.clone()
        opt.step()
        for i, (single, ref) in enumerate(zip(singles, refs)):
            single.grad = grad[i].clone()
            ref.step()
    assert torch.isfinite(stacked.detach()).all()
    for i, single in enumerate(singles):
        torch.testing.assert_close(stacked.detach()[i], single.detach(), **TOL)


def test_unknown_mars_type_is_rejected():
    g = torch.Generator().manual_seed(33)
    with pytest.raises(AssertionError, match='MARS type'):
        MARS([torch.nn.Parameter(torch.randn(4, 3, generator=g))], lr=1e-2, mars_type='mars-adam')


# ---- mars_type mars-muon: MARS-M (arXiv:2510.21800, Algorithm 2) ----


def ref_mars_muon_step(states, grads, step, lr, wd, beta1, eps, gamma, clip, betas_1d):
    """MARS-M written from Algorithm 2: M_t = beta M_{t-1} + (1 - beta) Clip(C_t, clip), no bias
    correction, then X_{t+1} = X_t - lr (0.2 sqrt(max(m, n)) NewtonSchulz(M_t) + wd X_t); 1-D
    params on AdamW. The exact polar factor U V^T stands in for Newton-Schulz here, so the
    reference returns per-matrix (m_t, U V^T) for the caller to compare directions against."""
    out = {}
    for idx, (st, grad) in enumerate(zip(states, grads)):
        g = grad.flatten().double().tolist()
        if st['ndim'] == 2:
            c = corrected(st, g, beta1, gamma, clip)
            for i in range(len(g)):
                st['m'][i] = beta1 * st['m'][i] + (1.0 - beta1) * c[i]
            m = torch.tensor(st['m'], dtype=torch.float64).view(grad.shape)
            u, _, vh = torch.linalg.svd(m, full_matrices=False)
            out[idx] = (m, u @ vh)
            # p is not advanced here: the direction check below is what the update is held to.
        else:
            for i in range(len(g)):
                ref_adamw_element(st, i, g[i], lr, wd, *betas_1d, eps, step)
        st['lg'] = g
    return out


def assert_polar_aligned(update, polar, factor):
    """update == factor * U S' V^T for the reference's U V^T and some positive diagonal S' (what
    the quintic Newton-Schulz delivers, S' in roughly [0.7, 1.3]): then update @ polar^T / factor
    = U S' U^T is symmetric positive definite and the singular values of update / factor are S'."""
    a = update.double() @ polar.T / factor
    torch.testing.assert_close(a, a.T, atol=1e-4, rtol=1e-4)
    assert torch.linalg.eigvalsh((a + a.T) / 2).min() > 0.3
    svals = torch.linalg.svdvals(update.double() / factor)
    assert svals.min() > 0.5 and svals.max() < 1.5


def test_mars_muon_matches_reference():
    model = make_model()
    params = list(model.parameters())
    states = as_state(params)
    lr, wd, eps, gamma, clip = 1e-3, 0.1, 1e-8, 0.025, 1.0
    beta1, betas_1d = 0.95, (0.9, 0.95)
    opt = MARS(params, lr=lr, betas=(beta1, 0.99), eps=eps, weight_decay=wd, gamma=gamma,
               clip=clip, mars_type='mars-muon', betas_1d=betas_1d)
    for step in range(1, 6):
        grads = make_grads(model, seed=step)
        before = [p.detach().clone() for p in params]
        set_grads(params, grads)
        opt.step()
        refs = ref_mars_muon_step(states, grads, step, lr, wd, beta1, eps, gamma, clip, betas_1d)
        for idx, (m, polar) in refs.items():
            p = params[idx]
            torch.testing.assert_close(opt.state[p]['exp_avg'].double(), m, **TOL)
            update = (before[idx] - p.detach()) / lr - wd * before[idx]
            assert_polar_aligned(update, polar, marsm_scale_factor(p))
    for p, st in zip(params, states):
        if p.ndim != 2:
            torch.testing.assert_close(
                p.detach().double().flatten(), torch.tensor(st['p'], dtype=torch.float64), **TOL
            )


def test_mars_muon_orthogonalizes_the_matrix_update():
    """The 2-D update is Newton-Schulz's U S' V^T times 0.2 sqrt(max(d_out, d_in)), so its singular
    values sit around that factor and its RMS around 0.2; shapes are preserved and 1-D params stay
    on the reference's AdamW branch, bit-identical to mars-adamw's."""
    g = torch.Generator().manual_seed(35)
    tall = torch.nn.Parameter(torch.randn(8, 4, generator=g))
    wide = torch.nn.Parameter(torch.randn(3, 9, generator=g))
    vector = torch.nn.Parameter(torch.randn(6, generator=g))
    params = [tall, wide, vector]
    plain = [q.detach().clone().requires_grad_(True) for q in params]
    lr = 1e-2
    opt = MARS(params, lr=lr, weight_decay=0.0, mars_type='mars-muon')
    ref = MARS(plain, lr=lr, weight_decay=0.0)
    before = [p.detach().clone() for p in params]
    grads = grads_like(params, seed=36)
    set_grads(params, grads)
    set_grads(plain, grads)
    opt.step()
    ref.step()
    for p, b in zip(params[:2], before[:2]):
        assert p.shape == b.shape
        update = (b - p.detach()) / lr
        assert torch.isfinite(update).all()
        factor = marsm_scale_factor(p)
        assert factor == 0.2 * math.sqrt(max(p.shape))
        svals = torch.linalg.svdvals(update.double())
        assert svals.min() > 0.5 * factor and svals.max() < 1.5 * factor
        rms = update.double().norm() / math.sqrt(p.numel())
        assert 0.5 * 0.2 < rms < 1.5 * 0.2
    torch.testing.assert_close(vector.detach(), plain[2].detach(), **TOL)
    assert not torch.allclose(tall.detach(), plain[0].detach())


def test_mars_muon_update_rms_is_the_muon_arm_scale_at_600m_shapes():
    """0.2 sqrt(max(d_out, d_in)) is --muon-scale-mode spectral times --muon-extra-scale-factor
    0.2, the mu2 Muon arm's recipe: the matrix update leaves at RMS ~0.2 at every 600m-llama
    shape (F137), where the bare factor is 32-75x."""
    g = torch.Generator().manual_seed(37)
    lr = 1e-3
    for shape in [(1024, 1024), (2048, 1024), (5632, 1024), (1024, 2816)]:
        p = torch.nn.Parameter(torch.randn(*shape, generator=g) * 0.02)
        opt = MARS([p], lr=lr, weight_decay=0.0, mars_type='mars-muon')
        p.grad = torch.randn(*shape, generator=g) * 1e-3
        before = p.detach().clone()
        opt.step()
        rms = ((before - p.detach()) / lr).double().norm() / math.sqrt(p.numel())
        assert 0.7 * 0.2 < rms < 1.3 * 0.2, (shape, rms)


def test_mars_muon_expert_stack_matches_independent_experts():
    stacked = expert_stack(size_out=6, size_in=4)
    num_experts = stacked.shape[0]
    singles = [torch.nn.Parameter(stacked.detach()[i].clone()) for i in range(num_experts)]
    kwargs = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, clip=0.5, mars_type='mars-muon')
    opt = MARS([stacked], **kwargs)
    refs = [MARS([s], **kwargs) for s in singles]
    for step in range(1, 6):
        grad = grads_like([stacked], seed=900 + step)[0]
        stacked.grad = grad.clone()
        opt.step()
        for i, (single, ref) in enumerate(zip(singles, refs)):
            single.grad = grad[i].clone()
            ref.step()
    assert torch.isfinite(stacked.detach()).all()
    for i, single in enumerate(singles):
        torch.testing.assert_close(stacked.detach()[i], single.detach(), **TOL)


def test_mars_muon_is_deterministic():
    model = make_model()
    params = list(model.parameters())
    twin = [p.detach().clone().requires_grad_(True) for p in params]
    kwargs = dict(lr=1e-2, weight_decay=0.1, mars_type='mars-muon')
    opt, opt_twin = MARS(params, **kwargs), MARS(twin, **kwargs)
    for step in range(1, 6):
        grads = make_grads(model, seed=1000 + step)
        set_grads(params, grads)
        set_grads(twin, grads)
        opt.step()
        opt_twin.step()
    for p, q in zip(params, twin):
        assert torch.equal(p.detach(), q.detach())
        assert torch.equal(opt.state[p]['exp_avg'], opt_twin.state[q]['exp_avg'])


def test_mars_muon_zero_gradient_is_finite():
    """A zero momentum hits Newton-Schulz's 1e-7 pre-normalization, not a division by zero."""
    model = make_model()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    opt = MARS(params, lr=1e-2, weight_decay=0.0, mars_type='mars-muon')
    set_grads(params, [torch.zeros_like(p) for p in params])
    opt.step()
    for p, b in zip(params, before):
        assert torch.isfinite(p.detach()).all()
        torch.testing.assert_close(p.detach(), b, **TOL)


def test_mars_muon_state_dict_round_trip():
    """F088/F092: the step and every declared state key survive save/load, on a matrix, an expert
    stack and a vector alike."""
    g = torch.Generator().manual_seed(38)
    params = [
        torch.nn.Parameter(torch.randn(5, 4, generator=g)),
        expert_stack(),
        torch.nn.Parameter(torch.randn(5, generator=g)),
    ]
    make_opt = lambda ps: MARS(ps, lr=1e-2, weight_decay=0.1, mars_type='mars-muon')
    opt = make_opt(params)
    for step in range(1, 4):
        set_grads(params, grads_like(params, seed=step))
        opt.step()
    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    opt2 = make_opt(resumed)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, q in zip(params, resumed):
        assert set(opt.state[p]) == set(opt2.state[q]) == {'step', 'exp_avg', 'exp_avg_sq',
                                                            'last_grad'}
        assert opt.state[p]['step'] == opt2.state[q]['step'] == 3
        for key in ('exp_avg', 'exp_avg_sq', 'last_grad'):
            torch.testing.assert_close(opt.state[p][key], opt2.state[q][key])
    check_state_dict_round_trip(make_opt, params=params)


# ---- --mars-muon-rms-target: Muon's per-matrix update RMS, on the 2-D group only ----


@pytest.mark.parametrize('cls,kwargs', [(MARS, {}), (Mu2MARS, {}),
                                        (Mu2MARS, {'variant': 'anytime', 'anytime_gamma': 1.0})])
def test_muon_rms_target_sets_the_matrix_update_rms(cls, kwargs):
    target, lr = 0.2, 1e-2
    model = make_model()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    opt = cls(params, lr=lr, weight_decay=0.0, muon_rms_target=target, **kwargs)
    set_grads(params, make_grads(model, seed=41))
    opt.step()
    matrices = [p for p in params if p.ndim == 2]
    assert matrices
    for p, b in zip(params, before):
        step = b - p.detach()
        assert torch.isfinite(step).all()
        if p.ndim != 2:
            continue
        rms = step.double().norm() / math.sqrt(p.numel())
        torch.testing.assert_close(rms, torch.tensor(lr * target, dtype=torch.float64), **TOL)


@pytest.mark.parametrize('cls,kwargs', [(MARS, {}), (Mu2MARS, {}),
                                        (Mu2MARS, {'variant': 'anytime'})])
def test_muon_rms_target_leaves_the_1d_group_untouched(cls, kwargs):
    """One flag, one group: the 2-D params must move differently and the 1-D params identically."""
    model = make_model()
    params = list(model.parameters())
    plain = [p.detach().clone().requires_grad_(True) for p in params]
    common = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, **kwargs)
    opt = cls(params, muon_rms_target=0.2, **common)
    ref = cls(plain, **common)
    for step in range(1, 6):
        grads = make_grads(model, seed=700 + step)
        set_grads(params, grads)
        set_grads(plain, grads)
        opt.step()
        ref.step()
    for p, q in zip(params, plain):
        if p.ndim == 2:
            assert not torch.allclose(p.detach(), q.detach())
        else:
            torch.testing.assert_close(p.detach(), q.detach(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_muon_rms_target_is_per_expert(cls):
    target, lr = 0.2, 1e-2
    stacked = expert_stack(size_out=6, size_in=4)
    before = stacked.detach().clone()
    opt = cls([stacked], lr=lr, weight_decay=0.0, muon_rms_target=target)
    stacked.grad = grads_like([stacked], seed=800)[0]
    opt.step()
    step = (before - stacked.detach()).double()
    for i in range(stacked.shape[0]):
        rms = step[i].norm() / math.sqrt(step[i].numel())
        torch.testing.assert_close(rms, torch.tensor(lr * target, dtype=torch.float64), **TOL)


@pytest.mark.parametrize('mars_type', ['mars-shampoo', 'mars-muon'])
def test_muon_rms_target_is_rejected_with_an_orthogonalizing_inner_step(mars_type):
    """mars-shampoo and mars-muon fix their own update scale; two scalings would silently compose."""
    g = torch.Generator().manual_seed(34)
    with pytest.raises(ValueError, match=mars_type):
        MARS([torch.nn.Parameter(torch.randn(4, 3, generator=g))], lr=1e-2,
             mars_type=mars_type, muon_rms_target=0.2)


def expert_stack(num_experts=3, size_out=5, size_in=4, seed=11):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(num_experts, size_out, size_in, generator=g))


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_expert_stack_matches_independent_experts(cls):
    """A grouped-expert weight [local_experts, out, in] must move exactly as the same experts
    held as separate 2D matrices. clip is small enough that it binds on every expert, so a
    joint ||c_t|| over the whole stack (the pre-fix --mars-optimize-1d reading) fails here."""
    stacked = expert_stack()
    num_experts = stacked.shape[0]
    singles = [torch.nn.Parameter(stacked.detach()[i].clone()) for i in range(num_experts)]
    kwargs = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, clip=0.5)
    opt = cls([stacked], **kwargs)
    refs = [cls([s], **kwargs) for s in singles]
    for step in range(1, 11):
        grad = grads_like([stacked], seed=400 + step)[0]
        stacked.grad = grad.clone()
        opt.step()
        for i, (single, ref) in enumerate(zip(singles, refs)):
            single.grad = grad[i].clone()
            ref.step()
    for i, single in enumerate(singles):
        torch.testing.assert_close(stacked.detach()[i], single.detach(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_expert_stack_takes_the_matrix_path(cls):
    """The stack must not fall through to the AdamW-on-betas_1d branch (F006)."""
    stacked = expert_stack()
    assert is_matrix_param(stacked)
    lr, wd, eps, betas_1d = 1e-2, 0.1, 1e-8, (0.9, 0.95)
    opt = cls([stacked], lr=lr, weight_decay=wd, eps=eps, betas_1d=betas_1d)
    adamw_param = stacked.detach().clone().requires_grad_(True)
    adamw = torch.optim.AdamW([adamw_param], lr=lr, betas=betas_1d, eps=eps, weight_decay=wd)
    for step in range(1, 6):
        grad = grads_like([stacked], seed=500 + step)[0]
        stacked.grad = grad.clone()
        adamw_param.grad = grad.clone()
        opt.step()
        adamw.step()
    assert not torch.allclose(stacked.detach(), adamw_param.detach())


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_router_weight_is_a_matrix(cls):
    """muon.py keeps router.weight (2D) in linear_params, so MARS keeps it on the matrix rule:
    an is_router tag must change nothing."""
    g = torch.Generator().manual_seed(13)
    router = torch.nn.Parameter(torch.randn(8, 6, generator=g))
    router.is_router = True
    plain = torch.nn.Parameter(router.detach().clone())
    assert is_matrix_param(router)
    opt = cls([router], lr=1e-2, weight_decay=0.1)
    ref = cls([plain], lr=1e-2, weight_decay=0.1)
    for step in range(1, 6):
        grad = grads_like([router], seed=600 + step)[0]
        router.grad = grad.clone()
        plain.grad = grad.clone()
        opt.step()
        ref.step()
    torch.testing.assert_close(router.detach(), plain.detach(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_expert_stack_state_dict_round_trip(cls):
    g = torch.Generator().manual_seed(17)
    check_state_dict_round_trip(
        lambda ps: cls(ps, lr=1e-2, weight_decay=0.1),
        params=[expert_stack(), torch.nn.Parameter(torch.randn(5, generator=g))],
    )


def test_mu2mars_anytime_expert_stack_state_dict_round_trip():
    check_state_dict_round_trip(
        lambda ps: Mu2MARS(ps, lr=1e-2, weight_decay=0.1, variant='anytime', anytime_gamma=0.1),
        params=[expert_stack()],
    )


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_param_with_more_than_three_dims_is_rejected(cls):
    """Anything the ndim routing does not know about must fail loudly, not pick a branch."""
    g = torch.Generator().manual_seed(19)
    p = torch.nn.Parameter(torch.randn(2, 3, 4, 5, generator=g))
    p.grad = torch.randn(p.shape, generator=g)
    with pytest.raises(AssertionError, match='param.ndim'):
        cls([p], lr=1e-2).step()


# ---- AdEMAMix checkpoint state: F088 (dist-opt placeholder keys) and F031 (step) ----


def make_ademamix(params, beta1=0.9):
    """One param group carrying the keys DistributedOptimizer matches groups on."""
    group = dict(
        params=params, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
    )
    return AdEMAMix([group], lr=1e-2, betas=(beta1, 0.999, 0.9999), weight_decay=0.1)


def fake_distributed_optimizer(opt, params):
    """A DistributedOptimizer carrying only what the state_dict/load_state_dict pair reads, so
    the "load before the first step" path can run on CPU, without CUDA or torch.distributed."""
    dist_opt = DistributedOptimizer.__new__(DistributedOptimizer)
    dist_opt.optimizer = opt
    dist_opt.grad_scaler = None
    dist_opt.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    dist_opt.config = SimpleNamespace(
        fp16=False,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
    )
    # One grad buffer holding every param whole (a single DP rank).
    dist_opt.gbuf_ranges = [
        {torch.float32: [{'param_map': {p: {'gbuf_world': range(p.numel())} for p in params}}]}
    ]
    dist_opt.model_param_group_index_map = {p: (0, i) for i, p in enumerate(params)}
    return dist_opt


def sharded_state_keys(opt, param):
    """The state a sharded param-state format writes per param: tensors, minus the 0-dim step."""
    return {k for k, v in opt.state[param].items() if torch.is_tensor(v) and v.dim() > 0}


def validate_global_keys(requested, in_checkpoint):
    """Mirrors dist_checkpointing.strategies.torch._validate_global_shapes: every sharded tensor
    the loading run asks for must be present in the checkpoint."""
    for key in sorted(requested):
        if key not in in_checkpoint:
            raise KeyError(f"{key} from model not in state dict: {sorted(in_checkpoint)}")


@pytest.mark.parametrize('beta1', [0.9, 0.0])
def test_ademamix_state_keys_match_the_allocated_state(beta1):
    params = list(make_model().parameters())
    opt = make_ademamix(params, beta1=beta1)
    set_grads(params, grads_like(params, seed=1))
    opt.step()
    keys = opt.state_keys(opt.param_groups[0])
    assert ('exp_avg_fast' in keys) == (beta1 != 0.0)
    for p in params:
        assert set(opt.state[p]) == set(keys)
        assert opt.state[p]['step'].shape == ()


def test_ademamix_beta1_zero_state_dict_round_trip():
    check_state_dict_round_trip(lambda ps: make_ademamix(ps, beta1=0.0))


@pytest.mark.parametrize('beta1', [0.9, 0.0])
def test_distributed_optimizer_load_allocates_ademamix_state(beta1):
    """F088: on `--load` the inner optimizer state is still empty, so DistributedOptimizer
    allocates a placeholder state whose keys name the checkpoint's sharded tensors. Those keys
    used to be Adam's, so every ademamix resume died in _validate_global_shapes."""
    params = list(make_model().parameters())
    trained = make_ademamix(params, beta1=beta1)
    for step in range(1, 4):
        set_grads(params, grads_like(params, seed=step))
        trained.step()
    checkpoint = fake_distributed_optimizer(trained, params).state_dict()
    assert [g['step'] for g in checkpoint['optimizer']['param_groups']] == [3]
    in_checkpoint = sharded_state_keys(trained, params[0])

    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    fresh = make_ademamix(resumed, beta1=beta1)
    dist_opt = fake_distributed_optimizer(fresh, resumed)
    assert len(fresh.state) == 0

    # sharded_state_dict(is_loading=True) preallocates the state that names the sharded tensors.
    dist_opt.load_state_dict(dist_opt.state_dict())
    expected = {'exp_avg_slow', 'exp_avg_sq'} | ({'exp_avg_fast'} if beta1 != 0.0 else set())
    assert in_checkpoint == expected
    for p in resumed:
        requested = sharded_state_keys(fresh, p)
        validate_global_keys(requested, in_checkpoint)
        assert requested == expected
    # The sharded param state holds these very tensors, so their identity must survive the load.
    steps = {id(fresh.state[p]['step']) for p in resumed}
    assert len(steps) == len(resumed)

    dist_opt.load_state_dict(copy.deepcopy(checkpoint))
    for p in resumed:
        assert sharded_state_keys(fresh, p) == expected
        assert fresh.state[p]['step'] == 3  # F031: the step survives the round trip
    assert {id(fresh.state[p]['step']) for p in resumed} == steps


def test_distributed_optimizer_load_keeps_adam_keys():
    """The default path must stay exactly Adam's two keys."""
    params = list(make_model().parameters())
    group = dict(
        params=params, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
    )
    trained = torch.optim.AdamW([group], lr=1e-2)
    set_grads(params, grads_like(params, seed=1))
    trained.step()
    checkpoint = fake_distributed_optimizer(trained, params).state_dict()

    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    fresh = torch.optim.AdamW([{**group, 'params': resumed}], lr=1e-2)
    dist_opt = fake_distributed_optimizer(fresh, resumed)
    assert dist_opt._inner_optimizer_state_keys(0) == ('exp_avg', 'exp_avg_sq')
    assert not dist_opt._inner_optimizer_keeps_step()
    dist_opt.load_state_dict(copy.deepcopy(checkpoint))
    for p in resumed:
        assert sharded_state_keys(fresh, p) == {'exp_avg', 'exp_avg_sq'}


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_distributed_optimizer_state_keys_default_for_other_optimizers(cls):
    params = list(make_model().parameters())
    dist_opt = fake_distributed_optimizer(cls(params, lr=1e-2), params)
    assert dist_opt._inner_optimizer_state_keys(0) == ('exp_avg', 'exp_avg_sq')
    assert not dist_opt._inner_optimizer_keeps_step()


# ---- AdEMAMix step seeding for pre-fix checkpoints: F092 ----


WAVE_A_WARMUP = dict(alpha=8.0, alpha_warmup=5000, beta3_warmup=5000)


def shard_shaped_params(seed=7):
    """Flat params, so that the state DistributedOptimizer allocates per shard (1-D, numel of the
    grad buffer range) has the shape the inner optimizer steps on, as it does on a real rank."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g).requires_grad_(True) for n in (12, 7)]


def ademamix_reference_update(p, grad, m_fast, m_slow, v, step, lr, betas, alpha_final,
                              alpha_warmup, wd, eps=1e-8):
    """One AdEMAMix update written from the published algorithm, element by element, for a step
    past the beta3 warmup (beta3 is then its final value)."""
    beta1, beta2, beta3 = betas
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    alpha = alpha_final * min(step / float(alpha_warmup), 1.0)
    out = []
    for i in range(len(p)):
        mf = beta1 * m_fast[i] + (1.0 - beta1) * grad[i]
        ms = beta3 * m_slow[i] + (1.0 - beta3) * grad[i]
        vv = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i]
        denom = math.sqrt(vv) / math.sqrt(bc2) + eps
        out.append(p[i] - lr * ((mf / bc1 + alpha * ms) / denom + wd * p[i]))
    return out


def ademamix_checkpoint(params, steps=3, **warmup):
    """A trained AdEMAMix and the checkpoint a DistributedOptimizer writes for it."""
    group = dict(
        params=params, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
    )
    trained = AdEMAMix([group], lr=1e-2, betas=(0.9, 0.999, 0.9999), weight_decay=0.1, **warmup)
    for step in range(1, steps + 1):
        set_grads(params, grads_like(params, seed=step))
        trained.step()
    return trained, fake_distributed_optimizer(trained, params).state_dict()


def resume_from(checkpoint, trained, params, **warmup):
    """Resume `checkpoint` the way `--load` does: allocate the placeholder state (that is what
    names the sharded tensors), let the sharded load write the checkpoint's tensors into it, then
    load the checkpoint's own (non-sharded) half."""
    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    group = dict(
        params=resumed, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
    )
    fresh = AdEMAMix([group], lr=1e-2, betas=(0.9, 0.999, 0.9999), weight_decay=0.1, **warmup)
    dist_opt = fake_distributed_optimizer(fresh, resumed)
    dist_opt.load_state_dict(dist_opt.state_dict())
    for src, dst in zip(params, resumed):
        # The 0-dim step is skipped by every sharded param-state format; only the shards land.
        for key, tensor in trained.state[src].items():
            if key != 'step':
                fresh.state[dst][key].copy_(tensor)
    dist_opt.load_state_dict(copy.deepcopy(checkpoint))
    return fresh, dist_opt, resumed


def test_ademamix_missing_step_is_seeded_from_the_training_iteration():
    """F092: a checkpoint written before the step was published to param_groups carries none, so
    the resumed run restarted the alpha warmup and the bias correction from step 0. Seeded from
    the iteration, the very next update is the one step 26001 prescribes, with alpha at 8."""
    params = shard_shaped_params()
    trained, checkpoint = ademamix_checkpoint(params, **WAVE_A_WARMUP)
    assert [g['step'] for g in checkpoint['optimizer']['param_groups']] == [3]
    for g in checkpoint['optimizer']['param_groups']:  # a pre-e0c0921fc checkpoint
        del g['step']

    fresh, dist_opt, resumed = resume_from(checkpoint, trained, params, **WAVE_A_WARMUP)
    assert dist_opt._step_missing_from_checkpoint
    assert all(fresh.state[p]['step'] == 0 for p in resumed)

    dist_opt.set_missing_step(26000)
    assert all(fresh.state[p]['step'] == 26000 for p in resumed)
    assert not dist_opt._step_missing_from_checkpoint

    p = resumed[0]
    before = dict(
        p=p.detach().double().flatten().tolist(),
        m_fast=fresh.state[p]['exp_avg_fast'].double().flatten().tolist(),
        m_slow=fresh.state[p]['exp_avg_slow'].double().flatten().tolist(),
        v=fresh.state[p]['exp_avg_sq'].double().flatten().tolist(),
    )
    grads = grads_like(resumed, seed=4)
    set_grads(resumed, grads)
    fresh.step()

    assert fresh.state[p]['step'] == 26001
    grad = grads[0].double().flatten().tolist()
    common = dict(
        lr=1e-2, betas=(0.9, 0.999, 0.9999), alpha_final=8.0, alpha_warmup=5000, wd=0.1
    )
    warmed = ademamix_reference_update(**before, grad=grad, step=26001, **common)
    restarted = ademamix_reference_update(**before, grad=grad, step=1, **common)
    got = p.detach().double().flatten().tolist()
    assert torch.allclose(torch.tensor(got), torch.tensor(warmed), **TOL)
    assert not torch.allclose(torch.tensor(got), torch.tensor(restarted), **TOL)


def test_ademamix_step_from_the_checkpoint_is_not_overridden():
    """A checkpoint that carries its own step keeps it: seeding is only for the pre-fix ones."""
    params = shard_shaped_params()
    trained, checkpoint = ademamix_checkpoint(params, **WAVE_A_WARMUP)
    fresh, dist_opt, resumed = resume_from(checkpoint, trained, params, **WAVE_A_WARMUP)
    assert not dist_opt._step_missing_from_checkpoint
    assert all(fresh.state[p]['step'] == 3 for p in resumed)

    dist_opt.set_missing_step(26000)
    assert all(fresh.state[p]['step'] == 3 for p in resumed)

    set_grads(resumed, grads_like(resumed, seed=4))
    fresh.step()
    assert all(fresh.state[p]['step'] == 4 for p in resumed)


def test_set_missing_step_is_a_no_op_for_optimizers_without_a_step():
    params = shard_shaped_params()
    group = dict(
        params=params, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
    )
    trained = torch.optim.AdamW([group], lr=1e-2)
    set_grads(params, grads_like(params, seed=1))
    trained.step()
    checkpoint = fake_distributed_optimizer(trained, params).state_dict()

    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    fresh = torch.optim.AdamW([{**group, 'params': resumed}], lr=1e-2)
    dist_opt = fake_distributed_optimizer(fresh, resumed)
    dist_opt.load_state_dict(copy.deepcopy(checkpoint))
    steps = [fresh.state[p]['step'].clone() for p in resumed]
    dist_opt.set_missing_step(26000)
    assert [fresh.state[p]['step'] for p in resumed] == steps


# ---- --mars-normalize-update-to-weight-norm: MuonMD's measured-norm match, on the 2-D group ----


WEIGHT_NORM_CASES = [(MARS, {}), (Mu2MARS, {}),
                     (Mu2MARS, {'variant': 'anytime', 'anytime_gamma': 1.0})]
LOOSE = dict(atol=1e-5, rtol=1e-4)


def frobenius(t):
    return t.detach().double().norm()


@pytest.mark.parametrize('cls,kwargs', WEIGHT_NORM_CASES)
def test_weight_norm_off_leaves_the_state_layout_alone(cls, kwargs):
    """Default off: no norm keys, so every checkpoint and update path measured so far is as is."""
    model = make_model()
    params = list(model.parameters())
    opt = cls(params, lr=1e-2, **kwargs)
    set_grads(params, make_grads(model, seed=1))
    opt.step()
    assert not opt.normalize_update_to_weight_norm
    for p in params:
        assert not set(opt.state[p]) & set(NORM_STATE_KEYS)


@pytest.mark.parametrize('cls,kwargs', WEIGHT_NORM_CASES)
def test_weight_norm_matches_every_matrix_step_to_its_recorded_norm(cls, kwargs):
    """With wd = 0 every 2-D step has ||dW||_F = lr * ||W_0||_F, at step 1 and still at step 5
    once the weights have moved: the target is the norm recorded before the first update."""
    model = make_model()
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    lr = 0.1
    opt = cls(params, lr=lr, weight_decay=0.0, normalize_update_to_weight_norm=True, **kwargs)
    for step in range(1, 6):
        before = [p.detach().clone() for p in params]
        set_grads(params, make_grads(model, seed=900 + step))
        opt.step()
        for p, b, w0 in zip(params, before, initial):
            delta = b - p.detach()
            assert torch.isfinite(delta).all()
            if p.ndim != 2:
                continue
            torch.testing.assert_close(frobenius(delta), lr * frobenius(w0), **TOL)
            torch.testing.assert_close(opt.state[p]['weight_norm'].double(), frobenius(w0), **TOL)
    for p, w0 in zip(params, initial):
        if p.ndim == 2:
            assert abs(frobenius(p) - frobenius(w0)) > 1e-2 * frobenius(w0)


@pytest.mark.parametrize('cls,kwargs', WEIGHT_NORM_CASES)
def test_weight_norm_leaves_the_1d_group_untouched(cls, kwargs):
    model = make_model()
    params = list(model.parameters())
    plain = [p.detach().clone().requires_grad_(True) for p in params]
    common = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, **kwargs)
    opt = cls(params, normalize_update_to_weight_norm=True, **common)
    ref = cls(plain, **common)
    for step in range(1, 6):
        grads = make_grads(model, seed=1000 + step)
        set_grads(params, grads)
        set_grads(plain, grads)
        opt.step()
        ref.step()
    for p, q in zip(params, plain):
        if p.ndim == 2:
            assert not torch.allclose(p.detach(), q.detach())
        else:
            torch.testing.assert_close(p.detach(), q.detach(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_weight_norm_scales_only_the_update_direction(cls):
    """The step is lr * (wd * W + U * ||W_0|| / ||U||) with U the plain optimizer's direction:
    the decoupled decay term is not rescaled, and ||U|| is what update_norm records."""
    model = make_model()
    params = list(model.parameters())
    plain = [p.detach().clone().requires_grad_(True) for p in params]
    lr, wd = 1e-2, 0.1
    opt = cls(params, lr=lr, weight_decay=wd, normalize_update_to_weight_norm=True)
    ref = cls(plain, lr=lr, weight_decay=wd)
    grads = make_grads(model, seed=1100)
    set_grads(params, grads)
    set_grads(plain, grads)
    before = [p.detach().clone() for p in params]
    opt.step()
    ref.step()
    for p, q, b in zip(params, plain, before):
        if p.ndim != 2:
            continue
        direction = (b - q.detach()) / lr - wd * b
        scale = frobenius(b) / frobenius(direction)
        expected = b - lr * (wd * b + direction * scale.float())
        torch.testing.assert_close(p.detach(), expected, **LOOSE)
        torch.testing.assert_close(opt.state[p]['update_norm'].double(), frobenius(direction),
                                   **LOOSE)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_weight_norm_is_per_expert(cls):
    stacked = expert_stack(size_out=6, size_in=4)
    initial = stacked.detach().clone()
    lr = 1e-2
    opt = cls([stacked], lr=lr, weight_decay=0.0, normalize_update_to_weight_norm=True)
    stacked.grad = grads_like([stacked], seed=1200)[0]
    opt.step()
    assert opt.state[stacked]['weight_norm'].shape == (stacked.shape[0],)
    assert opt.state[stacked]['update_norm'].shape == (stacked.shape[0],)
    delta = (initial - stacked.detach()).double()
    for i in range(stacked.shape[0]):
        torch.testing.assert_close(delta[i].norm(), lr * initial[i].double().norm(), **TOL)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_weight_norm_expert_stack_matches_independent_experts(cls):
    stacked = expert_stack()
    num_experts = stacked.shape[0]
    singles = [torch.nn.Parameter(stacked.detach()[i].clone()) for i in range(num_experts)]
    kwargs = dict(lr=1e-2, weight_decay=0.1, eps=1e-8, clip=0.5,
                  normalize_update_to_weight_norm=True)
    opt = cls([stacked], **kwargs)
    refs = [cls([s], **kwargs) for s in singles]
    for step in range(1, 6):
        grad = grads_like([stacked], seed=1300 + step)[0]
        stacked.grad = grad.clone()
        opt.step()
        for i, (single, ref) in enumerate(zip(singles, refs)):
            single.grad = grad[i].clone()
            ref.step()
    for i, single in enumerate(singles):
        torch.testing.assert_close(stacked.detach()[i], single.detach(), **TOL)


@pytest.mark.parametrize('cls,kwargs', WEIGHT_NORM_CASES)
def test_weight_norm_state_dict_round_trip(cls, kwargs):
    check_state_dict_round_trip(
        lambda ps: cls(ps, lr=1e-2, weight_decay=0.1, normalize_update_to_weight_norm=True,
                       **kwargs)
    )
    g = torch.Generator().manual_seed(23)
    check_state_dict_round_trip(
        lambda ps: cls(ps, lr=1e-2, weight_decay=0.1, normalize_update_to_weight_norm=True,
                       **kwargs),
        params=[expert_stack(), torch.nn.Parameter(torch.randn(5, generator=g))],
    )


def test_weight_norm_resume_keeps_the_recorded_norm_not_the_current_one():
    """A resumed run scales to the norm its parent recorded at ITS first step. Re-measuring at
    load would silently change every later update (the F092 failure mode)."""
    model = make_model()
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    lr = 0.1
    opt = MARS(params, lr=lr, weight_decay=0.0, normalize_update_to_weight_norm=True)
    for step in range(1, 4):
        set_grads(params, grads_like(params, seed=step))
        opt.step()

    resumed = [p.detach().clone().requires_grad_(True) for p in params]
    opt2 = MARS(resumed, lr=lr, weight_decay=0.0, normalize_update_to_weight_norm=True)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for q, w0 in zip(resumed, initial):
        if q.ndim != 2:
            continue
        torch.testing.assert_close(opt2.state[q]['weight_norm'].double(), frobenius(w0), **TOL)
        assert abs(frobenius(q) - frobenius(w0)) > 1e-2 * frobenius(w0)

    before = [q.detach().clone() for q in resumed]
    grads = grads_like(params, seed=99)
    set_grads(params, grads)
    set_grads(resumed, grads)
    opt.step()
    opt2.step()
    for p, q, b, w0 in zip(params, resumed, before, initial):
        torch.testing.assert_close(p.detach(), q.detach())
        if q.ndim == 2:
            torch.testing.assert_close(frobenius(b - q.detach()), lr * frobenius(w0), **TOL)


def test_weight_norm_placeholder_state_is_measured_at_step_zero():
    """Megatron's --load path preallocates the state (init_state_fn) before the checkpoint fills
    it. A fresh start that went through the same allocation holds zeros at step 0; those must be
    measured at the first step, never used as a target (which would zero every matrix update)."""
    model = make_model()
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    lr = 1e-2
    opt = MARS(params, lr=lr, weight_decay=0.0, normalize_update_to_weight_norm=True)
    for p in params:
        opt.state[p]['step'] = 0
        for key in ('exp_avg', 'exp_avg_sq', 'last_grad'):
            opt.state[p][key] = torch.zeros_like(p.data)
        if is_matrix_param(p):
            for key in NORM_STATE_KEYS:
                opt.state[p][key] = norm_state_like(p)
    set_grads(params, make_grads(model, seed=1400))
    opt.step()
    for p, w0 in zip(params, initial):
        if p.ndim != 2:
            continue
        torch.testing.assert_close(opt.state[p]['weight_norm'].double(), frobenius(w0), **TOL)
        torch.testing.assert_close(frobenius(w0 - p.detach()), lr * frobenius(w0), **TOL)


def test_norm_state_like_has_one_scalar_per_matrix():
    assert norm_state_like(expert_stack()).shape == (3,)
    assert norm_state_like(torch.nn.Parameter(torch.zeros(4, 3))).shape == ()
    assert norm_state_like(torch.nn.Parameter(torch.zeros(4, 3))).dtype == torch.float32


def test_weight_norm_zero_update_stays_zero_and_finite():
    """A zero gradient at the first step gives U = 0; the match maps it to zero, not to NaN."""
    model = make_model()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    opt = MARS(params, lr=1e-2, weight_decay=0.0, normalize_update_to_weight_norm=True)
    set_grads(params, [torch.zeros_like(p) for p in params])
    opt.step()
    for p, b in zip(params, before):
        assert torch.isfinite(p.detach()).all()
        torch.testing.assert_close(p.detach(), b)
        if p.ndim == 2:
            assert opt.state[p]['update_norm'].item() == 0.0


def test_weight_norm_is_deterministic():
    runs = []
    for _ in range(2):
        model = make_model()
        params = list(model.parameters())
        opt = MARS(params, lr=1e-2, weight_decay=0.1, normalize_update_to_weight_norm=True)
        for step in range(1, 6):
            set_grads(params, make_grads(model, seed=1500 + step))
            opt.step()
        runs.append(([p.detach().clone() for p in params],
                     [opt.state[p]['update_norm'].clone() for p in params if p.ndim == 2]))
    for a, b in zip(runs[0][0], runs[1][0]):
        assert torch.equal(a, b)
    for a, b in zip(runs[0][1], runs[1][1]):
        assert torch.equal(a, b)


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_weight_norm_rejects_a_fixed_target_alongside(cls):
    g = torch.Generator().manual_seed(35)
    with pytest.raises(ValueError, match='two targets'):
        cls([torch.nn.Parameter(torch.randn(4, 3, generator=g))], lr=1e-2, muon_rms_target=0.2,
            normalize_update_to_weight_norm=True)


@pytest.mark.parametrize('mars_type', ['mars-adamw', 'mars-lion', 'mars-shampoo', 'mars-muon'])
def test_weight_norm_supersedes_the_inner_optimizer_scale(mars_type):
    """As in MuonMD, the measured-norm match removes every scalar factor the inner step carries
    (Lion's unit entries, shampoo's and MARS-M's shape factors): all land on lr * ||W_0||_F."""
    g = torch.Generator().manual_seed(36)
    matrix = torch.nn.Parameter(torch.randn(8, 4, generator=g))
    w0 = matrix.detach().clone()
    lr = 1e-2
    opt = MARS([matrix], lr=lr, weight_decay=0.0, mars_type=mars_type,
               normalize_update_to_weight_norm=True)
    matrix.grad = grads_like([matrix], seed=37)[0]
    opt.step()
    torch.testing.assert_close(frobenius(w0 - matrix.detach()), lr * frobenius(w0), **TOL)


def test_collect_norm_stats_reports_per_shape_means():
    """Equal-weighted per-matrix means, globally and per shape, from the recorded state; nothing
    when the flag is off; found through Megatron's chained/mixed-precision wrappers."""
    model = make_model()
    params = list(model.parameters())
    opt = MARS(params, lr=1e-2, weight_decay=0.1, normalize_update_to_weight_norm=True)
    assert collect_norm_stats(opt) == {}
    set_grads(params, make_grads(model, seed=1600))
    opt.step()
    stats = collect_norm_stats(opt)
    matrices = [p for p in params if p.ndim == 2]
    shapes = [f'{p.shape[0]}x{p.shape[1]}' for p in matrices]
    assert shapes == ['5x6', '4x5']
    assert set(stats) == {f'mars/{name}{suffix}' for name in ('weight-rms', 'update-rms',
                                                             'update-scale')
                          for suffix in ['', *(f'/{s}' for s in shapes)]}
    per_matrix = {}
    for p, shape in zip(matrices, shapes):
        w, u = opt.state[p]['weight_norm'].item(), opt.state[p]['update_norm'].item()
        per_matrix[shape] = (w / math.sqrt(p.numel()), u / math.sqrt(p.numel()), w / u)
        assert stats[f'mars/weight-rms/{shape}'] == pytest.approx(per_matrix[shape][0])
        assert stats[f'mars/update-rms/{shape}'] == pytest.approx(per_matrix[shape][1])
        assert stats[f'mars/update-scale/{shape}'] == pytest.approx(per_matrix[shape][2])
    for i, name in enumerate(('weight-rms', 'update-rms', 'update-scale')):
        assert stats[f'mars/{name}'] == pytest.approx(
            sum(v[i] for v in per_matrix.values()) / len(per_matrix))
    assert all(math.isfinite(v) for v in stats.values())

    wrapped = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=opt)])
    assert collect_norm_stats(wrapped) == stats
    plain = MARS([p.detach().clone().requires_grad_(True) for p in params], lr=1e-2)
    assert collect_norm_stats(plain) == {}


@pytest.mark.parametrize('cls', [MARS, Mu2MARS])
def test_sharded_norm_state_checkpoints_the_norms_as_objects(cls):
    """Under --ckpt-format torch_dist every state key goes through optim_state_to_sharding_state,
    which asserts a param-shaped tensor unless the optimizer's hook claims the key: the recorded
    norms become ShardedObjects under the param's key and replica id, everything else falls
    through to the param-shaped path."""
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    model_param = SimpleNamespace(key='decoder.layers.0.mlp.linear_fc1.weight', replica_id=(0, 0, 3))
    value = torch.tensor(2.5)
    for key in NORM_STATE_KEYS:
        obj = cls.build_sharded_optimizer_state(model_param, value, key, f'optimizer.state.{key}')
        assert isinstance(obj, ShardedObject)
        assert obj.key == f'optimizer.state.{key}.{model_param.key}'
        assert obj.data is value
        assert obj.replica_id == (0, 0, 3)
        assert (obj.global_shape, obj.global_offset) == ((1,), (0,))
    for key in ('exp_avg', 'exp_avg_sq', 'last_grad', 'mu_avg', 'w'):
        assert cls.build_sharded_optimizer_state(model_param, value, key, 'p') is None


def test_weight_norm_state_goes_through_optim_state_to_sharding_state():
    """The real torch_dist conversion: param-shaped keys become ShardedTensors that follow the
    model param, the norms become ShardedObjects, and nothing trips the shape assertion."""
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.optimizer import optim_state_to_sharding_state

    params = [expert_stack(), torch.nn.Parameter(torch.randn(4, 3)),
              torch.nn.Parameter(torch.randn(3))]
    opt = MARS(params, lr=1e-2, weight_decay=0.1, normalize_update_to_weight_norm=True)
    set_grads(params, grads_like(params, seed=1700))
    opt.step()
    state_dict = opt.state_dict()
    id_map = {
        i: ShardedTensor.from_rank_offsets(f'param{i}', p.detach(), replica_id=(0, 0, 1))
        for i, p in enumerate(params)
    }
    optim_state_to_sharding_state(
        state_dict, id_map, exclude_keys='step', state_sharding_fn=MARS.build_sharded_optimizer_state
    )
    for i, p in enumerate(params):
        sharded = state_dict['state'][i]
        expected = {'exp_avg', 'exp_avg_sq', 'last_grad'} | (
            set(NORM_STATE_KEYS) if is_matrix_param(p) else set())
        assert set(sharded) == expected
        for key in set(NORM_STATE_KEYS) & set(sharded):
            assert isinstance(sharded[key], ShardedObject)
            assert sharded[key].key == f'optimizer.state.{key}.param{i}'
            assert sharded[key].replica_id == (0, 0, 1)
            assert sharded[key].data.shape == p.shape[:-2]
        for key in expected - set(NORM_STATE_KEYS):
            assert isinstance(sharded[key], ShardedTensor)
            assert sharded[key].local_shape == tuple(p.shape)
