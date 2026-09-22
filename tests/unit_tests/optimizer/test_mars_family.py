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
from megatron.core.optimizer.mars import MARS, is_matrix_param
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
