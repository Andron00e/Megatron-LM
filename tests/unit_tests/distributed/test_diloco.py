# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""CPU/gloo tests for megatron/core/distributed/diloco.py.

Megatron's own DDP allocates its grad buffers on `torch.cuda.current_device()`, so the multi-process
tests here emulate the only thing DDP contributes to the algorithm -- a gradient all-reduce over
`pg.dp`, which DiLoCo retargets to the worker sub-group -- with an explicit all-reduce over the
group `build_worker_process_groups` hands back. `test_pg_collection_routes_dp_to_worker_group`
covers the claim that setting `pg.dp` alone is enough to retarget every dense grad-buffer
collective, without needing a GPU.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.distributed.diloco import (
    DiLoCoConfig,
    DiLoCoOuterOptimizer,
    build_worker_process_groups,
)

H = 3
OUTER_ROUNDS = 5


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def run_outer_rounds(config, seed=0):
    """Drive the outer optimizer with a prescribed sequence of post-inner-loop parameters.

    The inner loop is replaced by writing `targets[t]` into the params, so Delta_t = x_t - target_t
    is a deterministic function of whatever the outer rule produced for x_t -- enough to pin the
    rule against a closed-form reference without any optimizer or data.
    """
    model = tiny_model(seed)
    params = list(model.parameters())
    inner = torch.optim.SGD(params, lr=0.0)
    outer = DiLoCoOuterOptimizer(inner, config)
    outer.allocate_state()

    torch.manual_seed(seed + 1000)
    targets = [[torch.randn_like(p) * 0.1 for p in params] for _ in range(OUTER_ROUNDS)]

    trace = []
    for t, target in enumerate(targets, start=1):
        with torch.no_grad():
            for p, value in zip(params, target):
                p.copy_(value)
        outer.outer_step(t * config.inner_steps)
        trace.append(flat(model).clone())
    return trace, [torch.cat([v.reshape(-1) for v in target]) for target in targets]


def reference(config, x0, targets):
    """Closed-form transcription of the rules documented in diloco.py, on one flat vector."""
    x = x0.clone()
    b = torch.zeros_like(x)
    dprev = torch.zeros_like(x)
    v = torch.zeros_like(x)
    m = torch.zeros_like(x)
    w = x0.clone()
    trace = []
    for t, target in enumerate(targets, start=1):
        delta = x - target
        if config.outer in ('nesterov', 'snoo'):
            b = config.outer_momentum * b + delta
            update = config.outer_momentum * b + delta if config.outer_nesterov else b
            x = x - config.outer_lr * update
        else:
            beta = config.mu2_beta
            c = delta + config.mu2_gamma * beta / (1.0 - beta) * (delta - dprev)
            dprev = delta
            norm = c.norm().item()
            if config.mu2_clip > 0.0 and norm > config.mu2_clip:
                c = c * (config.mu2_clip / norm)
            b = beta * b + (1.0 - beta) * c
            d_hat = b / (1.0 - beta**t)
            if config.outer_precond == 'adam':
                v = config.outer_beta2 * v + (1.0 - config.outer_beta2) * c * c
                denom = (v / (1.0 - config.outer_beta2**t)).sqrt() + config.outer_eps
            else:
                denom = torch.ones_like(x)
            if config.mu2_variant == 'anytime':
                w = w - config.outer_lr * d_hat / denom
                x = config.mu2_anytime_gamma * w + (1.0 - config.mu2_anytime_gamma) * x
            else:
                m = config.mu2_beta3 * m + (1.0 - config.mu2_beta3) * d_hat
                x = x - config.outer_lr * (m / (1.0 - config.mu2_beta3**t)) / denom
        trace.append(x.clone())
    return trace


@pytest.mark.parametrize(
    "config",
    [
        DiLoCoConfig(inner_steps=H, outer='nesterov', outer_lr=0.7, outer_momentum=0.9),
        DiLoCoConfig(inner_steps=H, outer='nesterov', outer_nesterov=False),
        DiLoCoConfig(inner_steps=H, outer='snoo', outer_lr=0.5, outer_momentum=0.8),
        DiLoCoConfig(inner_steps=H, outer='mu2', outer_lr=0.7, mu2_variant='anytime'),
        DiLoCoConfig(
            inner_steps=H, outer='mu2', mu2_variant='anytime', mu2_clip=0.0, mu2_gamma=0.5
        ),
        DiLoCoConfig(inner_steps=H, outer='mu2', mu2_variant='ema', mu2_beta3=0.9),
        DiLoCoConfig(inner_steps=H, outer='mu2', mu2_variant='anytime', outer_precond='adam'),
        DiLoCoConfig(inner_steps=H, outer='mu2', mu2_variant='ema', outer_precond='adam'),
    ],
)
def test_outer_rules_match_reference(config):
    x0 = flat(tiny_model(0)).clone()
    trace, targets = run_outer_rounds(config)
    expected = reference(config, x0, targets)
    for t, (got, want) in enumerate(zip(trace, expected), start=1):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6, msg=f'outer round {t}')


def test_mu2_reduces_to_diloco_without_variance_reduction():
    common = dict(inner_steps=H, outer_lr=0.7, mu2_clip=0.0)
    mu2 = DiLoCoConfig(
        outer='mu2', mu2_gamma=0.0, mu2_beta=0.0, mu2_anytime_gamma=1.0, **common
    )
    sgd = DiLoCoConfig(outer='nesterov', outer_momentum=0.0, outer_nesterov=False, **common)
    mu2_trace, _ = run_outer_rounds(mu2)
    sgd_trace, _ = run_outer_rounds(sgd)
    for got, want in zip(mu2_trace, sgd_trace):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_anytime_gamma_one_collapses_x_to_w():
    config = DiLoCoConfig(
        inner_steps=H, outer='mu2', mu2_variant='anytime', mu2_anytime_gamma=1.0
    )
    model = tiny_model(0)
    inner = torch.optim.SGD(list(model.parameters()), lr=0.0)
    outer = DiLoCoOuterOptimizer(inner, config)
    outer.allocate_state()
    torch.manual_seed(7)
    for t in range(1, OUTER_ROUNDS + 1):
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        outer.outer_step(t * H)
        for state, p in outer.items:
            torch.testing.assert_close(state['outer_w'], state['outer_x'], rtol=0, atol=0)


def test_state_dict_round_trip():
    config = DiLoCoConfig(inner_steps=H, outer='mu2', outer_precond='adam')
    model = tiny_model(0)
    inner = torch.optim.SGD(list(model.parameters()), lr=0.0)
    outer = DiLoCoOuterOptimizer(inner, config)
    outer.allocate_state()
    torch.manual_seed(11)
    for t in range(1, 3):
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        outer.outer_step(t * H)
    # deepcopy: torch's state_dict() hands back live references, a real checkpoint would not.
    saved = copy.deepcopy(inner.state_dict())

    reloaded = tiny_model(0)
    inner2 = torch.optim.SGD(list(reloaded.parameters()), lr=0.0)
    outer2 = DiLoCoOuterOptimizer(inner2, config)
    inner2.load_state_dict(saved)

    keys = outer.state_keys()
    assert set(keys) == {'outer_x', 'outer_b', 'outer_dprev', 'outer_w', 'outer_v'}
    for (state, _), (state2, _) in zip(outer.items, outer2.items):
        for key in keys:
            torch.testing.assert_close(state[key], state2[key], rtol=0, atol=0)

    # The outer step must continue identically from the reloaded state.
    torch.manual_seed(13)
    values = [torch.randn_like(p) * 0.1 for p in model.parameters()]
    for module, opt in ((model, outer), (reloaded, outer2)):
        with torch.no_grad():
            for p, value in zip(module.parameters(), values):
                p.copy_(value)
        opt.outer_step(3 * H)
    torch.testing.assert_close(flat(model), flat(reloaded), rtol=0, atol=0)


def test_init_state_fn_wrapper_allocates_outer_state():
    """attach_to() must add the outer tensors to whatever the inner init_state_fn builds."""

    class FakeMegatronOptimizer:
        def __init__(self, torch_optimizer):
            self.optimizer = torch_optimizer

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            opt.state[p]['exp_avg'] = torch.zeros_like(p)

            self.init_state_fn = init_state_fn

    model = tiny_model(0)
    inner = torch.optim.SGD(list(model.parameters()), lr=0.0)
    megatron_optimizer = FakeMegatronOptimizer(inner)
    outer = DiLoCoOuterOptimizer(inner, DiLoCoConfig(inner_steps=H, outer='mu2'))
    outer.attach_to(megatron_optimizer)

    megatron_optimizer.init_state_fn(inner, None)
    for p in model.parameters():
        assert 'exp_avg' in inner.state[p]
        for key in outer.state_keys():
            assert key in inner.state[p]
    torch.testing.assert_close(
        inner.state[list(model.parameters())[0]]['outer_x'],
        list(model.parameters())[0].detach(),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# 4 gloo processes: K = 2 workers x 2 ranks.
# ---------------------------------------------------------------------------

WORLD = 4
WORKERS = 2


def _spawn(fn, port):
    mp.spawn(_entry, args=(WORLD, fn.__name__, port), nprocs=WORLD, join=True)


def _entry(rank, world, fn_name, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        globals()[fn_name](rank, world)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _inner_step(model, optimizer, worker_group, rank, step):
    torch.manual_seed(1000 * rank + step)
    x = torch.randn(8, 4)
    optimizer.zero_grad()
    model(x).pow(2).mean().backward()
    size = 1 if worker_group is None else dist.get_world_size(worker_group)
    for p in model.parameters():
        if size > 1:
            dist.all_reduce(p.grad, group=worker_group)
            p.grad.div_(size)
    optimizer.step()


def _worker_group_ranks(rank):
    return [0, 1] if rank < 2 else [2, 3]


def _body_subgroups(rank, world):
    group = build_worker_process_groups(dist.group.WORLD, WORKERS)
    assert dist.get_process_group_ranks(group) == _worker_group_ranks(rank)
    value = torch.tensor([float(rank)])
    dist.all_reduce(value, group=group)
    assert value.item() == float(sum(_worker_group_ranks(rank)))


def _body_workers_diverge_then_agree(rank, world):
    group = build_worker_process_groups(dist.group.WORLD, WORKERS)
    model = tiny_model(0)
    inner = torch.optim.SGD(list(model.parameters()), lr=0.1)
    config = DiLoCoConfig(workers=WORKERS, inner_steps=H, outer='nesterov')
    outer = DiLoCoOuterOptimizer(inner, config, dp_group=dist.group.WORLD)
    outer.allocate_state()

    for step in range(H - 1):
        _inner_step(model, inner, group, rank, step)

    # Same worker => identical params; different worker => different params.
    gathered = [torch.zeros_like(flat(model)) for _ in range(world)]
    dist.all_gather(gathered, flat(model))
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    torch.testing.assert_close(gathered[2], gathered[3], rtol=0, atol=0)
    assert not torch.equal(gathered[0], gathered[2])

    _inner_step(model, inner, group, rank, H - 1)
    outer.outer_step(H)
    dist.all_gather(gathered, flat(model))
    for other in gathered[1:]:
        torch.testing.assert_close(gathered[0], other, rtol=0, atol=0)


def _body_identity_outer_step(rank, world):
    """K = 1 (no worker split), outer lr 1 / momentum 0 => the outer step is algebraically x_H.

    Not bit-exact: x - (x - p) round-trips exactly only while Sterbenz's condition holds, which it
    does not for a weight whose magnitude is far below the round's parameter change.
    """
    config = DiLoCoConfig(workers=1, inner_steps=H, outer_lr=1.0, outer_momentum=0.0)
    config.outer_nesterov = False

    with_outer = tiny_model(0)
    inner = torch.optim.SGD(list(with_outer.parameters()), lr=0.1)
    outer = DiLoCoOuterOptimizer(inner, config, dp_group=dist.group.WORLD)
    outer.allocate_state()
    plain = tiny_model(0)
    plain_inner = torch.optim.SGD(list(plain.parameters()), lr=0.1)

    for step in range(3 * H):
        _inner_step(with_outer, inner, dist.group.WORLD, rank, step)
        _inner_step(plain, plain_inner, dist.group.WORLD, rank, step)
        if (step + 1) % H == 0:
            outer.outer_step(step + 1)
        torch.testing.assert_close(flat(with_outer), flat(plain), rtol=0, atol=1e-9)


def _body_h1_equals_parameter_averaging(rank, world):
    """H = 1, outer lr 1, momentum 0 => x_{t+1} is exactly the mean of the workers' params."""
    group = build_worker_process_groups(dist.group.WORLD, WORKERS)
    config = DiLoCoConfig(workers=WORKERS, inner_steps=1, outer_lr=1.0, outer_momentum=0.0)
    config.outer_nesterov = False
    model = tiny_model(0)
    inner = torch.optim.SGD(list(model.parameters()), lr=0.1)
    outer = DiLoCoOuterOptimizer(inner, config, dp_group=dist.group.WORLD)
    outer.allocate_state()

    averaged = tiny_model(0)
    averaged_inner = torch.optim.SGD(list(averaged.parameters()), lr=0.1)

    for step in range(5):
        _inner_step(model, inner, group, rank, step)
        expected = flat(model).clone()
        dist.all_reduce(expected)
        expected.div_(world)
        outer.outer_step(step + 1)
        torch.testing.assert_close(flat(model), expected, rtol=0, atol=1e-6)

        # ... and that matches plain data-parallel SGD over the whole world.
        _inner_step(averaged, averaged_inner, dist.group.WORLD, rank, step)
        torch.testing.assert_close(flat(model), flat(averaged), rtol=0, atol=1e-6)


def test_build_worker_process_groups():
    _spawn(_body_subgroups, 29531)


def test_inner_all_reduce_is_worker_local():
    _spawn(_body_workers_diverge_then_agree, 29532)


def test_single_worker_outer_step_is_identity():
    _spawn(_body_identity_outer_step, 29533)


def test_h1_equals_parameter_averaging():
    _spawn(_body_h1_equals_parameter_averaging, 29534)


def test_pg_collection_routes_dp_to_worker_group():
    """`pg.dp` alone must reach the dense grad buffers' collective group (intra_dp_cp)."""
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    class FakeGroup:
        def __init__(self, name):
            self.name = name

    pg_collection = ProcessGroupCollection()
    pg_collection.dp = FakeGroup('worker')
    pg_collection.tp = FakeGroup('tp')
    pg_collection.pp = FakeGroup('pp')
    pg_collection.ep = FakeGroup('ep')
    pg_collection.expt_dp = FakeGroup('expt_dp')

    class FakeConfig:
        context_parallel_size = 1

    groups = ProcessGroupCollection.setup_process_groups_for_ddp(
        pg_collection, FakeConfig(), DistributedDataParallelConfig()
    )
    assert groups['dp_group'] is pg_collection.dp
    assert groups['dp_cp_group'] is pg_collection.dp
    assert groups['intra_dp_cp_group'] is pg_collection.dp
