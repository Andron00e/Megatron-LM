# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""CPU/gloo tests for megatron/core/distributed/diloco.py.

Megatron's own DDP allocates its grad buffers on `torch.cuda.current_device()`, so the multi-process
tests here emulate the only thing DDP contributes to the algorithm -- a gradient all-reduce over
`pg.dp`, which DiLoCo retargets to the worker sub-group -- with an explicit all-reduce over the
group `build_worker_process_groups` hands back. `test_pg_collection_routes_dp_to_worker_group`
covers the claim that setting `pg.dp` alone is enough to retarget every dense grad-buffer
collective, without needing a GPU.
"""

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
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.optimizer.mars import MARS
from megatron.core.optimizer.mu2mars import Mu2MARS

H = 3
OUTER_ROUNDS = 5

# Every inner optimizer of interest allocates its per-param state lazily under
# `if len(state) == 0`, which is what the outer state must not defeat (F020). SGD is not in the
# list on purpose: with momentum 0 it has no per-param state at all, so it cannot see the bug.
INNER_OPTIMIZERS = {
    'adamw': lambda params: torch.optim.AdamW(params, lr=1e-2),
    'adam': lambda params: torch.optim.Adam(params, lr=1e-2),
    'mars': lambda params: MARS(params, lr=1e-2),
    'mu2mars-ema': lambda params: Mu2MARS(params, lr=1e-2, variant='ema'),
    'mu2mars-anytime': lambda params: Mu2MARS(params, lr=1e-2, variant='anytime'),
}


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


def local_step(model, optimizer, step):
    """One ordinary inner iteration, no distribution."""
    torch.manual_seed(step)
    optimizer.zero_grad()
    model(torch.randn(8, 4)).pow(2).mean().backward()
    optimizer.step()


def prescribed_grad_step(params, optimizer, grads):
    """One inner step on a fixed gradient; returns the l2 norm of the parameter displacement."""
    before = [p.detach().clone() for p in params]
    for p, grad in zip(params, grads):
        p.grad = grad.clone()
    optimizer.step()
    return torch.cat(
        [(p.detach() - b).reshape(-1) for p, b in zip(params, before)]
    ).norm().item()


@pytest.mark.parametrize("name", sorted(INNER_OPTIMIZERS))
def test_outer_state_never_enters_the_inner_optimizer(name):
    """The outer tensors stay in the controller, before and after a round.

    Writing them into `opt.state[p]` defeated every inner optimizer's lazy init (F020) and, once
    that was deferred, made the inner `state_dict()` responsible for keys it does not know --
    which is how TE's FusedAdam killed every save (F039).
    """
    model = tiny_model(0)
    inner = INNER_OPTIMIZERS[name](model.parameters())
    outer = DiLoCoOuterOptimizer(inner, DiLoCoConfig(inner_steps=H, outer='mu2'))
    outer.allocate_state()
    for p in model.parameters():
        assert inner.state.get(p, {}) == {}

    for step in range(H):
        local_step(model, inner, step)  # KeyError('exp_avg') before the fix
        outer.note_inner_step(step + 1)
    assert outer.should_sync(H)
    outer.outer_step(H)

    for p in model.parameters():
        for key in outer.state_keys():
            assert key not in inner.state[p]
            assert key in outer.outer_state[p]


class StrictStateOptimizer(torch.optim.Optimizer):
    """A stand-in for TransformerEngine's FusedAdam, which is what `--optimizer adam` resolves to
    in the cluster container and cannot be imported here.

    Its `state_dict()` looks every per-param state key up in a fixed `name_to_dtype_map` and
    raises `KeyError` on anything else, which is the exact shape of F039.
    """

    name_to_dtype_map = {'exp_avg': torch.float32, 'exp_avg_sq': torch.float32}

    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg'].mul_(0.9).add_(p.grad, alpha=0.1)
                p.add_(state['exp_avg'], alpha=-group['lr'])

    def state_dict(self):
        for state in self.state.values():
            for name in state:
                _ = self.name_to_dtype_map[name]  # KeyError: 'outer_x' (F039)
        return super().state_dict()


def test_inner_state_dict_survives_a_round_with_a_fused_adam_like_optimizer():
    """The regression: a full round, then a save, with a fixed-key-table inner optimizer."""
    model = tiny_model(0)
    inner = StrictStateOptimizer(model.parameters())
    outer = DiLoCoOuterOptimizer(inner, DiLoCoConfig(inner_steps=H, outer='mu2'))
    outer.allocate_state()
    for step in range(H):
        local_step(model, inner, step)
        outer.note_inner_step(step + 1)
    outer.outer_step(H)

    saved = inner.state_dict()  # KeyError('outer_x') before the fix
    for state in saved['state'].values():
        assert set(state) == {'exp_avg', 'exp_avg_sq'}


def test_outer_step_translates_the_anytime_inner_descent_sequence():
    """mu2mars anytime keeps a copy of the params; left stale it undoes part of the outer step."""
    model = tiny_model(0)
    inner = Mu2MARS(model.parameters(), lr=0.1, variant='anytime', anytime_gamma=0.1)
    outer = DiLoCoOuterOptimizer(inner, DiLoCoConfig(inner_steps=H, outer='nesterov'))
    outer.allocate_state()
    for step in range(H):
        local_step(model, inner, step)
        outer.note_inner_step(step + 1)

    # Only the matrix params take the anytime branch and so carry `w`; 1-D params are AdamW.
    before = {
        p: (p.detach().clone(), inner.state[p]['w'].clone())
        for p in model.parameters()
        if 'w' in inner.state[p]
    }
    outer.outer_step(H)

    moved = 0
    for p, (p_before, w_before) in before.items():
        displacement = p.detach() - p_before
        if displacement.norm() > 1e-3:
            moved += 1
        # w follows the params: it is neither left on the pre-jump trajectory (F026) nor reset
        # onto x_{t+1}, which would zero the lead w - p the anytime pair runs on (F032).
        torch.testing.assert_close(
            inner.state[p]['w'], w_before + displacement, rtol=1e-6, atol=1e-7
        )
    assert moved > 0


def test_outer_step_preserves_the_anytime_inner_step_size():
    """The whole point of translating w: the inner optimizer must not feel the outer jump (F032).

    The gradient is prescribed rather than taken from the model, so with weight_decay 0 the
    mu2mars update depends only on the optimizer state -- a control that never saw the outer step
    then gives the inner displacement the round would have had, exactly.
    """
    model, control_model = tiny_model(0), tiny_model(0)
    params = list(model.parameters())
    control_params = list(control_model.parameters())
    inner = Mu2MARS(params, lr=0.05, variant='anytime', anytime_gamma=0.1, weight_decay=0.0)
    control_inner = Mu2MARS(
        control_params, lr=0.05, variant='anytime', anytime_gamma=0.1, weight_decay=0.0
    )
    outer = DiLoCoOuterOptimizer(inner, DiLoCoConfig(inner_steps=H, outer='nesterov'))
    outer.allocate_state()

    torch.manual_seed(7)
    grads = [torch.randn_like(p) * 0.1 for p in params]
    for step in range(20):
        prescribed_grad_step(params, inner, grads)
        prescribed_grad_step(control_params, control_inner, grads)
        outer.note_inner_step(step + 1)

    # The 20 inner steps have moved the params away from the anchor, so this is a real jump.
    outer.outer_step(20)
    assert (
        torch.cat([(p.detach() - c.detach()).reshape(-1) for p, c in zip(params, control_params)])
        .norm()
        .item()
        > 1e-3
    )

    for _ in range(5):
        after = prescribed_grad_step(params, inner, grads)
        undisturbed = prescribed_grad_step(control_params, control_inner, grads)
        assert undisturbed > 0
        assert after == pytest.approx(undisturbed, rel=1e-6)


def test_forced_sync_starts_a_full_round_and_does_not_replay_t():
    """A preemption sync at an unaligned iteration must not shorten the next round (F025)."""
    config = DiLoCoConfig(inner_steps=100, outer='mu2')
    model = tiny_model(0)
    outer = DiLoCoOuterOptimizer(torch.optim.SGD(list(model.parameters()), lr=0.0), config)
    outer.allocate_state()

    assert outer.should_sync(100)
    outer.outer_step(100)
    assert outer.outer_steps == 1
    assert not outer.should_sync(150)
    outer.outer_step(150)  # forced, unscheduled
    assert outer.outer_steps == 2
    assert not outer.should_sync(200)
    assert outer.should_sync(250)
    outer.outer_step(250)
    assert outer.outer_steps == 3

    outer.resume_from(250)
    assert outer.outer_steps == 3
    assert not outer.should_sync(300)
    assert outer.should_sync(350)


def _trained_controller(config, seed=0, rounds=2, optimizer_factory=None):
    """A controller with a non-trivial outer state: `rounds` full rounds of real inner steps."""
    model = tiny_model(seed)
    factory = optimizer_factory or (lambda params: torch.optim.AdamW(params, lr=1e-2))
    inner = factory(model.parameters())
    outer = DiLoCoOuterOptimizer(inner, config)
    outer.allocate_state()
    torch.manual_seed(11)
    for t in range(1, rounds + 1):
        for step in range(H):
            local_step(model, inner, t * H + step)
            outer.note_inner_step((t - 1) * H + step + 1)
        outer.outer_step(t * H)
    return model, inner, outer


def _save(outer):
    """What a checkpoint keeps: values, not the live tensors `state_dict()` hands back."""
    return {key: [t.clone() for t in tensors] for key, tensors in outer.state_dict().items()}


def test_state_dict_round_trip():
    config = DiLoCoConfig(inner_steps=H, outer='mu2', outer_precond='adam')
    model, inner, outer = _trained_controller(config)
    saved = _save(outer)

    # Nothing of the outer state leaked into the inner optimizer's own checkpoint section (F039).
    for state in inner.state_dict()['state'].values():
        assert not any(key.startswith('outer_') for key in state)

    reloaded = tiny_model(0)
    inner2 = torch.optim.AdamW(reloaded.parameters(), lr=1e-2)
    outer2 = DiLoCoOuterOptimizer(inner2, config)
    outer2.load_state_dict(saved)  # as load_checkpoint does
    outer2.resume_from(2 * H)  # ... and then the outer clock is re-anchored (F025)
    assert outer2.outer_steps == outer.outer_steps

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


def test_state_dict_is_independent_of_the_worker_count():
    """A K=1 checkpoint must resume at K=2 and back: outer state is per-param, not per-worker."""
    saved_k1 = _save(_trained_controller(DiLoCoConfig(workers=1, inner_steps=H, outer='mu2'))[2])
    saved_k2 = _save(_trained_controller(DiLoCoConfig(workers=2, inner_steps=H, outer='mu2'))[2])
    assert saved_k1.keys() == saved_k2.keys()

    for config, saved in (
        (DiLoCoConfig(workers=2, inner_steps=H, outer='mu2'), saved_k1),
        (DiLoCoConfig(workers=1, inner_steps=H, outer='mu2'), saved_k2),
    ):
        model = tiny_model(0)
        outer = DiLoCoOuterOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-2), config)
        outer.load_state_dict(saved)
        for key, tensors in saved.items():
            for (state, _), tensor in zip(outer.items, tensors):
                torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)


class FakeMegatronOptimizer:
    """The `Float16OptimizerWithFloat16Params` shape the sharded param map is built from: model
    params, their fp32 copies in lockstep, and the inner optimizer over the fp32 copies."""

    def __init__(self, model):
        self.float16_groups = [list(model.parameters())]
        self.fp32_from_float16_groups = [
            [p.detach().clone().float() for p in model.parameters()]
        ]
        self.fp32_from_fp32_groups = [[]]
        self.optimizer = torch.optim.AdamW(self.fp32_from_float16_groups[0], lr=1e-2)


def _fake_model_sharded_state_dict(megatron_optimizer):
    """`keep_vars=True` model state: the ShardedTensors wrap the parameters themselves."""
    return {
        'model': {
            f'layer{i}.weight': ShardedTensor.from_rank_offsets(
                f'layer{i}.weight', p, replica_id=(0, 0, 3)
            )
            for i, p in enumerate(megatron_optimizer.float16_groups[0])
        }
    }


def test_sharded_state_dict_round_trip():
    """The torch_dist path: outer tensors as ShardedTensors under the model's own sharding."""
    config = DiLoCoConfig(inner_steps=H, outer='mu2', outer_precond='adam')
    model = tiny_model(0)
    megatron_optimizer = FakeMegatronOptimizer(model)
    outer = DiLoCoOuterOptimizer(megatron_optimizer.optimizer, config)
    outer.attach_to(megatron_optimizer)
    model_sd = _fake_model_sharded_state_dict(megatron_optimizer)

    sharded = outer.sharded_state_dict(model_sd, is_loading=True)
    assert set(sharded) == set(outer.state_keys())
    main_params = megatron_optimizer.fp32_from_float16_groups[0]
    for key, entry in sharded.items():
        assert sorted(entry) == list(range(len(main_params)))
        for index, sh_ten in entry.items():
            assert sh_ten.key == f'diloco.{key}.layer{index}.weight'
            assert sh_ten.dtype == torch.float32
            # The model's sharding, replica_id included, so one DP rank writes the outer state.
            assert sh_ten.replica_id == (0, 0, 3)
            assert sh_ten.data is outer.outer_state[main_params[index]][key]

    torch.manual_seed(5)
    for state, _ in outer.items:
        for key in outer.state_keys():
            state[key].copy_(torch.randn_like(state[key]))
    # What comes back from a torch_dist load: the same structure, plain tensors.
    loaded = {
        key: {index: sh_ten.data.clone() for index, sh_ten in entry.items()}
        for key, entry in outer.sharded_state_dict(model_sd).items()
    }

    reloaded = tiny_model(1)
    megatron_optimizer2 = FakeMegatronOptimizer(reloaded)
    outer2 = DiLoCoOuterOptimizer(megatron_optimizer2.optimizer, config)
    outer2.attach_to(megatron_optimizer2)
    outer2.load_state_dict(loaded)
    for (state, _), (state2, _) in zip(outer.items, outer2.items):
        for key in outer.state_keys():
            torch.testing.assert_close(state[key], state2[key], rtol=0, atol=0)


def test_sharded_state_dict_rejects_a_param_the_model_does_not_cover():
    config = DiLoCoConfig(inner_steps=H, outer='nesterov')
    model = tiny_model(0)
    megatron_optimizer = FakeMegatronOptimizer(model)
    outer = DiLoCoOuterOptimizer(megatron_optimizer.optimizer, config)
    outer.attach_to(megatron_optimizer)
    model_sd = _fake_model_sharded_state_dict(megatron_optimizer)
    model_sd['model'].pop('layer0.weight')
    with pytest.raises(ValueError, match='does not match any model sharded param'):
        outer.sharded_state_dict(model_sd)


def test_load_state_dict_rejects_a_mismatched_checkpoint():
    config = DiLoCoConfig(inner_steps=H, outer='mu2')
    model = tiny_model(0)
    outer = DiLoCoOuterOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-2), config)
    saved = _save(outer)
    with pytest.raises(ValueError, match='missing from the checkpoint'):
        outer.load_state_dict({key: value for key, value in saved.items() if key != 'outer_w'})
    truncated = dict(saved)
    truncated['outer_w'] = truncated['outer_w'][:-1]
    with pytest.raises(ValueError, match='expected'):
        outer.load_state_dict(truncated)


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


def _body_fresh_run_with_real_inner_optimizer(rank, world):
    """A fresh K=2 run over 3 outer rounds with an inner optimizer that initialises lazily.

    This is the production sequence -- allocate_state() at setup, then note_inner_step /
    should_sync / outer_step per iteration -- and it raised KeyError('exp_avg') on the first
    inner step before F020 was fixed.
    """
    group = build_worker_process_groups(dist.group.WORLD, WORKERS)
    for name in ('adamw', 'mu2mars-ema', 'mu2mars-anytime'):
        model = tiny_model(0)
        inner = INNER_OPTIMIZERS[name](model.parameters())
        config = DiLoCoConfig(
            workers=WORKERS, inner_steps=H, outer='mu2', mu2_variant='anytime', verify_sync=True
        )
        outer = DiLoCoOuterOptimizer(inner, config, dp_group=dist.group.WORLD)
        outer.allocate_state()

        gathered = [torch.zeros_like(flat(model)) for _ in range(world)]
        for step in range(3 * H):
            _inner_step(model, inner, group, rank, step)
            outer.note_inner_step(step + 1)
            if outer.should_sync(step + 1):
                leads = {
                    p: inner.state[p]['w'] - p.detach().float()
                    for p in model.parameters()
                    if 'w' in inner.state[p]
                }
                outer.outer_step(step + 1)
                # The anytime descent sequence is translated by the outer jump, so the lead it
                # runs on survives the round boundary (F032); it stays worker-local like the
                # moments, so it is not part of the cross-worker agreement checked below.
                for p, lead in leads.items():
                    torch.testing.assert_close(
                        inner.state[p]['w'] - p.detach().float(),
                        lead,
                        rtol=1e-6,
                        atol=1e-7,
                        msg=name,
                    )
                dist.all_gather(gathered, flat(model))
                for other in gathered[1:]:
                    torch.testing.assert_close(gathered[0], other, rtol=0, atol=0, msg=name)
            else:
                dist.all_gather(gathered, flat(model))
                assert not torch.equal(gathered[0], gathered[2]), name

        assert outer.outer_steps == 3, name
        for p in model.parameters():
            assert torch.isfinite(p).all(), name
            for key in outer.state_keys():
                assert key not in inner.state[p], (name, key)
                assert key in outer.outer_state[p], (name, key)
            if name == 'mu2mars-anytime' and 'w' in inner.state[p]:
                assert (inner.state[p]['w'] - p.detach().float()).abs().max() > 0, name


def _body_verify_sync_catches_cancelling_divergence(rank, world):
    """A signed whole-model sum cannot separate [1, -1] from [-1, 1] (F029)."""
    model = tiny_model(0)
    outer = DiLoCoOuterOptimizer(
        torch.optim.SGD(list(model.parameters()), lr=0.0),
        DiLoCoConfig(workers=WORKERS, inner_steps=H, verify_sync=True),
        dp_group=dist.group.WORLD,
    )
    diverged = [torch.tensor([1.0, -1.0] if rank < 2 else [-1.0, 1.0])]
    assert diverged[0].sum().item() == 0.0
    with pytest.raises(AssertionError, match='workers disagree'):
        outer._verify_sync(diverged)
    outer._verify_sync([torch.tensor([1.0, -1.0])])


def test_build_worker_process_groups():
    _spawn(_body_subgroups, 29531)


def test_inner_all_reduce_is_worker_local():
    _spawn(_body_workers_diverge_then_agree, 29532)


def test_single_worker_outer_step_is_identity():
    _spawn(_body_identity_outer_step, 29533)


def test_h1_equals_parameter_averaging():
    _spawn(_body_h1_equals_parameter_averaging, 29534)


def test_fresh_run_with_real_inner_optimizer():
    _spawn(_body_fresh_run_with_real_inner_optimizer, 29535)


def test_verify_sync_catches_cancelling_divergence():
    _spawn(_body_verify_sync_catches_cancelling_divergence, 29536)


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
