# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL-phase optimizer policy: moment reset, gains freeze, plain AdamW, RL LRs, weight snapshot."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import rl_phase
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.training.checkpointing import _weight_tensors, restore_weights_snapshot

GAIN_MOMENTS = ("row_gain_m", "row_gain_v", "col_gain_m", "col_gain_v")


def _md(params, **kwargs):
    kwargs.setdefault("lr", 1e-2)
    kwargs.setdefault("betas", (0.9, 0.95))
    return MDDecoupling(
        params=params,
        hypersphere_mode="row",
        hypersphere_gains_mode="rowcol",
        gains_lr=1e-2,
        use_orthogonal_updates=False,
        pg_collection=None,
        **kwargs,
    )


def _param(seed=0, shape=(4, 6)):
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape, generator=gen))


def _step(optimizer, params, seed):
    gen = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen)
    optimizer.step()


def _clone_state(optimizer):
    return {
        id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        for p, s in optimizer.state.items()
    }


def test_reset_moments_zeroes_md_and_adam_moments_only():
    md_param, adam_param = _param(0), _param(1, (5,))
    md = _md([md_param])
    adam = torch.optim.AdamW([adam_param], lr=1e-2)
    for seed in range(3):
        _step(md, [md_param], seed)
        _step(adam, [adam_param], 10 + seed)
    weights = md_param.detach().clone()
    gains = {k: md.state[md_param][k].clone() for k in ("row_gain", "col_gain")}
    assert md.state[md_param]["exp_avg"].abs().sum() > 0
    assert md.param_groups[0]["step"] == 3

    rl_phase.reset_moments(SimpleNamespace(chained_optimizers=[md, adam]))

    state = md.state[md_param]
    for key in ("exp_avg", "exp_avg_sq") + GAIN_MOMENTS:
        assert state[key].abs().sum() == 0, key
    assert md.param_groups[0]["step"] == 0
    torch.testing.assert_close(md_param.detach(), weights, rtol=0, atol=0)
    for key, value in gains.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    adam_state = adam.state[adam_param]
    assert adam_state["exp_avg"].abs().sum() == 0
    assert adam_state["exp_avg_sq"].abs().sum() == 0
    assert float(adam_state["step"]) == 0

    # Fresh moments behave like a fresh optimizer: the next step uses step-1 bias correction.
    _step(md, [md_param], 99)
    assert md.param_groups[0]["step"] == 1


def test_freeze_gains_keeps_gains_and_moves_direction():
    param = _param(0)
    md = _md([param], freeze_gains=True)
    _step(md, [param], 0)  # initializes the gain state
    state = md.state[param]
    before = {k: state[k].clone() for k in ("row_gain", "col_gain") + GAIN_MOMENTS}
    weights = param.detach().clone()

    for seed in range(1, 4):
        _step(md, [param], seed)

    for key, value in before.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    assert not torch.allclose(param.detach(), weights)


def test_gains_lr_override_is_used_verbatim():
    params = [_param(0), _param(0)]
    frozen, overridden = _md([params[0]]), _md([params[1]])
    frozen.freeze_gains = True
    overridden.gains_lr_override = 0.0
    for seed in range(3):
        _step(frozen, [params[0]], seed)
        _step(overridden, [params[1]], seed)
    # A zero gains LR leaves the gains where freezing does.
    for key in ("row_gain", "col_gain"):
        torch.testing.assert_close(overridden.state[params[1]][key], frozen.state[params[0]][key])


def test_plain_adamw_matches_torch_adamw_and_leaves_md_state():
    param, reference = _param(0), _param(0)
    md = _md([param], weight_decay=0.01)
    _step(md, [param], 0)  # MD state (momentum, gains, gain moments) exists
    reference.data.copy_(param.data)
    md_state = _clone_state(md)
    md_step = md.param_groups[0]["step"]

    rl_phase.set_plain_adamw(SimpleNamespace(chained_optimizers=[md]), True)
    rl_phase.set_adam_hparams(SimpleNamespace(chained_optimizers=[md]), beta2=0.95, eps=1e-15)
    torch_adamw = torch.optim.AdamW(
        [reference], lr=md.param_groups[0]["lr"], betas=(0.9, 0.95), eps=1e-15, weight_decay=0.01
    )
    for seed in range(1, 5):
        _step(md, [param], seed)
        _step(torch_adamw, [reference], seed)
        torch.testing.assert_close(param.detach(), reference.detach(), rtol=1e-6, atol=1e-7)

    assert md.param_groups[0]["step"] == md_step
    for key, value in md_state[id(param)].items():
        torch.testing.assert_close(md.state[param][key], value, rtol=0, atol=0)
    assert md.param_groups[0]["plain_adamw_step"] == 4


def test_set_learning_rates_routes_groups():
    muon_param, adam_param, external_param = _param(0), _param(1), _param(2, (5,))
    md = MDDecoupling(
        params=[
            {"params": [muon_param], "use_orthogonal_updates": True, "wd_mult": 1.0},
            {"params": [adam_param], "use_orthogonal_updates": False, "wd_mult": 0.0},
        ],
        lr=1.0,
        hypersphere_gains_mode="rowcol",
        pg_collection=None,
    )
    external = torch.optim.AdamW([external_param], lr=1.0)
    chained = SimpleNamespace(chained_optimizers=[md, external])

    rl_phase.set_learning_rates(
        chained, lr=1e-6, matrix_lr=4e-5, gains_lr=3e-5, weight_decay=0.1, scale=0.5
    )
    assert md.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert md.param_groups[1]["lr"] == pytest.approx(5e-7)
    assert external.param_groups[0]["lr"] == pytest.approx(5e-7)
    assert md.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert md.param_groups[1]["weight_decay"] == 0.0
    assert md.gains_lr_override == pytest.approx(1.5e-5)
    assert md.gains_weight_decay == pytest.approx(0.1)

    rl_phase.set_plain_adamw(chained, True)
    rl_phase.set_learning_rates(chained, lr=1e-6, matrix_lr=4e-5, gains_lr=3e-5, weight_decay=0.0)
    assert md.param_groups[0]["lr"] == pytest.approx(1e-6)

    rl_phase.set_adam_hparams(chained, beta2=0.95, eps=1e-15)
    assert md.param_groups[1]["beta2"] == 0.95 and md.param_groups[1]["eps"] == 1e-15
    assert external.param_groups[0]["betas"] == (0.9, 0.95)
    assert external.param_groups[0]["eps"] == 1e-15
    assert md.gains_betas[1] == 0.95 and md.gains_eps == 1e-15


def test_warmup_scale():
    assert rl_phase.warmup_scale(0, 0) == 1.0
    assert rl_phase.warmup_scale(0, 15) == pytest.approx(1 / 15)
    assert rl_phase.warmup_scale(14, 15) == 1.0
    assert rl_phase.warmup_scale(100, 15) == 1.0


def test_weight_snapshot_restore_round_trip():
    model = [torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3)]
    masters = [[torch.randn(3, 4)], [torch.randn(3)]]
    optimizer = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(fp32_from_float16_groups=masters)]
    )
    snapshot = [t.detach().clone() for t in _weight_tensors(model, optimizer)]
    with torch.no_grad():
        for t in _weight_tensors(model, optimizer):
            t.add_(1)

    restore_weights_snapshot(model, optimizer, snapshot)

    for tensor, saved in zip(_weight_tensors(model, optimizer), snapshot):
        torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        restore_weights_snapshot(model, optimizer, snapshot[:-1])
