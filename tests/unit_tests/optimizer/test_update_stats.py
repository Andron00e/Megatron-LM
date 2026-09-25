# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Tests for megatron/core/optimizer/update_stats.py.

Run with ``torchrun --nproc-per-node N -m pytest``; the layout tests need 4 ranks.
"""

import copy
import math
import os
import re

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer import (
    HAVE_EMERGING_OPTIMIZERS,
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.md_decoupling import get_megatron_mddecoupling_optimizer
from megatron.core.optimizer.muon_logging import collect_md_gain_stats
from megatron.core.optimizer.neutrino import get_megatron_neutrino_optimizer
from megatron.core.optimizer.update_stats import UpdateStatsCollector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


@pytest.fixture(autouse=True)
def test_environment():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    init_num_microbatches_calculator(0, None, 1, 1, 1)  # read by the MoE router's load tracking
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn
    destroy_num_microbatches_calculator()
    Utils.destroy_model_parallel()


def _pg_collection():
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    return pg_collection


# ----------------------------------------------------------------------
# exactness on a toy model against direct torch computations
# ----------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, layer_number, hidden):
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = nn.LayerNorm(hidden, bias=False)
        self.self_attention = nn.Module()
        self.self_attention.linear_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.self_attention.linear_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(hidden, 2 * hidden, bias=False)
        self.mlp.linear_fc2 = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, h):
        q, k, v = self.self_attention.linear_qkv(self.input_layernorm(h)).chunk(3, dim=-1)
        h = h + self.self_attention.linear_proj(torch.tanh(q) * k + v)
        return h + self.mlp.linear_fc2(torch.relu(self.mlp.linear_fc1(h)))


class _Toy(nn.Module):
    def __init__(self, vocab=64, hidden=32, layers=2):
        super().__init__()
        self.embedding = nn.Module()
        self.embedding.word_embeddings = nn.Embedding(vocab, hidden)
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Block(i + 1, hidden) for i in range(layers)])

    def forward(self, tokens):
        h = self.embedding.word_embeddings(tokens)
        for layer in self.decoder.layers:
            h = layer(h)
        return h @ self.embedding.word_embeddings.weight.t()


_TOY_FAMILIES = {
    "word_embeddings": "embedding",
    "input_layernorm": "layernorm",
    "linear_qkv": "attention-in",
    "linear_proj": "attention-out",
    "linear_fc1": "dense-mlp-in",
    "linear_fc2": "dense-mlp-out",
}


def _toy_family(name, optimizer_name):
    family = next(f for key, f in _TOY_FAMILIES.items() if key in name)
    adam = optimizer_name == "adam" or family in ("embedding", "layernorm")
    return family + ("-adam" if adam else "")


def _build_toy(optimizer_name, bf16):
    torch.manual_seed(0)
    model = _Toy().cuda()
    if bf16:
        model = model.bfloat16()
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        model,
    )
    config = OptimizerConfig(
        optimizer=optimizer_name,
        lr=3e-2,
        weight_decay=0.1,
        bf16=bf16,
        use_distributed_optimizer=False,
        clip_grad=0.0,
        muon_momentum=0.9,
        muon_use_nesterov=True,
        muon_scale_mode="shape_up",
        neutrino_k=8,
        neutrino_metrics_interval=0,
    )
    if optimizer_name == "neutrino":
        optimizer = get_megatron_neutrino_optimizer(config, [model], pg_collection=_pg_collection())
    else:
        optimizer = get_megatron_optimizer(config, [model])
    return model, optimizer


def _reference(model, captured, inputs, optimizer_name):
    """Direct fp64 computations of the family metrics of one step."""
    buckets = {}
    for name, param in model.module.named_parameters():
        if param.ndim < 2 and "layernorm" not in name:
            continue
        main = getattr(param, "main_param", param)
        prev, grad, momentum = captured[name]
        cur = main.detach().double()
        layer = re.search(r"layers\.(\d+)\.", name)
        keys = [("update", _toy_family(name, optimizer_name))]
        if layer:
            keys.append((f"update/layers/{layer.group(1)}", _toy_family(name, optimizer_name)))
        for key in keys:
            buckets.setdefault(key, []).append((name, prev, cur, grad, momentum))
    expected = {}
    for (prefix, family), items in buckets.items():
        prev = torch.cat([p.reshape(-1) for _, p, _, _, _ in items])
        cur = torch.cat([c.reshape(-1) for _, _, c, _, _ in items])
        grad = torch.cat([g.reshape(-1) for _, _, _, g, _ in items])
        delta = cur - prev

        def e(metric, value):
            head, _, tail = metric.partition("/")
            expected[f"{prefix}/{head}/{family}" + (f"/{tail}" if tail else "")] = float(value)

        e("relative", delta.norm() / prev.norm())
        e("weight-norm", prev.norm())
        cos = (cur @ prev) / (cur.norm() * prev.norm())
        e("angle-cos", cos)
        e("angle-deg", math.degrees(math.acos(min(1.0, float(cos)))))
        e("grad-norm", grad.norm())
        radial = (grad @ prev) / prev.norm()
        e("grad-weight-cos", (grad @ prev) / (grad.norm() * prev.norm()))
        e("grad-radial", radial)
        tangential = (grad.square().sum() - radial**2).clamp_min(0).sqrt()
        e("grad-tangential", tangential)
        e("grad-tangential-fraction", tangential / grad.norm())
        if all(m is not None for *_, m in items):
            momentum = torch.cat([m.reshape(-1) for *_, m in items])
            e("grad-momentum-cos", (grad @ momentum) / (grad.norm() * momentum.norm()))
            e("momentum-grad-norm-ratio", momentum.norm() / grad.norm())
        matrices = [(p, c) for _, p, c, _, _ in items if p.ndim == 2]
        if matrices:
            ratios = torch.cat([((c - p).norm(dim=1) / p.norm(dim=1)) for p, c in matrices])
            e("per-neuron-relative/mean", ratios.mean())
            e("per-neuron-relative/max", ratios.max())
            rows = torch.cat([(c - p).norm(dim=1) for p, c in matrices])
            cols = torch.cat([(c - p).norm(dim=0) for p, c in matrices])
            e("delta-row-norm/mean", rows.mean())
            e("delta-row-norm/std", rows.std(unbiased=False))
            e("delta-col-norm/mean", cols.mean())
            e("delta-col-norm/std", cols.std(unbiased=False))
            srank = [p.square().sum() / torch.linalg.matrix_norm(p, 2) ** 2 for p, _ in matrices]
            e("stable-rank", sum(srank) / len(srank))
            with_momentum = [m for _, _, _, _, m in items if m is not None and m.ndim == 2]
            if with_momentum:
                reff = [torch.linalg.svdvals(m).sum() ** 2 / m.square().sum() for m in with_momentum]
                e("momentum-r-eff", sum(reff) / len(reff))
        dy = [(inputs[n], p, c) for n, p, c, _, _ in items if n in inputs]
        if dy:
            num = sum((x @ (c - p).t()).square().sum() for x, p, c in dy)
            den = sum((x @ p.t()).square().sum() for x, p, c in dy)
            e("delta-y", (num / den).sqrt())
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("optimizer_name", ["adam", "neutrino"])
@pytest.mark.parametrize("bf16", [False, True])
def test_toy_exactness(optimizer_name, bf16):
    """Every family metric equals a direct fp64 computation; DP replicas count once."""
    Utils.initialize_model_parallel(1, 1)
    model, optimizer = _build_toy(optimizer_name, bf16)
    collector = UpdateStatsCollector(
        [model], optimizer, interval=1, per_layer=True, spectral_interval=1, delta_y=True, num_layers=2
    )
    generator = torch.Generator().manual_seed(1)
    inputs = {}
    hooks = [
        module.register_forward_hook(
            lambda m, args, out, name=name: inputs.__setitem__(name + ".weight", args[0].detach().double().reshape(-1, args[0].shape[-1]))
        )
        for name, module in model.module.named_modules()
        if name.endswith(("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"))
    ]
    for iteration in range(3):
        tokens = torch.randint(0, 64, (4, 16), generator=generator).cuda()
        model.zero_grad_buffer()
        optimizer.zero_grad()
        collector.begin_step(iteration)
        logits = model(tokens).float()
        loss = nn.functional.cross_entropy(logits.reshape(-1, 64), tokens.roll(1, 1).reshape(-1))
        loss.backward()
        model.finish_grad_sync()
        captured = {}
        states = {}
        for wrapper in optimizer.chained_optimizers:
            states.update(wrapper.optimizer.state)
        for name, param in model.module.named_parameters():
            main = getattr(param, "main_param", param)
            state = states.get(main, {})
            momentum = next(
                (state[k].double().clone() for k in ("momentum_buffer", "exp_avg") if k in state), None
            )
            captured[name] = (main.detach().double().clone(), param.main_grad.double().clone(), momentum)
        collector.before_optimizer_step()
        assert optimizer.step()[0]
        stats = collector.after_optimizer_step(True)
        expected = _reference(model, captured, inputs, optimizer_name)
        for key, value in expected.items():
            assert key in stats, (key, sorted(stats))
            assert stats[key] == pytest.approx(value, **_tolerance(key, expected)), (key, stats[key], value)
        assert stats["update/phase-step"] == iteration + 1
    for hook in hooks:
        hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers is not installed")
def test_md_decoupling_with_gain_logging():
    """MDDecoupling: update stats and muon_logging's gain stats on the same step, and dW includes
    the hypersphere re-projection."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(0)
    model = DistributedDataParallel(
        # MDDecoupling splits linear_qkv per head group from this config: one 32-wide head.
        TransformerConfig(num_attention_heads=1, num_layers=2, hidden_size=32),
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        _Toy().cuda(),
    )
    config = OptimizerConfig(
        optimizer="md_decoupling", lr=3e-2, bf16=False, use_distributed_optimizer=False, clip_grad=0.0
    )
    optimizer = get_megatron_mddecoupling_optimizer(config, [model])
    collector = UpdateStatsCollector([model], optimizer, interval=1, per_layer=True, num_layers=2)
    generator = torch.Generator().manual_seed(1)
    qkv = [p for n, p in model.module.named_parameters() if "linear_qkv" in n]
    for iteration in range(2):
        tokens = torch.randint(0, 64, (4, 16), generator=generator).cuda()
        model.zero_grad_buffer()
        optimizer.zero_grad()
        collector.begin_step(iteration)
        logits = model(tokens)
        nn.functional.cross_entropy(logits.reshape(-1, 64), tokens.roll(1, 1).reshape(-1)).backward()
        model.finish_grad_sync()
        prev = [p.detach().double().clone() for p in qkv]
        collector.before_optimizer_step()
        assert optimizer.step()[0]
        stats = collector.after_optimizer_step(True)
        gains = collect_md_gain_stats(optimizer, per_layer=True)
        assert any(key.startswith("muon-md/gains/attention-in/") for key in gains), sorted(gains)
        delta = sum((p.detach().double() - w).square().sum() for p, w in zip(qkv, prev))
        expected = (delta / sum(w.square().sum() for w in prev)).sqrt()
        assert stats["update/relative/attention-in"] == pytest.approx(float(expected), rel=1e-5)
        assert "update/grad-weight-cos/attention-in" in stats


def _tolerance(key, expected):
    """fp32 products against an fp64 reference. A std is compared relative to its mean: under
    Adam's first (sign-like) step all rows have almost the same norm and std/mean ~ 1e-5."""
    if key.endswith("/std"):
        return dict(rel=1e-5, abs=1e-7 * expected[key[: -len("std")] + "mean"])
    if "stable-rank" in key or "r-eff" in key or "angle-deg" in key:
        return dict(rel=1e-4, abs=1e-8)
    return dict(rel=1e-5, abs=1e-8)


def test_cadence():
    collector = UpdateStatsCollector.__new__(UpdateStatsCollector)
    collector.enabled, collector.interval, collector.dense_window = True, 10, 3
    collector.spectral_interval, collector._delta_y_modules = 20, []
    collector._phase_start, collector._phase_index, collector._spectral_pending = None, 0, False
    collector._inputs = {}
    active = [step + 1 for step in range(100, 145) if collector.begin_step(step)]
    assert active == [101, 102, 103, 110, 120, 130, 140]
    collector.spectral_interval = 25
    assert collector.begin_step(174) and collector._spectral  # step 175: spectra imply logging
    collector.notify_phase_change("rl")
    active = [step + 1 for step in range(180, 190) if collector.begin_step(step)]
    assert active == [181, 182, 183, 190]
    assert collector.begin_step(180) and collector._spectral is False  # pending consumed at 181


# ----------------------------------------------------------------------
# layout invariance on a tiny GPT / MoE
# ----------------------------------------------------------------------
_VOCAB, _SEQ, _BATCH = 128, 32, 4


def _gpt_config(tp, ep, moe, bf16):
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        sequence_parallel=tp > 1,
        num_moe_experts=4 if moe else None,
        moe_ffn_hidden_size=64 if moe else None,
        moe_router_topk=2,
        moe_grouped_gemm=moe,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        attention_backend=AttnBackend.unfused,
    )


def _build_gpt(tp, ep, moe, bf16, full=None):
    Utils.initialize_model_parallel(tp, 1, expert_model_parallel_size=ep)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = _gpt_config(tp, ep, moe, bf16)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=4 if moe else None, moe_grouped_gemm=moe
        ),
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    ).cuda()
    if full is not None:
        _load_full(model, full)
    return model, config


def _load_full(model, full):
    """Copy an unsharded (TP1/EP1) state into this rank's shards."""
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    for name, param in model.named_parameters():
        expert = not getattr(param, "allreduce", True)
        source = name
        if expert:
            local = model.decoder.layers[0].mlp.experts.num_local_experts
            source = re.sub(r"weight(\d+)$", lambda m: f"weight{ep_rank * local + int(m.group(1))}", name)
        value = full[source]
        group = (
            parallel_state.get_expert_tensor_parallel_group()
            if expert
            else parallel_state.get_tensor_model_parallel_group()
        )
        if getattr(param, "tensor_model_parallel", False) and group.size() > 1:
            value = value.chunk(group.size(), dim=param.partition_dim)[group.rank()]
        with torch.no_grad():
            param.copy_(value)


def _ddp(model, config, layer_wise):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False, overlap_grad_reduce=layer_wise, bucket_size=10000
    )
    return DistributedDataParallel(config, ddp_config, model)


def _adam(model, bf16, layer_wise):
    config = OptimizerConfig(
        optimizer="adam", lr=1e-2, weight_decay=0.1, bf16=False, use_distributed_optimizer=False, clip_grad=0.0
    )
    optimizer = get_megatron_optimizer(config, [model])
    if not layer_wise:
        assert not bf16
        return optimizer
    config = copy.copy(config)
    config.bf16 = bf16
    return LayerWiseDistributedOptimizer(optimizer.chained_optimizers, config, _pg_collection())


def _run_gpt(model, optimizer, config, steps=2, collector_kwargs=None):
    collector = UpdateStatsCollector(
        [model],
        optimizer,
        interval=1,
        per_layer=True,
        spectral_interval=1,
        delta_y=True,
        num_layers=2,
        num_experts=config.num_moe_experts,
        **(collector_kwargs or {}),
    )
    generator = torch.Generator().manual_seed(7)
    all_stats = []
    for iteration in range(steps):
        tokens = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator).cuda()
        position_ids = torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)
        mask = torch.triu(torch.ones(1, 1, _SEQ, _SEQ, dtype=torch.bool, device="cuda"), 1)
        model.zero_grad_buffer()
        optimizer.zero_grad()
        collector.begin_step(iteration)
        losses = model(tokens, position_ids, mask, labels=tokens.roll(-1, 1))
        losses.float().mean().backward()
        finalize_model_grads([model])
        collector.before_optimizer_step()
        assert optimizer.step()[0]
        all_stats.append(collector.after_optimizer_step(True))
    collector.remove_hooks()
    return all_stats


def _full_state(moe, bf16):
    model, _ = _build_gpt(1, 1, moe, bf16)
    full = {name: param.detach().clone() for name, param in model.named_parameters()}
    Utils.destroy_model_parallel()
    return full


def _assert_stats_close(reference, stats, rtol):
    assert reference.keys() == stats.keys(), sorted(set(reference) ^ set(stats))
    for key, value in reference.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(stats[key], value, rtol=rtol, atol=1e-6, msg=key)
        else:
            assert stats[key] == pytest.approx(value, rel=rtol, abs=1e-6), (key, stats[key], value)


_LAYOUTS = [(1, 1, True), (2, 1, False), (2, 1, True), (1, 2, True), (2, 2, True), (1, 4, True)]


@pytest.mark.skipif(WORLD_SIZE < 4, reason="needs 4 ranks")
@pytest.mark.parametrize("moe", [False, True])
def test_layout_invariance_fp32_adam(moe):
    """Same init and data at TP1 and TP2 (+SP), EP1/2/4, replicated vs layer-wise ownership.

    Adam is elementwise, so the optimizer itself is layout-invariant and every metric must
    match up to fp32 reduction-order noise of the forward/backward (rtol 1e-4).
    """
    full = _full_state(moe, False)
    results = {}
    for tp, ep, layer_wise in [(1, 1, False)] + _LAYOUTS:
        if not moe and ep > 1:
            continue
        model, config = _build_gpt(tp, ep, moe, False, full)
        model = _ddp(model, config, layer_wise)
        optimizer = _adam(model, False, layer_wise)
        results[(tp, ep, layer_wise)] = _run_gpt(model, optimizer, config)
        Utils.destroy_model_parallel()
    reference = results.pop((1, 1, False))
    families = {key.split("/")[2] for key in reference[0] if key.startswith("update/relative/")}
    assert {"attention-in-adam", "attention-out-adam", "embedding-adam", "output-adam", "layernorm-adam"} <= families
    assert ("expert-in-adam" in families) == moe
    assert "update/delta-y/attention-in-adam" in reference[0]
    if moe:
        assert "update/experts/expert-in/relative/cv" in reference[0]
        assert "update/layers/1/experts/expert-out/grad-momentum-cos/argmax" in reference[1]
    for layout, stats in results.items():
        for step, (expected, actual) in enumerate(zip(reference, stats)):
            try:
                _assert_stats_close(expected, actual, rtol=1e-4)
            except AssertionError as error:
                raise AssertionError(f"layout {layout} step {step}: {error}") from error


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs 2 ranks")
def test_layer_wise_bf16_matches_replicated():
    """bf16 params with fp32 main params: layer-wise ownership vs DP-rank-0 ownership."""
    full = _full_state(False, True)
    results = []
    for layer_wise in (False, True):
        model, config = _build_gpt(1, 1, False, True, full)
        model = _ddp(model, config, layer_wise)
        if layer_wise:
            optimizer = _adam(model, True, True)
        else:
            optimizer_config = OptimizerConfig(
                optimizer="adam", lr=1e-2, weight_decay=0.1, bf16=True, use_distributed_optimizer=False, clip_grad=0.0
            )
            optimizer = get_megatron_optimizer(optimizer_config, [model])
        results.append(_run_gpt(model, optimizer, config))
        Utils.destroy_model_parallel()
    for expected, actual in zip(*results):
        _assert_stats_close(expected, actual, rtol=1e-5)


@pytest.mark.skipif(WORLD_SIZE < 4, reason="needs 4 ranks")
def test_neutrino_moe_layer_wise_smoke():
    """The production path: Neutrino + chained AdamW, layer-wise, TP2 + SP, EP2, bf16."""
    full = _full_state(True, True)
    model, config = _build_gpt(2, 2, True, True, full)
    model = _ddp(model, config, True)
    optimizer_config = OptimizerConfig(
        optimizer="neutrino",
        lr=1e-3,
        weight_decay=0.1,
        bf16=True,
        use_distributed_optimizer=False,
        clip_grad=1.0,
        muon_momentum=0.99,
        muon_use_nesterov=True,
        muon_scale_mode="shape_up_rank",
        neutrino_k=8,
        neutrino_no_error_feedback=True,
        neutrino_metrics_interval=0,
    )
    optimizer = get_megatron_neutrino_optimizer(
        optimizer_config, [model], layer_wise_distributed_optimizer=True, pg_collection=_pg_collection()
    )
    stats = _run_gpt(model, optimizer, config, steps=3)[-1]
    for family in ("attention-in", "attention-out", "expert-in", "expert-out", "router", "embedding-adam"):
        assert 0 < stats[f"update/relative/{family}"] < 1, family
        assert -1 <= stats[f"update/grad-momentum-cos/{family}"] <= 1, family
        assert stats[f"update/stable-rank/{family}"] >= 1 - 1e-6, family
    assert stats["update/momentum-r-eff/expert-in"] >= 1 - 1e-6
    assert isinstance(stats["update/experts/expert-in/relative/hist"], torch.Tensor)
    assert stats["update/experts/expert-in/relative/hist"].numel() == 2 * 4
    assert all(math.isfinite(v) for v in stats.values() if not isinstance(v, torch.Tensor))
