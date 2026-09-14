# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _update_router_qb_beta,
    reset_model_temporary_tensors,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _router_qb_model(config, **buffers):
    model = torch.nn.Module()
    model.config = config
    model.ddp_config = DistributedDataParallelConfig()
    model.router = torch.nn.Module()
    for name, tensor in buffers.items():
        model.router.register_buffer(name, tensor.clone())
    return model


def _router_qb_config(ema=0.25, method="average", num_bins=1000, freeze=False):
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_ema=ema,
        moe_router_quantile_balancing_freeze=freeze,
        moe_router_quantile_balancing_method=method,
        moe_router_quantile_balancing_num_bins=num_bins,
    )


class TestUpdateRouterQBBeta:
    def test_frozen_qb_preserves_beta_without_collectives(self, monkeypatch):
        config = _router_qb_config(freeze=True)
        model = _router_qb_model(
            config,
            qb_beta=torch.tensor([1.0, -1.0, 0.0]),
            qb_beta_accum=torch.tensor([4.0, 2.0, 0.0]),
            qb_beta_count=torch.tensor(2, dtype=torch.long),
        )
        before = model.router.qb_beta.clone()

        def fail_all_reduce(*args, **kwargs):
            raise AssertionError("all_reduce should not run when QB is frozen")

        monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, before)

    def test_update_router_qb_beta_ema_centers_and_resets(self, monkeypatch):
        config = _router_qb_config(ema=0.25)
        model = _router_qb_model(
            config,
            qb_beta=torch.tensor([1.0, -1.0, 0.0]),
            qb_beta_accum=torch.tensor([4.0, 2.0, 0.0]),
            qb_beta_count=torch.tensor(2, dtype=torch.long),
        )
        dp_cp_group = object()
        all_reduce_calls = []
        waits = []

        class FakeWork:
            def __init__(self, index):
                self.index = index

            def wait(self):
                waits.append(self.index)

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            all_reduce_calls.append(
                {"value": tensor.clone(), "op": op, "group": group, "async_op": async_op}
            )
            return FakeWork(len(all_reduce_calls) - 1)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=dp_cp_group)

        local_avg = torch.tensor([2.0, 1.0, 0.0])
        expected = 0.25 * torch.tensor([1.0, -1.0, 0.0]) + 0.75 * local_avg
        expected = expected - expected.mean()
        torch.testing.assert_close(all_reduce_calls[0]["value"], torch.tensor([[4.0, 2.0, 0.0]]))
        torch.testing.assert_close(all_reduce_calls[1]["value"], torch.tensor([2]))
        assert all_reduce_calls[0]["op"] == torch.distributed.ReduceOp.SUM
        assert all_reduce_calls[1]["op"] == torch.distributed.ReduceOp.SUM
        assert all_reduce_calls[0]["group"] is dp_cp_group
        assert all_reduce_calls[1]["group"] is dp_cp_group
        assert all_reduce_calls[0]["async_op"] is True
        assert all_reduce_calls[1]["async_op"] is True
        assert waits == [0, 1]
        torch.testing.assert_close(model.router.qb_beta, expected)
        torch.testing.assert_close(model.router.qb_beta.mean(), torch.zeros(()))

        reset_model_temporary_tensors(config, [model])
        torch.testing.assert_close(
            model.router.qb_beta_accum, torch.zeros_like(model.router.qb_beta_accum)
        )
        assert model.router.qb_beta_count.item() == 0

    def test_update_router_qb_beta_skips_eval_modules(self, monkeypatch):
        config = _router_qb_config(ema=0.0)
        model = _router_qb_model(
            config,
            qb_beta=torch.tensor([1.0, -1.0, 0.0]),
            qb_beta_accum=torch.tensor([4.0, 2.0, 0.0]),
            qb_beta_count=torch.tensor(1, dtype=torch.long),
        )
        model.router.eval()
        before = model.router.qb_beta.clone()

        def fail_all_reduce(*args, **kwargs):
            raise AssertionError("all_reduce should not run for eval QB routers")

        monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, before)

    def test_update_router_qb_beta_preserves_beta_without_observations(self, monkeypatch):
        config = _router_qb_config(ema=0.0)
        model = _router_qb_model(
            config,
            qb_beta=torch.tensor([2.0, -1.0, -1.0]),
            qb_beta_accum=torch.tensor([0.0, 0.0, 0.0]),
            qb_beta_count=torch.tensor(0, dtype=torch.long),
        )

        class FakeWork:
            def wait(self):
                pass

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            del tensor, op, group, async_op
            return FakeWork()

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, torch.tensor([2.0, -1.0, -1.0]))

    def test_histogram_update_reduces_once_decodes_centers_and_resets(self, monkeypatch):
        config = _router_qb_config(ema=0.25, method="histogram", num_bins=4)
        histogram = torch.tensor([[1, 3, 0, 0], [0, 2, 2, 0], [0, 0, 4, 0]], dtype=torch.int64)
        model = _router_qb_model(
            config, qb_beta=torch.tensor([0.2, 0.0, -0.2]), qb_histogram=histogram
        )
        tp_dp_cp_group = object()
        all_reduce_calls = []

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            all_reduce_calls.append(
                {"value": tensor.clone(), "op": op, "group": group, "async_op": async_op}
            )

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object(), tp_dp_cp_group=tp_dp_cp_group)

        expected = torch.tensor([0.3375, -0.0875, -0.25])
        assert len(all_reduce_calls) == 1
        assert torch.equal(all_reduce_calls[0]["value"], histogram.unsqueeze(0))
        assert all_reduce_calls[0]["op"] == torch.distributed.ReduceOp.SUM
        assert all_reduce_calls[0]["group"] is tp_dp_cp_group
        assert all_reduce_calls[0]["async_op"] is False
        torch.testing.assert_close(model.router.qb_beta, expected)

        reset_model_temporary_tensors(config, [model])
        assert torch.equal(model.router.qb_histogram, torch.zeros_like(histogram))

    def test_histogram_update_preserves_beta_without_observations(self, monkeypatch):
        config = _router_qb_config(ema=0.0, method="histogram", num_bins=4)
        model = _router_qb_model(
            config,
            qb_beta=torch.tensor([2.0, -1.0, -1.0]),
            qb_histogram=torch.zeros((3, 4), dtype=torch.int64),
        )

        monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

        _update_router_qb_beta([model], config, dp_cp_group=object(), tp_dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, torch.tensor([2.0, -1.0, -1.0]))


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=True,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)
