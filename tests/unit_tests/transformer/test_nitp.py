# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Next Implicit Token Prediction (NITP) unit tests. Run inside the training container:
    torchrun --nproc-per-node=1 -m pytest tests/unit_tests/transformer/test_nitp.py -x -q
    torchrun --nproc-per-node=2 -m pytest tests/unit_tests/transformer/test_nitp.py -x -q -k tp2
"""

import os
import sys

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.nitp import (
    NITPLossAutoScaler,
    NITPLossLoggingHelper,
    nitp_per_token_loss,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, get_args, set_args, set_global_variables
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import unwrap_model
from tests.unit_tests.test_utilities import Utils

_SEED = 42


class TestNITP:
    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        NITPLossLoggingHelper.tracker = {}
        NITPLossAutoScaler.main_loss_backward_scale = None

    def create_test_args(self, tp, coeff=1.0, head='linear', shift=1, num_layers=4,
                         horizon=1, readout=0.0, tie=True):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        sys.argv = ['test_nitp.py']
        args = parse_args()
        args.num_layers = num_layers
        args.nitp_loss_coeff = coeff
        args.nitp_target_layer_frac = 0.25
        args.nitp_head = head
        args.nitp_shift = shift
        args.nitp_horizon = horizon
        args.nitp_token_readout_coeff = readout
        args.nitp_tie_chain = tie
        args.vocab_size = 1024
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = tp > 1
        args.position_embedding_type = 'rope'
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        args.bf16 = True
        args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True
        validate_args(args)
        set_global_variables(args, False)
        return args

    def model_provider(self, pre_process=True, post_process=True, config=None, **kwargs):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        if config is None:
            config = core_transformer_config_from_args(args)
        spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            **{k: v for k, v in kwargs.items() if k == 'pg_collection'},
        )

    def get_batch(self):
        s, b = self.seq_length, self.micro_batch_size
        g = torch.Generator(device='cpu').manual_seed(_SEED)
        tokens = torch.randint(0, 1024, (b, s), generator=g).cuda()
        labels = torch.roll(tokens, -1, dims=1)
        position_ids = torch.arange(s).repeat((b, 1)).cuda()
        attention_mask = torch.ones((b, 1, s, s), dtype=bool).cuda()
        loss_mask = torch.ones((b, s)).cuda()
        loss_mask[:, -3:] = 0  # a few padded positions at the end
        return tokens, labels, loss_mask, attention_mask, position_ids

    def _run(self, tp, **kw):
        args = self.create_test_args(tp, **kw)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch()
        models, _, _ = setup_model_and_optimizer(self.model_provider, ModelType.encoder_or_decoder)
        model = unwrap_model(models[0])
        NITPLossAutoScaler.set_loss_scale(torch.ones(1, device='cuda'))
        out = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        return args, model, out, loss_mask

    def test_config_resolves_target_layer(self):
        args = self.create_test_args(tp=1, num_layers=8)
        config = core_transformer_config_from_args(args)
        assert config.nitp_target_layer == 2  # round(0.25 * 8)

    def test_loss_logged_and_finite(self):
        args, model, out, loss_mask = self._run(tp=1)
        tracker = NITPLossLoggingHelper.tracker
        assert 'values' in tracker
        v = tracker['values'][0].item()
        assert torch.isfinite(torch.tensor(v)) and 0.0 <= v <= 2.0, v
        # the shallow capture has the decoder's shape and carries no graph
        t = model.decoder.nitp_target_hidden
        assert t.shape == (self.seq_length, self.micro_batch_size, args.hidden_size)
        assert not t.requires_grad

    def test_loss_matches_manual_recompute(self):
        args, model, out, loss_mask = self._run(tp=1)
        tracker_val = NITPLossLoggingHelper.tracker['values'][0].item()
        # Recompute from the captured target and the head applied to the final hidden state.
        # The final hidden state is not stored, so rerun the decoder path with hooks.
        captured = {}
        h = model.decoder.register_forward_hook(lambda m, i, o: captured.__setitem__('final', o))
        NITPLossLoggingHelper.clean_loss_in_tracker()
        tokens, labels, lm, attention_mask, position_ids = self.get_batch()
        model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask,
              labels=labels, loss_mask=lm)
        h.remove()
        final = captured['final']
        target = model.decoder.nitp_target_hidden
        shift = args.nitp_shift
        pred = model.nitp_head(final[:-shift])
        per_tok = nitp_per_token_loss(pred, target[shift:], 'cosine')
        mask = lm.transpose(0, 1)[shift:]
        manual = (per_tok * mask).sum() / mask.sum()
        assert abs(manual.item() - tracker_val) < 1e-4, (manual.item(), tracker_val)

    def test_backward_reaches_head_and_scales_with_coeff(self):
        grads = {}
        for coeff in (0.5, 1.0):
            self.teardown_method(None)
            args, model, out, loss_mask = self._run(tp=1, coeff=coeff)
            (out.view(-1) * loss_mask.view(-1)).sum().backward()
            w = model.nitp_head.proj.weight
            g = w.main_grad if hasattr(w, 'main_grad') and w.main_grad is not None else w.grad
            assert g is not None and torch.isfinite(g).all()
            grads[coeff] = g.float().clone()
        # head grads come only from the NITP term, so they must scale linearly with the coefficient
        ratio = grads[1.0].norm() / grads[0.5].norm()
        assert abs(ratio.item() - 2.0) < 0.05, ratio.item()

    def test_disabled_has_no_head(self):
        args, model, out, loss_mask = self._run(tp=1, coeff=0.0)
        assert not hasattr(model, 'nitp_head')
        assert 'values' not in NITPLossLoggingHelper.tracker

    def test_horizon2_readout_logged_and_trainable(self):
        args, model, out, loss_mask = self._run(tp=1, horizon=2, readout=0.1)
        v = NITPLossLoggingHelper.tracker['values']
        assert v.numel() == 4  # 2 representation depths + 2 token read-outs
        assert torch.isfinite(v).all() and (v > 0).all(), v
        assert 0.0 <= v[0].item() <= 2.0 and 0.0 <= v[1].item() <= 2.0
        assert v[2].item() > 1.0 and v[3].item() > 1.0  # CE over a 1024 vocab at init
        (out.view(-1) * loss_mask.view(-1)).sum().backward()
        chain = model.nitp_chain
        assert len(chain.heads) == 1 and len(chain.merges) == 1  # tied
        for name, w in [('merge', chain.merges[0].weight), ('readout', chain.readout.weight)]:
            g = w.main_grad if getattr(w, 'main_grad', None) is not None else w.grad
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name

    def test_horizon2_depth2_matches_manual_recompute(self):
        args, model, out, loss_mask = self._run(tp=1, horizon=2, readout=0.0)
        v2 = NITPLossLoggingHelper.tracker['values'][1].item()
        captured = {}
        h = model.decoder.register_forward_hook(lambda m, i, o: captured.__setitem__('final', o))
        NITPLossLoggingHelper.clean_loss_in_tracker()
        tokens, labels, lm, attention_mask, position_ids = self.get_batch()
        model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask,
              labels=labels, loss_mask=lm)
        h.remove()
        final, target = captured['final'], model.decoder.nitp_target_hidden
        preds = model.nitp_chain(final, tokens, position_ids, model.embedding)
        shift = args.nitp_shift + 1
        per_tok = nitp_per_token_loss(preds[1][:-shift], target[shift:], 'cosine')
        mask = lm.transpose(0, 1)[shift:]
        manual = (per_tok * mask).sum() / mask.sum()
        assert abs(manual.item() - v2) < 1e-4, (manual.item(), v2)

    def test_untied_chain_has_separate_heads(self):
        args, model, out, loss_mask = self._run(tp=1, horizon=3, tie=False)
        chain = model.nitp_chain
        assert len(chain.heads) == 3 and len(chain.merges) == 2
        assert NITPLossLoggingHelper.tracker['values'].numel() == 6

    @pytest.mark.skipif(int(os.environ.get('WORLD_SIZE', '1')) < 2, reason='needs a 2-rank launch')
    def test_tp2_horizon2_matches_tp1(self):
        _ = self._run(tp=1, horizon=2, readout=0.1)
        ref = NITPLossLoggingHelper.tracker['values'].clone()
        self.teardown_method(None)
        _ = self._run(tp=2, horizon=2, readout=0.1)
        val = NITPLossLoggingHelper.tracker['values']
        assert (val[:2] - ref[:2]).abs().max().item() < 0.2, (val, ref)

    @pytest.mark.skipif(int(os.environ.get('WORLD_SIZE', '1')) < 2, reason='needs a 2-rank launch')
    def test_tp2_sequence_parallel_matches_tp1(self):
        """The loss is a per-token mean over the full sequence, so it must not depend on TP/SP."""
        _, _, _, _ = self._run(tp=1)
        ref = NITPLossLoggingHelper.tracker['values'][0].item()
        self.teardown_method(None)
        _, model, _, _ = self._run(tp=2)
        val = NITPLossLoggingHelper.tracker['values'][0].item()
        # different TP init and bf16 -> loose tolerance; the sequence-split bug would show as a
        # large systematic deviation (missing shard boundaries), not a small one
        assert abs(val - ref) < 0.2, (val, ref)
