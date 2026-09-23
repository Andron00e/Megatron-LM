# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Next Implicit Token Prediction (NITP), arXiv 2605.24956.

An auxiliary objective: the final hidden state at position t, passed through a small head,
predicts the stop-gradient output of a shallow layer at position t + shift (same forward pass).
The loss is attached to the main hidden states through an autograd function so it needs no
change to the training loop (the pattern used by MTP), and the head is unused at inference.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class NITPHead(MegatronModule):
    """Projection head on the final hidden state. Replicated across TP (its input and loss are
    identical on every TP rank), so it needs no parallel linear."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        h = config.hidden_size
        if config.nitp_head == 'mlp':
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(h, h, bias=False), torch.nn.GELU(), torch.nn.Linear(h, h, bias=False)
            )
        else:
            self.proj = torch.nn.Linear(h, h, bias=False)
        for p in self.proj.parameters():
            config.init_method(p)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)


class NITPLossLoggingHelper:
    """Accumulates the NITP loss across micro-batches for logging (mirrors MTPLossLoggingHelper)."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(loss: Tensor, reduce_group=None, avg_group=None):
        tracker = NITPLossLoggingHelper.tracker
        if 'values' not in tracker:
            tracker['values'] = torch.zeros(1, device=loss.device)
        tracker['values'][0] += loss.detach()
        tracker['reduce_group'] = reduce_group
        tracker['avg_group'] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        tracker = NITPLossLoggingHelper.tracker
        if 'values' in tracker:
            tracker['values'].zero_()
        tracker['reduce_group'] = None
        tracker['avg_group'] = None

    @staticmethod
    def reduce_loss_in_tracker():
        tracker = NITPLossLoggingHelper.tracker
        if 'values' not in tracker:
            return
        values = tracker['values']
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker['reduce_group'])
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    @staticmethod
    def track_nitp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        NITPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = NITPLossLoggingHelper.tracker
        if 'values' not in tracker:
            return
        loss = tracker['values'][0] * loss_scale
        name = 'nitp loss'
        if total_loss_dict is not None:
            if name in total_loss_dict:
                total_loss_dict[name] += loss
            else:
                total_loss_dict[name] = loss
        if writer is not None:
            writer.add_scalar(name, loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({name: loss}, iteration)
        NITPLossLoggingHelper.clean_loss_in_tracker()


class NITPLossAutoScaler(torch.autograd.Function):
    """Identity on `output`; in backward it emits grad = main_loss_backward_scale for the
    attached per-token loss tensor, so the NITP loss joins the main loss's backward pass with the
    same scaling (grad scaler / num_microbatches). Same design as MTPLossAutoScaler."""

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: Tensor, nitp_loss: Tensor) -> Tensor:
        ctx.save_for_backward(nitp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        (nitp_loss,) = ctx.saved_tensors
        scale = NITPLossAutoScaler.main_loss_backward_scale
        if scale is None:
            scale = torch.ones(1, device=nitp_loss.device)
        return grad_output, torch.ones_like(nitp_loss) * scale

    @staticmethod
    def set_loss_scale(scale: Tensor) -> None:
        NITPLossAutoScaler.main_loss_backward_scale = scale


def nitp_per_token_loss(pred: Tensor, target: Tensor, loss_type: str) -> Tensor:
    """[s, b, h] x [s, b, h] -> [s, b] loss, computed in fp32."""
    pred = pred.float()
    target = target.float()
    if loss_type == 'cosine':
        return 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-6)
    return ((pred - target) ** 2).mean(dim=-1)


def process_nitp_loss(
    hidden_states: Tensor,
    target_hidden: Tensor,
    head: NITPHead,
    loss_mask: Optional[Tensor],
    config: TransformerConfig,
    is_training: bool,
    tp_group=None,
) -> Tensor:
    """Attach the NITP loss to `hidden_states` ([s, b, h], the decoder output on the post-process
    rank) and return them unchanged for the LM head.

    target_hidden: [s, b, h] detached output of the shallow target layer (same sharding as
    hidden_states). loss_mask: [b, s] or None (all ones)."""
    shift = config.nitp_shift
    if config.sequence_parallel:
        # Sequence-parallel shards split the sequence across TP ranks; the t+shift target can live
        # on the next rank. Gather both to the full sequence; the loss is then identical on every
        # TP rank, so the backward of the gather must split (not reduce-scatter) the gradient.
        hidden_states_full = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp_group
        )
        target_full = gather_from_sequence_parallel_region(
            target_hidden, tensor_parallel_output_grad=False, group=tp_group
        )
    else:
        hidden_states_full, target_full = hidden_states, target_hidden

    s = hidden_states_full.shape[0]
    if s <= shift:
        return hidden_states
    pred = head(hidden_states_full[:-shift])
    target = target_full[shift:].detach()
    loss = nitp_per_token_loss(pred, target, config.nitp_loss_type)  # [s-shift, b]

    if loss_mask is None:
        mask = torch.ones_like(loss)
        original_num_tokens = mask.numel()
    else:
        # Valid iff the target position is a real (unmasked) token.
        original_num_tokens = loss_mask.sum()
        mask = loss_mask.transpose(0, 1)[shift:].to(loss.dtype)
    loss = loss * mask
    num_tokens = mask.sum()
    safe_num_tokens = num_tokens.clamp(min=1)

    if is_training:
        NITPLossLoggingHelper.save_loss_to_tracker(
            loss.sum() / safe_num_tokens,
            avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    coeff = config.nitp_loss_coeff
    if config.calculate_per_token_loss:
        # finalize_model_grads divides every gradient by the main loss's token count; rescale so
        # the NITP gradient is a true per-token average over its (shorter) valid range.
        scaled = coeff * loss * (original_num_tokens / safe_num_tokens)
    else:
        scaled = coeff * loss / safe_num_tokens
    return NITPLossAutoScaler.apply(hidden_states, scaled)
