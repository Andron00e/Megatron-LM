# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Next Implicit Token Prediction (NITP, arXiv 2605.24956) and its horizon extension (latent MTP).

Depth 1 (NITP): the final hidden state at position t, through a small head P, predicts the
stop-gradient output of a shallow layer at position t+1 from the same forward pass (cosine loss).

Depth j > 1 (latent MTP): the chain continues DeepSeek-MTP style, teacher-forced with the true
token x_{t+j-1}:  z^_{t+j} = P(M[RMSNorm(z^_{t+j-1}); RMSNorm(Emb(x_{t+j-1}))]), and predicts the
shallow state at t+j. An optional token read-out OutHead(U z^_{t+j}) -> x_{t+j+1} turns the chain
into a drafter for speculative decoding. All heads are dropped at inference in quality mode.

The losses join the main backward through an autograd function (MTP's pattern), so the training
loop is unchanged; per-depth losses are logged as 'nitp loss', 'nitp_j loss', 'nitp_tok_j loss'.
"""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_token_prediction import roll_tensor
from megatron.core.transformer.transformer_config import TransformerConfig


def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class NITPHead(MegatronModule):
    """Projection head P: hidden -> hidden. Replicated across TP (its input and loss are identical
    on every TP rank after the sequence gather), so it needs no parallel linear."""

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


class NITPChain(MegatronModule):
    """P (per depth or tied), the 2h->h merge M used from depth 2 on, and the read-out U."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        h = config.hidden_size
        k = config.nitp_horizon
        n_heads = 1 if (config.nitp_tie_chain or k == 1) else k
        self.heads = torch.nn.ModuleList([NITPHead(config) for _ in range(n_heads)])
        self.head = self.heads[0]  # depth-1 head; kept as an attribute for tests/checkpoint names
        if k > 1:
            n_merge = 1 if config.nitp_tie_chain else k - 1
            self.merges = torch.nn.ModuleList(
                [torch.nn.Linear(2 * h, h, bias=False) for _ in range(n_merge)]
            )
            for m in self.merges:
                config.init_method(m.weight)
        if config.nitp_token_readout_coeff > 0:
            self.readout = torch.nn.Linear(h, h, bias=False)
            config.init_method(self.readout.weight)

    def _head(self, j: int) -> NITPHead:
        return self.heads[0] if len(self.heads) == 1 else self.heads[j - 1]

    def _merge(self, j: int) -> torch.nn.Linear:
        return self.merges[0] if len(self.merges) == 1 else self.merges[j - 2]

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor],
        embedding: Optional[Callable],
        cp_group=None,
        packed_seq_params=None,
    ) -> List[Tensor]:
        """Returns [z^_{t+1}, ..., z^_{t+k}] in the same (possibly sequence-sharded) layout as
        hidden_states. Depth j>1 is teacher-forced with Emb(x_{t+j-1}); the embedding module is the
        GPT model's, so under sequence parallel it returns the same shard as hidden_states."""
        preds = [self._head(1)(hidden_states)]
        ids, pos = input_ids, position_ids
        for j in range(2, self.config.nitp_horizon + 1):
            ids, _ = roll_tensor(ids, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params)
            pos, _ = roll_tensor(pos, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params)
            emb = embedding(input_ids=ids, position_ids=pos)  # [s, b, h] (sharded under SP)
            x = torch.cat([_rms_norm(preds[-1]), _rms_norm(emb.to(preds[-1].dtype))], dim=-1)
            preds.append(self._head(j)(self._merge(j)(x)))
        return preds


class NITPLossLoggingHelper:
    """Accumulates per-depth NITP losses across micro-batches (mirrors MTPLossLoggingHelper).
    Slots [0, k) = representation losses per depth, [k, 2k) = token read-out losses per depth."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(loss: Tensor, slot: int, num_slots: int, avg_group=None):
        tracker = NITPLossLoggingHelper.tracker
        if 'values' not in tracker or tracker['values'].numel() != num_slots:
            tracker['values'] = torch.zeros(num_slots, device=loss.device)
        tracker['values'][slot] += loss.detach()
        tracker['avg_group'] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        tracker = NITPLossLoggingHelper.tracker
        if 'values' in tracker:
            tracker['values'].zero_()
        tracker['avg_group'] = None

    @staticmethod
    def reduce_loss_in_tracker():
        tracker = NITPLossLoggingHelper.tracker
        if 'values' in tracker and tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                tracker['values'], group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    @staticmethod
    def track_nitp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        NITPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = NITPLossLoggingHelper.tracker
        if 'values' not in tracker:
            return
        values = tracker['values'] * loss_scale
        k = values.numel() // 2
        for slot in range(values.numel()):
            j = slot % k + 1
            name = 'nitp loss' if slot == 0 else (f'nitp_{j} loss' if slot < k else f'nitp_tok_{j} loss')
            if slot >= k and values[slot] == 0:
                continue  # read-out disabled
            loss = values[slot]
            if total_loss_dict is not None:
                total_loss_dict[name] = total_loss_dict.get(name, 0) + loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({name: loss}, iteration)
        NITPLossLoggingHelper.clean_loss_in_tracker()


class NITPLossAutoScaler(torch.autograd.Function):
    """Identity on `output`; in backward it emits grad = main_loss_backward_scale for the attached
    per-token loss tensor, so the NITP losses join the main loss's backward pass with the same
    scaling (grad scaler / num_microbatches). Same design as MTPLossAutoScaler."""

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


def _attach(hidden_states: Tensor, loss: Tensor, coeff: float, num_valid: Tensor,
            original_num_tokens, config: TransformerConfig) -> Tensor:
    """Scale a masked per-token loss tensor the way MTP does and attach it to hidden_states."""
    safe = num_valid.clamp(min=1)
    if config.calculate_per_token_loss:
        scaled = coeff * loss * (original_num_tokens / safe)
    else:
        scaled = coeff * loss / safe
    return NITPLossAutoScaler.apply(hidden_states, scaled)


def process_nitp_loss(
    hidden_states: Tensor,
    target_hidden: Tensor,
    chain: NITPChain,
    loss_mask: Optional[Tensor],
    config: TransformerConfig,
    is_training: bool,
    tp_group=None,
    input_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    embedding: Optional[Callable] = None,
    labels: Optional[Tensor] = None,
    output_layer: Optional[Callable] = None,
    output_weight: Optional[Tensor] = None,
    runtime_gather_output: Optional[bool] = None,
    compute_language_model_loss: Optional[Callable] = None,
    cp_group=None,
    packed_seq_params=None,
) -> Tensor:
    """Attach the NITP / latent-MTP losses to `hidden_states` ([s, b, h], decoder output on the
    post-process rank) and return them unchanged for the LM head.

    target_hidden: [s, b, h] detached output of the shallow target layer (same sharding).
    loss_mask / labels: [b, s] or None."""
    k = config.nitp_horizon
    shift0 = config.nitp_shift
    preds = chain(hidden_states, input_ids, position_ids, embedding, cp_group, packed_seq_params)

    if config.sequence_parallel:
        # The t+j target can live on the next TP rank's sequence shard: gather to the full
        # sequence. The loss is then identical on every TP rank, so the gather's backward must
        # split (not reduce-scatter) the gradient.
        gather = lambda t: gather_from_sequence_parallel_region(
            t, tensor_parallel_output_grad=False, group=tp_group
        )
        preds_full = [gather(p) for p in preds]
        target_full = gather(target_hidden)
    else:
        preds_full, target_full = preds, target_hidden

    s = target_full.shape[0]
    avg_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    num_slots = 2 * k
    if loss_mask is None:
        base_mask = torch.ones(s, target_full.shape[1], device=target_full.device)
        original_num_tokens = base_mask.numel()
    else:
        base_mask = loss_mask.transpose(0, 1).float()  # [s, b]
        original_num_tokens = loss_mask.sum()

    for j in range(1, k + 1):
        shift = shift0 + (j - 1)
        if s <= shift:
            break
        pred = preds_full[j - 1][:-shift]
        target = target_full[shift:].detach()
        loss = nitp_per_token_loss(pred, target, config.nitp_loss_type) * base_mask[shift:]
        num_valid = base_mask[shift:].sum()
        if is_training:
            NITPLossLoggingHelper.save_loss_to_tracker(
                loss.sum() / num_valid.clamp(min=1), j - 1, num_slots, avg_group=avg_group
            )
        coeff = config.nitp_loss_coeff * (config.nitp_horizon_decay ** (j - 1))
        hidden_states = _attach(hidden_states, loss, coeff, num_valid, original_num_tokens, config)

    if config.nitp_token_readout_coeff > 0 and labels is not None:
        # z^_{t+j} reads out x_{t+j+1}: labels[t] = x_{t+1}, rolled j times (MTP's cumulative roll).
        tok_labels, tok_mask = labels, (loss_mask if loss_mask is not None else torch.ones_like(labels))
        for j in range(1, k + 1):
            tok_labels, _ = roll_tensor(tok_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params)
            tok_mask, num_tok = roll_tensor(tok_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params)
            logits, _ = output_layer(
                chain.readout(preds[j - 1]), weight=output_weight, runtime_gather_output=runtime_gather_output
            )
            ce = compute_language_model_loss(tok_labels, logits) * tok_mask  # [b, s]
            if is_training:
                NITPLossLoggingHelper.save_loss_to_tracker(
                    ce.sum() / num_tok.clamp(min=1), k + j - 1, num_slots, avg_group=avg_group
                )
            hidden_states = _attach(
                hidden_states, ce, config.nitp_token_readout_coeff, num_tok, original_num_tokens, config
            )
    return hidden_states
