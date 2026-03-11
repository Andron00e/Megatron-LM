# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Main logger class for model internals logging to W&B."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .config import InternalsLoggingConfig
from .hooks import InternalsHookManager
from .state_manager import InternalsStateManager
from .metrics import (
    compute_activation_stats,
    compute_gradient_norm,
    compute_gradient_weight_alignment,
    compute_grad_shard_norm_sq,
    compute_grad_weight_dot_shard,
    get_param_grad,
    _extract_layer_index,
    _rerun_linear,
    _split_qkv_dim,
    _mean_squared_norm_change,
    _QKV_NAMES,
)
from .hooks import _LINEAR_LAYER_PATHS

from megatron.training.utils import prettify_metric_keys


def _build_model_info(model: nn.Module):
    """Walk TransformerLayers once to extract all static model structure info.

    Builds two data structures that are invariant across training steps:
    - split_info: identifies fused weight matrices and their split config
    - linear_entries: maps (linear_attr, layer_number) to module references

    Returns:
        split_info: Dict[int, dict] — id(param) → {type, ...split config}
        linear_entries: Dict[Tuple[str, int], Tuple[nn.Module, nn.Module]]
            — (linear_attr, layer_number) → (linear_module, transformer_layer)
    """
    from megatron.core.transformer.transformer_layer import TransformerLayer

    split_info: Dict[int, dict] = {}
    linear_entries: Dict[Tuple[str, int], Tuple[nn.Module, nn.Module]] = {}

    for _, module in model.named_modules():
        if not isinstance(module, TransformerLayer):
            continue

        layer_number = module.layer_number

        for parent_attr, linear_attr in _LINEAR_LAYER_PATHS:
            parent = getattr(module, parent_attr, None)
            if parent is None:
                continue
            linear_module = getattr(parent, linear_attr, None)
            if linear_module is None:
                continue
            linear_entries[(linear_attr, layer_number)] = (linear_module, module)

        attn = getattr(module, 'self_attention', None)
        if attn is not None:
            qkv = getattr(attn, 'linear_qkv', None)
            if qkv is not None and hasattr(qkv, 'weight'):
                split_info[id(qkv.weight)] = {
                    'type': 'qkv',
                    'num_query_groups': attn.num_query_groups_per_partition,
                    'num_query_heads_per_group': (
                        attn.num_attention_heads_per_partition
                        // attn.num_query_groups_per_partition
                    ),
                    'head_dim': attn.hidden_size_per_attention_head,
                }

        if getattr(module.config, 'gated_linear_unit', False):
            mlp = getattr(module, 'mlp', None)
            if mlp is not None:
                fc1 = getattr(mlp, 'linear_fc1', None)
                if fc1 is not None and hasattr(fc1, 'weight'):
                    split_info[id(fc1.weight)] = {'type': 'swiglu'}

    return split_info, linear_entries


class InternalsLogger:
    """Main logger class that orchestrates model internals logging to W&B.

    This class coordinates the hook manager, state manager, and metric computation
    to log comprehensive model internals to Weights & Biases.

    In distributed settings, all DP ranks participate in gradient metric computation
    (via all-reduce of per-shard statistics), but only the logging rank writes to W&B.

    Attributes:
        config: Configuration for internals logging.
        hook_manager: Manages forward hooks for capturing activations/attention.
        state_manager: Manages state for delta computations.
        is_logging_rank: Whether this rank should write to W&B.
    """

    def __init__(
        self,
        config: InternalsLoggingConfig,
        hook_manager: InternalsHookManager,
        state_manager: InternalsStateManager,
        is_logging_rank: bool = True,
    ):
        """Initialize the internals logger.

        Args:
            config: Configuration specifying what to log.
            hook_manager: Hook manager for capturing forward pass data.
            state_manager: State manager for delta computations.
            is_logging_rank: Whether this rank is responsible for W&B logging.
        """
        self.config = config
        self.hook_manager = hook_manager
        self.state_manager = state_manager
        self.is_logging_rank = is_logging_rank
        self._dist_optimizer = None
        self._dp_group = None
        self._split_info = None
        self._linear_entries = None

    def snapshot_weights(self, model: nn.Module) -> None:
        """Snapshot current weights before the training step.

        Called at the start of a logging iteration so that delta_W and angular
        metrics measure the single-step weight change. Only the logging rank
        needs the snapshot (delta/angular metrics are logging-rank-only).

        Args:
            model: The model whose weights to snapshot.
        """
        if self.is_logging_rank and (self.config.log_relative_updates or self.config.log_angular_metrics):
            self.state_manager.snapshot_weights(model)

    def bind_optimizer(self, optimizer) -> None:
        """Bind optimizer for distributed gradient metric computation.

        For ChainedOptimizer, finds the first DistributedOptimizer instance.
        Always stores the DP group (needed for activation stat reduction even
        without distributed optimizer).

        Args:
            optimizer: The training optimizer (ChainedOptimizer or similar).
        """
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.parallel_state import get_data_parallel_group

        if hasattr(optimizer, 'chained_optimizers'):
            for opt in optimizer.chained_optimizers:
                if isinstance(opt, DistributedOptimizer):
                    self._dist_optimizer = opt
                    self._dp_group = opt.data_parallel_group
                    break
        elif isinstance(optimizer, DistributedOptimizer):
            self._dist_optimizer = optimizer
            self._dp_group = optimizer.data_parallel_group

        # Always store the DP group for activation stat reduction
        if self._dp_group is None:
            try:
                self._dp_group = get_data_parallel_group()
            except AssertionError:
                pass  # parallel state not initialized (e.g., single-GPU)

    def log_internals(
        self,
        model: nn.Module,
        iteration: int,
        wandb_writer: Any,
    ) -> None:
        """Main entry point for logging model internals.

        All DP ranks must call this method — gradient metrics and activation
        statistics involve all-reduce collectives across the DP group.
        Only the logging rank writes to W&B.

        Args:
            model: The model to analyze (should have hooks registered).
            iteration: Current training iteration.
            wandb_writer: W&B writer instance (None on non-logging ranks).
        """
        if self._split_info is None:
            self._split_info, self._linear_entries = _build_model_info(model)

        metrics: Dict[str, float] = {}

        # 1. Activation statistics (all DP ranks participate for reduction)
        if self.config.log_activation_stats:
            activation_metrics = self._compute_activation_metrics()
            metrics.update(activation_metrics)

        # 2. Gradient statistics
        if self.config.log_gradient_stats:
            if self._dist_optimizer is not None:
                # Distributed optimizer: all ranks participate in all-reduce
                gradient_metrics = self._compute_gradient_metrics_distributed(model)
            elif self.is_logging_rank:
                # Non-distributed: only logging rank computes (grads are all-reduced)
                gradient_metrics = self._compute_gradient_metrics(model)
            else:
                gradient_metrics = {}
            metrics.update(gradient_metrics)

        # 3. Relative weight updates (logging rank only, weights are all-gathered)
        if self.config.log_relative_updates and self.is_logging_rank:
            weight_delta_metrics = self.state_manager.compute_weight_deltas(
                model, self._split_info,
            )
            metrics.update(weight_delta_metrics)

        # 4. Angular metrics
        if self.config.log_angular_metrics:
            # Angular updates: logging rank only (uses param.data, identical on all ranks)
            if self.is_logging_rank:
                angular_metrics = self.state_manager.compute_angular_updates(model)
                metrics.update(angular_metrics)

            # Gradient-weight alignment: needs gradient data
            if self._dist_optimizer is not None:
                alignment_metrics = self._compute_gradient_weight_alignment_distributed(model)
            elif self.is_logging_rank:
                alignment_metrics = self._compute_gradient_weight_alignment(model)
            else:
                alignment_metrics = {}
            metrics.update(alignment_metrics)

        # 5. Delta Y — all DP ranks must participate (DP reduction)
        if self.config.log_delta_y:
            delta_y_metrics = self._compute_delta_y_metrics(model)
            metrics.update(delta_y_metrics)

        # Log all metrics to W&B (logging rank only)
        if self.is_logging_rank and metrics and wandb_writer is not None:
            metrics = prettify_metric_keys(metrics)
            wandb_writer.log(metrics, step=iteration)

        # Clear captured data and weight snapshot to free memory
        self.hook_manager.clear_captured_data()
        self.state_manager.clear()

    def _compute_activation_metrics(self) -> Dict[str, float]:
        """Compute activation statistics with DP reduction.

        Each DP rank calls compute_activation_stats on its local micro-batch.
        For DP > 1 the per-rank results are all-reduced (AVG for mean/std/
        kurtosis, MIN/MAX for extremes). This is exact when every rank has
        the same number of activation vectors (uniform micro-batch size).

        ALL DP ranks must call this method when DP > 1.
        Only the logging rank returns populated metrics.

        Returns:
            Dictionary mapping metric names to values.
        """
        captured = self.hook_manager.captured_activations
        # Keys are (module_name, layer_num) tuples, e.g. ("layer", 0), ("attention", 0), ("mlp", 0)
        sorted_keys = sorted(captured.keys())

        dp_world_size = 1
        if self._dp_group is not None:
            dp_world_size = torch.distributed.get_world_size(group=self._dp_group)

        if dp_world_size <= 1:
            if not self.is_logging_rank:
                return {}
            metrics = {}
            for (module_name, layer_num), activation in captured.items():
                stats = compute_activation_stats(activation)
                for stat_name, value in stats.items():
                    metrics[f'activations/{stat_name}/{module_name}/layer_{layer_num:02d}'] = value
            return metrics

        n_entries = len(sorted_keys)
        if n_entries == 0:
            return {}

        # Each rank computes local per-vector stats, then we all-reduce.
        # Pack: 6 AVG-reducible stats per entry (mean, var, kurtosis, rms, min, max)
        A = 6
        avg_stats = torch.zeros(n_entries * A, dtype=torch.float64,
                                device=torch.cuda.current_device())

        for idx, key in enumerate(sorted_keys):
            stats = compute_activation_stats(captured[key])
            off = idx * A
            avg_stats[off + 0] = stats['mean']
            avg_stats[off + 1] = stats['var']
            avg_stats[off + 2] = stats['kurtosis']
            avg_stats[off + 3] = stats['rms']
            avg_stats[off + 4] = stats['min']
            avg_stats[off + 5] = stats['max']

        torch.distributed.all_reduce(avg_stats, op=torch.distributed.ReduceOp.AVG, group=self._dp_group)

        if not self.is_logging_rank:
            return {}

        metrics = {}
        for idx, (module_name, layer_num) in enumerate(sorted_keys):
            off = idx * A
            metrics[f'activations/mean/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 0].item()
            metrics[f'activations/var/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 1].item()
            metrics[f'activations/kurtosis/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 2].item()
            metrics[f'activations/rms/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 3].item()
            metrics[f'activations/min/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 4].item()
            metrics[f'activations/max/{module_name}/layer_{layer_num:02d}'] = avg_stats[off + 5].item()

        return metrics

    def _compute_gradient_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compute per-layer gradient statistics (non-distributed path).

        For non-distributed optimizer, gradients are all-reduced across DP ranks
        so every rank has the same full gradient. Only the logging rank calls this.

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to gradient statistics.
        """
        metrics = {}
        layer_grad_norms: Dict[int, List[float]] = defaultdict(list)

        for name, param in model.named_parameters():
            if get_param_grad(param) is not None:
                grad_norm = compute_gradient_norm(param)

                clean_name = name.replace('.', '/')
                metrics[f'gradients/norm/{clean_name}'] = grad_norm

                layer_idx = _extract_layer_index(name)
                if layer_idx is not None:
                    layer_grad_norms[layer_idx].append(grad_norm)

        self._add_layer_aggregate_metrics(metrics, layer_grad_norms)
        return metrics

    def _compute_gradient_metrics_distributed(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient metrics with distributed optimizer.

        Each DP rank computes ||grad_shard||^2 for its owned portion of each
        parameter's gradient. These partial squared norms are batched into a
        single tensor and all-reduced (SUM) across the DP group.

        ALL DP ranks must call this method.
        Only the logging rank returns populated metrics; other ranks return {}.

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to gradient statistics.
        """
        # Build ordered list of (name, param, param_range) for params in the optimizer
        param_entries = self._get_distributed_param_entries(model)

        if not param_entries:
            return {}

        # Compute local partial squared norms and batch into single tensor.
        # Params not on this rank's shard stay at 0.
        local_norm_sqs = torch.zeros(len(param_entries), dtype=torch.float32,
                                     device=torch.cuda.current_device())
        for i, (name, param, param_range) in enumerate(param_entries):
            if param_range is not None:
                local_norm_sqs[i] = compute_grad_shard_norm_sq(param, param_range).squeeze()

        # Single all-reduce for all parameters at once
        torch.distributed.all_reduce(
            local_norm_sqs,
            op=torch.distributed.ReduceOp.SUM,
            group=self._dp_group,
        )

        if not self.is_logging_rank:
            return {}

        # Assemble per-parameter norms and layer aggregates
        metrics = {}
        layer_grad_norms: Dict[int, List[float]] = defaultdict(list)

        for i, (name, param, param_range) in enumerate(param_entries):
            grad_norm = local_norm_sqs[i].sqrt().item()

            clean_name = name.replace('.', '/')
            metrics[f'gradients/norm/{clean_name}'] = grad_norm

            layer_idx = _extract_layer_index(name)
            if layer_idx is not None:
                layer_grad_norms[layer_idx].append(grad_norm)

        self._add_layer_aggregate_metrics(metrics, layer_grad_norms)
        return metrics

    def _compute_gradient_weight_alignment(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient-weight alignment (non-distributed path).

        Measures whether gradients point along weights (radial) or orthogonal (tangential).

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to alignment statistics.
        """
        metrics = {}
        layer_alignments: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {'cos': [], 'radial': [], 'tangential': []}
        )

        for name, param in model.named_parameters():
            if get_param_grad(param) is not None:
                align_stats = compute_gradient_weight_alignment(param)

                clean_name = name.replace('.', '/')
                metrics[f'grad_weight_align/cos/{clean_name}'] = align_stats['cos_alignment']
                metrics[f'grad_weight_align/radial/{clean_name}'] = align_stats['radial_component']
                metrics[f'grad_weight_align/tangential/{clean_name}'] = align_stats['tangential_component']

                layer_idx = _extract_layer_index(name)
                if layer_idx is not None:
                    layer_alignments[layer_idx]['cos'].append(align_stats['cos_alignment'])
                    layer_alignments[layer_idx]['radial'].append(align_stats['radial_component'])
                    layer_alignments[layer_idx]['tangential'].append(align_stats['tangential_component'])

        self._add_layer_alignment_metrics(metrics, layer_alignments)
        return metrics

    def _compute_gradient_weight_alignment_distributed(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient-weight alignment with distributed optimizer.

        Requires two quantities per parameter that need all-reduce:
        - ||grad||^2 (from grad shard)
        - grad . weight (from shards)
        Plus ||weight|| which is computed locally (weights are all-gathered).

        We pack both into a [2*N] tensor and do one all-reduce.

        ALL DP ranks must call this method.

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to alignment statistics.
        """
        param_entries = self._get_distributed_param_entries(model)

        if not param_entries:
            return {}

        n = len(param_entries)
        # Pack: [grad_norm_sq_0, ..., grad_norm_sq_{n-1}, dot_0, ..., dot_{n-1}]
        local_stats = torch.zeros(2 * n, dtype=torch.float32,
                                  device=torch.cuda.current_device())

        for i, (name, param, param_range) in enumerate(param_entries):
            if param_range is not None:
                local_stats[i] = compute_grad_shard_norm_sq(param, param_range).squeeze()
                local_stats[n + i] = compute_grad_weight_dot_shard(param, param_range).squeeze()

        # Single all-reduce for both quantities
        torch.distributed.all_reduce(
            local_stats,
            op=torch.distributed.ReduceOp.SUM,
            group=self._dp_group,
        )

        if not self.is_logging_rank:
            return {}

        metrics = {}
        layer_alignments: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {'cos': [], 'radial': [], 'tangential': []}
        )

        for i, (name, param, param_range) in enumerate(param_entries):
            grad_norm_sq = local_stats[i].item()
            dot_product = local_stats[n + i].item()

            grad_norm = grad_norm_sq ** 0.5
            weight_norm = param.data.float().norm().item()

            if grad_norm < 1e-10 or weight_norm < 1e-10:
                cos_align = 0.0
                radial = 0.0
                tangential = 0.0
            else:
                cos_align = dot_product / (grad_norm * weight_norm)
                cos_align = max(-1.0, min(1.0, cos_align))
                radial = grad_norm * cos_align
                tangential = grad_norm * max(0.0, 1.0 - cos_align ** 2) ** 0.5

            clean_name = name.replace('.', '/')
            metrics[f'grad_weight_align/cos/{clean_name}'] = cos_align
            metrics[f'grad_weight_align/radial/{clean_name}'] = radial
            metrics[f'grad_weight_align/tangential/{clean_name}'] = tangential

            layer_idx = _extract_layer_index(name)
            if layer_idx is not None:
                layer_alignments[layer_idx]['cos'].append(cos_align)
                layer_alignments[layer_idx]['radial'].append(radial)
                layer_alignments[layer_idx]['tangential'].append(tangential)

        self._add_layer_alignment_metrics(metrics, layer_alignments)
        return metrics

    def _compute_delta_y_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compute delta_Y metrics by re-running linear layers with stored inputs.

        For each linear layer with captured (input, output) from the forward pass,
        re-runs the layer with the stored input and updated weights, then computes
        ||Y_new - Y_old|| / ||Y_old||.

        For fused projections that pack distinct operations into one matmul,
        the output is split before computing delta_Y per component:
        - linear_qkv: split into Q, K, V (interleaved by query group)
        - linear_fc1 with SwiGLU: split into gate and up

        ALL TP ranks must call this method because the linear layer forward
        involves TP collective communication.

        For DP > 1, squared norms are all-reduced (SUM) across the DP group
        so that the final ratio is computed as if over one large concatenated
        batch:  delta_Y = sqrt(sum_r ||dY_r||^2) / sqrt(sum_r ||Y_r||^2).

        Args:
            model: The model with updated weights.

        Returns:
            Dictionary mapping metric names to delta_Y values.
        """
        captured = self.hook_manager.captured_linear_io
        if not captured:
            return {}

        # First pass: re-run linear layers and collect squared norms.
        # entries: list of (metric_key, delta_sq, old_sq)
        entries: List[Tuple[str, float, float]] = []

        for key in sorted(captured.keys()):
            linear_name, layer_number = key
            entry = self._linear_entries.get(key)
            if entry is None:
                continue

            linear_module, _ = entry
            stored_input, stored_output = captured[key]

            with torch.no_grad():
                new_out = _rerun_linear(linear_module, stored_input)

            info = (
                self._split_info.get(id(linear_module.weight))
                if hasattr(linear_module, 'weight') else None
            )

            if info is not None and info['type'] == 'qkv':
                qkv_args = (
                    info['num_query_groups'],
                    info['num_query_heads_per_group'],
                    info['head_dim'],
                )
                new_comps = _split_qkv_dim(new_out, *qkv_args, qkv_dim=-1)
                old_comps = _split_qkv_dim(stored_output, *qkv_args, qkv_dim=-1)
                for name, nc, oc in zip(_QKV_NAMES, new_comps, old_comps):
                    entries.append((
                        f'delta_Y/{name}/layer_{layer_number:02d}',
                        *_mean_squared_norm_change(nc, oc),
                    ))

            elif info is not None and info['type'] == 'swiglu':
                new_gate, new_up = torch.chunk(new_out, 2, dim=-1)
                old_gate, old_up = torch.chunk(stored_output, 2, dim=-1)
                for comp_name, nc, oc in [('gate', new_gate, old_gate),
                                          ('up', new_up, old_up)]:
                    entries.append((
                        f'delta_Y/{comp_name}/layer_{layer_number:02d}',
                        *_mean_squared_norm_change(nc, oc),
                    ))

            else:
                entries.append((
                    f'delta_Y/{linear_name}/layer_{layer_number:02d}',
                    *_mean_squared_norm_change(new_out, stored_output),
                ))

        if not entries:
            return {}

        # DP reduction: sum squared norms across DP ranks so the metric
        # reflects the full batch (all micro-batches concatenated).
        dp_world_size = 1
        if self._dp_group is not None:
            dp_world_size = torch.distributed.get_world_size(group=self._dp_group)

        n = len(entries)

        if dp_world_size > 1:
            # Pack [delta_msq_0, ..., delta_msq_{n-1}, old_msq_0, ..., old_msq_{n-1}]
            local_msqs = torch.zeros(
                2 * n, dtype=torch.float64, device=torch.cuda.current_device(),
            )
            for i, (_, delta_msq, old_msq) in enumerate(entries):
                local_msqs[i] = delta_msq
                local_msqs[n + i] = old_msq

            torch.distributed.all_reduce(
                local_msqs, op=torch.distributed.ReduceOp.AVG, group=self._dp_group,
            )

            if not self.is_logging_rank:
                return {}

            metrics = {}
            for i, (metric_key, _, _) in enumerate(entries):
                delta_msq = local_msqs[i].item()
                old_msq = local_msqs[n + i].item()
                if old_msq > 1e-20:
                    metrics[metric_key] = (delta_msq ** 0.5) / (old_msq ** 0.5)
                else:
                    metrics[metric_key] = 0.0
            return metrics

        # Single DP rank — compute ratios directly.
        if not self.is_logging_rank:
            return {}

        metrics = {}
        for metric_key, delta_msq, old_msq in entries:
            if old_msq > 1e-20:
                metrics[metric_key] = (delta_msq ** 0.5) / (old_msq ** 0.5)
            else:
                metrics[metric_key] = 0.0
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_distributed_param_entries(self, model: nn.Module):
        """Build ordered list of (name, param, param_range) for distributed optimizer.

        Uses model.named_parameters() as the canonical ordering so ALL DP ranks
        produce lists of the same length (required for batched all-reduce).
        For parameters not in this rank's shard, param_range is None and the
        rank contributes zero to the all-reduce.

        Returns:
            List of (name, param, param_range) tuples. param_range is a Range
            object with .start/.end into the flattened parameter tensor, or
            None if this rank doesn't own a shard of this parameter.
        """
        entries = []
        for name, param in model.named_parameters():
            if get_param_grad(param) is None:
                continue
            # Check if this param has a shard on this rank
            if param in self._dist_optimizer.model_param_gbuf_map:
                param_range_map = self._dist_optimizer._get_model_param_range_map(param)
                param_range = param_range_map["param"]
            else:
                param_range = None  # not on this rank, will contribute 0
            entries.append((name, param, param_range))
        return entries

    @staticmethod
    def _add_layer_aggregate_metrics(
        metrics: Dict[str, float],
        layer_grad_norms: Dict[int, List[float]],
    ) -> None:
        """Add per-layer aggregate gradient metrics and gradient flow ratios."""
        for layer_idx, norms in layer_grad_norms.items():
            if norms:
                total_norm = (sum(n ** 2 for n in norms)) ** 0.5
                avg_norm = sum(norms) / len(norms)
                max_norm = max(norms)

                metrics[f'per_layer_gradients/total_norm/layer_{layer_idx:02d}'] = total_norm
                metrics[f'per_layer_gradients/avg_norm/layer_{layer_idx:02d}'] = avg_norm
                metrics[f'per_layer_gradients/max_norm/layer_{layer_idx:02d}'] = max_norm

        sorted_layers = sorted(layer_grad_norms.keys())
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            norms_l1 = layer_grad_norms[l1]
            norms_l2 = layer_grad_norms[l2]

            if norms_l1 and norms_l2:
                total_l1 = (sum(n ** 2 for n in norms_l1)) ** 0.5
                total_l2 = (sum(n ** 2 for n in norms_l2)) ** 0.5

                if total_l1 > 1e-10:
                    metrics[f'gradient_flow/layer_{l1:02d}_to_{l2:02d}'] = total_l2 / total_l1

    @staticmethod
    def _add_layer_alignment_metrics(
        metrics: Dict[str, float],
        layer_alignments: Dict[int, Dict[str, List[float]]],
    ) -> None:
        """Add per-layer aggregate alignment metrics."""
        for layer_idx, stats in layer_alignments.items():
            if stats['cos']:
                metrics[f'per_layer_grad_weight_align/cos/layer_{layer_idx:02d}'] = (
                    sum(stats['cos']) / len(stats['cos'])
                )
                metrics[f'per_layer_grad_weight_align/radial/layer_{layer_idx:02d}'] = (
                    sum(stats['radial']) / len(stats['radial'])
                )
                metrics[f'per_layer_grad_weight_align/tangential/layer_{layer_idx:02d}'] = (
                    sum(stats['tangential']) / len(stats['tangential'])
                )
