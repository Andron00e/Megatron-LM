# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""State management for tracking previous values across iterations."""

from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn

from .metrics import (
    compute_weight_delta_per_neuron,
    compute_weight_delta_qkv_split,
    compute_weight_delta_swiglu_split,
    compute_angular_update,
    _extract_layer_index,
)


class InternalsStateManager:
    """Manages state for computing delta metrics across training iterations.

    This class caches previous weights to compute relative changes (delta_W)
    between iterations. Weights are stored on CPU to conserve GPU memory.

    Attributes:
        previous_weights: Dictionary mapping parameter names to their previous values.
        initialized: Whether the first iteration's state has been captured.
        weights_on_gpu: Whether to keep previous weights on GPU (faster but uses more memory).
    """

    def __init__(self, weights_on_gpu: bool = False):
        """Initialize the state manager with empty caches.

        Args:
            weights_on_gpu: If True, keep previous weights on GPU to avoid
                GPU-CPU transfer overhead. Uses more GPU memory but eliminates
                performance variance from PCIe transfers.
        """
        self.previous_weights: Dict[str, Tensor] = {}
        self.weights_on_gpu = weights_on_gpu

    def snapshot_weights(self, model: nn.Module) -> None:
        """Snapshot current weights before the training step.

        Called at the start of a logging iteration (before forward/backward/
        optimizer) so that delta metrics measure the single-step change.
        The snapshot is cleared after logging via clear().

        Args:
            model: The model whose weights to snapshot.
        """
        self.previous_weights.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                cloned = param.data.detach().clone()
                if not self.weights_on_gpu:
                    cloned = cloned.cpu()
                self.previous_weights[name] = cloned

    def compute_weight_deltas(
        self, model: nn.Module, split_info: Dict[int, dict],
    ) -> Dict[str, float]:
        """Compute relative weight updates for all cached parameters.

        For fused projections that pack distinct operations into one weight matrix,
        the weight is split before computing delta_W per component:
        - linear_qkv: split into Q, K, V (interleaved by query group)
        - linear_fc1 with SwiGLU: split into gate and up

        Args:
            model: The model with current weights.
            split_info: Pre-computed mapping from id(param) to split config,
                built once by the logger via _build_model_info().

        Returns:
            Dictionary mapping metric names to relative weight changes.
        """
        if not self.previous_weights:
            return {}

        metrics = {}
        for name, param in model.named_parameters():
            if name not in self.previous_weights:
                continue

            previous = self.previous_weights[name]
            clean_name = name.replace('.', '/')
            info = split_info.get(id(param))

            if info is not None and info['type'] == 'qkv':
                comp_stats = compute_weight_delta_qkv_split(
                    param.data, previous,
                    num_query_groups=info['num_query_groups'],
                    num_query_heads_per_group=info['num_query_heads_per_group'],
                    head_dim=info['head_dim'],
                )
                for comp, stats in comp_stats.items():
                    comp_name = clean_name.replace('linear_qkv', comp)
                    metrics[f'delta_W/{comp_name}'] = stats['full']
                    if 'per_neuron_mean' in stats:
                        metrics[f'delta_W_per_neuron/mean/{comp_name}'] = stats['per_neuron_mean']

            elif info is not None and info['type'] == 'swiglu':
                comp_stats = compute_weight_delta_swiglu_split(param.data, previous)
                for comp, stats in comp_stats.items():
                    comp_name = clean_name.replace('linear_fc1', comp)
                    metrics[f'delta_W/{comp_name}'] = stats['full']
                    if 'per_neuron_mean' in stats:
                        metrics[f'delta_W_per_neuron/mean/{comp_name}'] = stats['per_neuron_mean']

            else:
                delta_stats = compute_weight_delta_per_neuron(param.data, previous)
                metrics[f'delta_W/{clean_name}'] = delta_stats['full']

                # Log per-neuron stats for weight matrices (2D+)
                if param.dim() >= 2:
                    metrics[f'delta_W_per_neuron/mean/{clean_name}'] = delta_stats['per_neuron_mean']
                    # metrics[f'delta_W_per_neuron/std/{clean_name}'] = delta_stats['per_neuron_std']
                    # metrics[f'delta_W_per_neuron/max/{clean_name}'] = delta_stats['per_neuron_max']
                    # metrics[f'delta_W_per_neuron/min/{clean_name}'] = delta_stats['per_neuron_min']

        #         # Aggregate by layer for summary metrics
        #         if 'layers' in name:
        #             # Extract layer index from name like 'decoder.layers.0.self_attention.linear_qkv.weight'
        #             parts = name.split('.')
        #             for i, part in enumerate(parts):
        #                 if part == 'layers' and i + 1 < len(parts):
        #                     try:
        #                         layer_idx = int(parts[i + 1])
        #                         if layer_idx not in layer_deltas:
        #                             layer_deltas[layer_idx] = []
        #                             layer_per_neuron_stats[layer_idx] = {
        #                                 'means': [], 'stds': [], 'maxs': [], 'mins': []
        #                             }
        #                         layer_deltas[layer_idx].append(delta_stats['full'])
        #                         if param.dim() >= 2:
        #                             layer_per_neuron_stats[layer_idx]['means'].append(delta_stats['per_neuron_mean'])
        #                             # layer_per_neuron_stats[layer_idx]['stds'].append(delta_stats['per_neuron_std'])
        #                             # layer_per_neuron_stats[layer_idx]['maxs'].append(delta_stats['per_neuron_max'])
        #                             # layer_per_neuron_stats[layer_idx]['mins'].append(delta_stats['per_neuron_min'])
        #                     except ValueError:
        #                         pass
        #                     break

        return metrics

    def compute_angular_updates(self, model: nn.Module) -> Dict[str, float]:
        """Compute angular changes (direction changes) for all cached parameters.

        Useful for spherical/normalized training analysis (nGPT, EDM2).

        Args:
            model: The model with current weights.

        Returns:
            Dictionary mapping metric names to angular change values.
        """
        if not self.previous_weights:
            return {}

        metrics = {}
        layer_angular = {}  # For aggregating per-layer metrics

        for name, param in model.named_parameters():
            if name in self.previous_weights:
                angular_stats = compute_angular_update(param.data, self.previous_weights[name])

                # Create a clean metric name
                clean_name = name.replace('.', '/')
                metrics[f'angular/cos_similarity/{clean_name}'] = angular_stats['cos_similarity']
                metrics[f'angular/degrees/{clean_name}'] = angular_stats['angular_change_degrees']

                layer_idx = _extract_layer_index(name)
                if layer_idx is not None:
                    if layer_idx not in layer_angular:
                        layer_angular[layer_idx] = {
                            'cos_sims': [],
                            'degrees': [],
                        }
                    layer_angular[layer_idx]['cos_sims'].append(angular_stats['cos_similarity'])
                    layer_angular[layer_idx]['degrees'].append(angular_stats['angular_change_degrees'])

        # Compute per-layer aggregate angular metrics
        for layer_idx, stats in layer_angular.items():
            if stats['degrees']:
                avg_degrees = sum(stats['degrees']) / len(stats['degrees'])
                max_degrees = max(stats['degrees'])
                avg_cos = sum(stats['cos_sims']) / len(stats['cos_sims'])
                metrics[f'per_layer_angular/degrees/layer_{layer_idx:02d}'] = avg_degrees
                metrics[f'per_layer_angular/max_degrees/layer_{layer_idx:02d}'] = max_degrees
                metrics[f'per_layer_angular/cos_similarity/layer_{layer_idx:02d}'] = avg_cos

        return metrics

    def clear(self) -> None:
        """Clear all cached state to free memory."""
        self.previous_weights.clear()
